"""Recording implementation of GES with BIC scoring."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from graphs import ARROW, NO, TAIL, adjacent, is_directed, is_undirected


def und_neighbors(E, v):
    return [u for u in adjacent(E, v) if is_undirected(E, u, v)]


def _is_clique(E, nodes) -> bool:
    nodes = list(nodes)
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if E[nodes[a], nodes[b]] == NO:
                return False
    return True


def _semi_directed_blocked(E, y: int, x: int, blocked) -> bool:
    blk = set(blocked)
    seen = {y}
    stack = [y]
    while stack:
        u = stack.pop()
        for w in adjacent(E, u):
            if w in seen or w in blk:
                continue
            if is_directed(E, u, w) or is_undirected(E, u, w):
                if w == x:
                    return False
                seen.add(w)
                stack.append(w)
    return True


def pdag_to_dag(E: np.ndarray) -> np.ndarray:
    """Consistent extension of a PDAG."""
    d = E.shape[0]
    W = E.copy()
    out = E.copy()
    alive = set(range(d))
    while alive:
        found = None
        for x in alive:
            if any(is_directed(W, x, w) for w in adjacent(W, x)):
                continue
            nbu = [u for u in adjacent(W, x) if is_undirected(W, u, x)]
            adj_x = set(adjacent(W, x))
            ok = all(all(W[u, w] != NO for w in adj_x if w != u) for u in nbu)
            if ok:
                found = x
                break
        if found is None:
            raise ValueError("PDAG admits no consistent extension")
        for u in list(adjacent(W, found)):
            if is_undirected(W, u, found):
                out[u, found] = TAIL
                out[found, u] = ARROW
            W[u, found] = NO
            W[found, u] = NO
        alive.discard(found)
    return out


def dag_to_cpdag(D: np.ndarray) -> np.ndarray:
    """Skeleton + v-structures, then Meek rules R1-R3."""
    d = D.shape[0]
    E = np.zeros_like(D)
    for a in range(d):
        for b in range(d):
            if D[a, b] != NO:
                E[a, b] = TAIL
                E[b, a] = TAIL
    for c in range(d):
        pa = [u for u in range(d) if is_directed(D, u, c)]
        for a in range(len(pa)):
            for b in range(a + 1, len(pa)):
                if D[pa[a], pa[b]] == NO:
                    for u in (pa[a], pa[b]):
                        E[u, c] = TAIL
                        E[c, u] = ARROW
    changed = True
    while changed:
        changed = False
        for a in range(d):
            for b in range(d):
                if not is_undirected(E, a, b):
                    continue
                if any(is_directed(E, c, a) and E[c, b] == NO
                       for c in range(d)):
                    E[a, b], E[b, a] = TAIL, ARROW
                    changed = True
                    continue
                if any(is_directed(E, a, c) and is_directed(E, c, b)
                       for c in range(d)):
                    E[a, b], E[b, a] = TAIL, ARROW
                    changed = True
                    continue
                cand = [c for c in range(d)
                        if is_undirected(E, a, c) and is_directed(E, c, b)]
                if any(E[c1, c2] == NO
                       for c1 in cand for c2 in cand if c1 < c2):
                    E[a, b], E[b, a] = TAIL, ARROW
                    changed = True
    return E


class DetRegistry:
    def __init__(self):
        self.idx: Dict[FrozenSet[int], int] = {}
        self.sets: List[FrozenSet[int]] = []

    def get(self, s: FrozenSet[int]) -> Optional[int]:
        if not s:
            return None
        if s not in self.idx:
            self.idx[s] = len(self.sets)
            self.sets.append(s)
        return self.idx[s]


@dataclass
class Constraint:
    terms: List[Tuple[int, float]]
    const: float
    geq: bool


@dataclass
class GESRun:
    cpdag: np.ndarray
    registry: DetRegistry
    constraints: List[Constraint]
    path: List[tuple]
    n: int
    score: float
    rows: Optional[np.ndarray] = None


def _delta_terms(reg: DetRegistry, y: int, before: FrozenSet[int],
                 after: FrozenSet[int], n: int, lam: float = 2.0):
    terms = []
    for s, c in ((after | {y}, n), (after, -n), (before | {y}, -n), (before, n)):
        k = reg.get(frozenset(s))
        if k is not None:
            terms.append((k, float(c)))
    return terms, float(lam * (len(after) - len(before)) * np.log(n))


class _ScoreCache:
    def __init__(self, K: np.ndarray, n: int, lam: float = 2.0):
        self.K = K
        self.n = n
        self.lam = lam
        self.cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def logcondvar(self, y: int, P: FrozenSet[int]) -> float:
        key = (y, P)
        if key in self.cache:
            return self.cache[key]
        if not P:
            v = np.log(self.K[y, y])
        else:
            idx = sorted(P)
            KPP = self.K[np.ix_(idx, idx)]
            KPy = self.K[np.ix_(idx, [y])]
            v = np.log(self.K[y, y] - (KPy.T @ np.linalg.solve(KPP, KPy)).item())
        self.cache[key] = v
        return v

    def delta(self, y: int, before: FrozenSet[int],
              after: FrozenSet[int]) -> float:
        return (self.n * (self.logcondvar(y, after) - self.logcondvar(y, before))
                + self.lam * (len(after) - len(before)) * np.log(self.n))


MAX_SUBSET = 14


def _insert_candidates(E, sc: _ScoreCache, max_parents: float):
    d = E.shape[0]
    out = []
    for x in range(d):
        for y in range(d):
            if x == y or E[x, y] != NO:
                continue
            if sum(1 for u in adjacent(E, y) if is_directed(E, u, y)) > max_parents:
                continue
            nb_y = set(und_neighbors(E, y))
            adj_x = set(adjacent(E, x))
            na = frozenset(nb_y & adj_x)
            t0 = sorted(nb_y - adj_x)
            if len(t0) > MAX_SUBSET:
                raise RuntimeError(f"insert T0 too large ({len(t0)})")
            pa = frozenset(u for u in adjacent(E, y) if is_directed(E, u, y))
            for r in range(len(t0) + 1):
                for T in itertools.combinations(t0, r):
                    Ts = frozenset(T)
                    if not _is_clique(E, na | Ts):
                        continue
                    if not _semi_directed_blocked(E, y, x, na | Ts):
                        continue
                    before = frozenset(pa | na | Ts)
                    after = frozenset(before | {x})
                    out.append(("insert", x, y, Ts, before, after,
                                sc.delta(y, before, after)))
    return out


def _delete_candidates(E, sc: _ScoreCache):
    d = E.shape[0]
    out = []
    for x in range(d):
        for y in range(d):
            if x == y:
                continue
            if not (is_directed(E, x, y) or is_undirected(E, x, y)):
                continue
            nb_y = set(und_neighbors(E, y))
            adj_x = set(adjacent(E, x))
            na = sorted(nb_y & adj_x)
            if len(na) > MAX_SUBSET:
                raise RuntimeError(f"delete NA too large ({len(na)})")
            pa = frozenset(u for u in adjacent(E, y) if is_directed(E, u, y))
            for r in range(len(na) + 1):
                for H in itertools.combinations(na, r):
                    Hs = frozenset(H)
                    keep = frozenset(set(na) - Hs)
                    if not _is_clique(E, keep):
                        continue
                    before = frozenset(keep | pa | {x})
                    after = frozenset((keep | pa) - {x})
                    out.append(("delete", x, y, Hs, before, after,
                                sc.delta(y, before, after)))
    return out


def _apply(E, op) -> np.ndarray:
    kind, x, y, S, *_ = op
    W = E.copy()
    if kind == "insert":
        W[x, y], W[y, x] = TAIL, ARROW
        for t in S:
            W[t, y], W[y, t] = TAIL, ARROW
    else:
        W[x, y] = W[y, x] = NO
        for h in S:
            if is_undirected(W, y, h):
                W[y, h], W[h, y] = TAIL, ARROW
            if is_undirected(W, x, h):
                W[x, h], W[h, x] = TAIL, ARROW
    return dag_to_cpdag(pdag_to_dag(W))


def run_ges_recorded(Y: np.ndarray, rows: Optional[np.ndarray] = None,
                     bic_lambda: float = 2.0,
                     max_parents: Optional[float] = None) -> GESRun:
    """Two-phase GES with full comparison recording."""
    Ysel = Y if rows is None else Y[rows]
    n = Ysel.shape[0]
    d = Y.shape[1]
    if max_parents is None:
        max_parents = d / 2
    Yt = Ysel - Ysel.mean(axis=0, keepdims=True)
    K = Yt.T @ Yt
    sc = _ScoreCache(K, n, lam=bic_lambda)
    reg = DetRegistry()
    cons: List[Constraint] = []
    path = []
    E = np.zeros((d, d), dtype=int)
    total = 0.0

    for phase, enum in (("fwd", _insert_candidates), ("bwd", _delete_candidates)):
        while True:
            cands = (enum(E, sc, max_parents) if phase == "fwd"
                     else enum(E, sc))
            if not cands:
                break
            deltas = np.array([c[-1] for c in cands])
            b = int(np.argmin(deltas))
            best = cands[b]
            if best[-1] >= 0:
                for c in cands:
                    terms, const = _delta_terms(reg, c[2], c[4], c[5], n,
                                                lam=bic_lambda)
                    cons.append(Constraint(terms, const, geq=True))
                break
            tb, cb = _delta_terms(reg, best[2], best[4], best[5], n,
                                  lam=bic_lambda)
            cons.append(Constraint(tb, cb, geq=False))
            for k, c in enumerate(cands):
                if k == b:
                    continue
                te, ce = _delta_terms(reg, c[2], c[4], c[5], n,
                                      lam=bic_lambda)
                merged: Dict[int, float] = {}
                for idx, cf in tb:
                    merged[idx] = merged.get(idx, 0.0) + cf
                for idx, cf in te:
                    merged[idx] = merged.get(idx, 0.0) - cf
                terms = [(i, cf) for i, cf in merged.items() if cf != 0.0]
                cons.append(Constraint(terms, cb - ce, geq=False))
            E = _apply(E, best)
            path.append(best[:4])
            total += best[-1]
    return GESRun(cpdag=E, registry=reg, constraints=cons, path=path,
                  n=n, score=total, rows=rows)
