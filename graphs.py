"""Endpoint-matrix graph utilities for PAGs, CPDAGs, MAGs and DAGs."""
from __future__ import annotations

import numpy as np

NO, TAIL, ARROW, CIRCLE = 0, -1, 1, 2


def n_nodes(E: np.ndarray) -> int:
    return E.shape[0]


def has_edge(E, a, b) -> bool:
    return E[a, b] != NO


def adjacent(E, a):
    return np.nonzero(E[a] != NO)[0]


def is_directed(E, a, b) -> bool:
    return E[a, b] == TAIL and E[b, a] == ARROW


def is_bidirected(E, a, b) -> bool:
    return E[a, b] == ARROW and E[b, a] == ARROW


def is_undirected(E, a, b) -> bool:
    return E[a, b] == TAIL and E[b, a] == TAIL


def parents(E, v):
    return [u for u in adjacent(E, v) if is_directed(E, u, v)]


def dag_to_endpoint(adj: np.ndarray) -> np.ndarray:
    d = adj.shape[0]
    E = np.zeros((d, d), dtype=int)
    for u in range(d):
        for v in range(d):
            if adj[u, v]:
                E[u, v] = TAIL
                E[v, u] = ARROW
    return E


def poss_de(E, sources, exclude=()):
    """Possible descendants (reflexive), avoiding exclude."""
    excl = set(exclude)
    seen = set(s for s in sources if s not in excl)
    stack = list(seen)
    while stack:
        u = stack.pop()
        for w in adjacent(E, u):
            if w in seen or w in excl:
                continue
            if E[u, w] in (TAIL, CIRCLE):
                seen.add(w)
                stack.append(w)
    return seen


def poss_an(E, targets, exclude=()):
    """Possible ancestors (reflexive), paths avoiding exclude."""
    excl = set(exclude)
    seen = set(t for t in targets if t not in excl)
    stack = list(seen)
    while stack:
        w = stack.pop()
        for u in adjacent(E, w):
            if u in seen or u in excl:
                continue
            if E[u, w] in (TAIL, CIRCLE):
                seen.add(u)
                stack.append(u)
    return seen


def ancestors(E, targets):
    """Ancestors via directed edges only (reflexive)."""
    seen = set(targets)
    stack = list(seen)
    while stack:
        w = stack.pop()
        for u in adjacent(E, w):
            if u not in seen and is_directed(E, u, w):
                seen.add(u)
                stack.append(u)
    return seen


def descendants_directed(E, sources, exclude=()):
    """Descendants via directed edges only (reflexive), avoiding exclude."""
    excl = set(exclude)
    seen = set(s for s in sources if s not in excl)
    stack = list(seen)
    while stack:
        u = stack.pop()
        for w in adjacent(E, u):
            if w not in seen and w not in excl and is_directed(E, u, w):
                seen.add(w)
                stack.append(w)
    return seen


def edge_visible(E, x, y, kind: str) -> bool:
    """Is the directed edge x -> y visible?"""
    if kind in ("dag", "cpdag"):
        return is_directed(E, x, y)
    if not is_directed(E, x, y):
        return False
    adj_y = set(adjacent(E, y))
    for dnode in adjacent(E, x):
        if E[x, dnode] == ARROW and dnode != y and dnode not in adj_y:
            return True
    pa_y = set(parents(E, y))
    frontier = [q for q in pa_y if is_bidirected(E, q, x)]
    seen = set(frontier)
    while frontier:
        q = frontier.pop()
        for w in adjacent(E, q):
            if E[q, w] == ARROW:
                if w != y and w not in adj_y and w != x:
                    return True
                if w in pa_y and w not in seen and is_bidirected(E, w, q):
                    seen.add(w)
                    frontier.append(w)
    return False


def _mcs_order(comp_adj: dict) -> list:
    nodes = list(comp_adj)
    weight = {v: 0 for v in nodes}
    order = []
    remaining = set(nodes)
    while remaining:
        v = max(remaining, key=lambda u: (weight[u], -u))
        order.append(v)
        remaining.discard(v)
        for w in comp_adj[v]:
            if w in remaining:
                weight[w] += 1
    return order


def representative_mag(E: np.ndarray, kind: str):
    """Return (M, valid_peo) where M is a representative MAG/DAG."""
    d = n_nodes(E)
    M = E.copy()
    if kind in ("dag", "mag"):
        return M, True
    if kind == "pag":
        comp_mark = CIRCLE
        for a in range(d):
            for b in range(d):
                if M[a, b] == CIRCLE and M[b, a] == ARROW:
                    M[a, b] = TAIL
    elif kind == "cpdag":
        comp_mark = TAIL
    else:
        raise ValueError(kind)
    comp_adj = {}
    for a in range(d):
        for b in range(a + 1, d):
            if M[a, b] == comp_mark and M[b, a] == comp_mark and (
                kind == "pag" or not is_directed(E, a, b) and not is_directed(E, b, a)
            ):
                comp_adj.setdefault(a, set()).add(b)
                comp_adj.setdefault(b, set()).add(a)
    if comp_adj:
        order = _mcs_order(comp_adj)
        peo = list(reversed(order))
        pos = {v: k for k, v in enumerate(peo)}
        valid = True
        for k, v in enumerate(peo):
            later = [w for w in comp_adj[v] if pos[w] > k]
            for ii in range(len(later)):
                for jj in range(ii + 1, len(later)):
                    if later[jj] not in comp_adj[later[ii]]:
                        valid = False
        for a in comp_adj:
            for b in comp_adj[a]:
                if pos[a] > pos[b]:
                    M[a, b] = TAIL
                    M[b, a] = ARROW
    else:
        valid = True
    return M, valid


def m_separated(M: np.ndarray, x: int, y: int, Z) -> bool:
    """True iff x and y are m-separated given Z."""
    Zs = set(Z)
    if x == y:
        return False
    anz = ancestors(M, Zs) if Zs else set()
    from collections import deque

    start = []
    for w in adjacent(M, x):
        start.append((w, M[w, x] == ARROW))
    seen = set(start)
    dq = deque(start)
    while dq:
        v, into = dq.popleft()
        if v == y:
            return False
        for w in adjacent(M, v):
            if w == x:
                continue
            out_arrow_at_v = M[v, w] == ARROW
            if into and out_arrow_at_v:
                ok = v in anz
            else:
                ok = v not in Zs
            if ok:
                st = (w, M[w, v] == ARROW)
                if st not in seen:
                    seen.add(st)
                    dq.append(st)
    return True


def from_causallearn(G) -> np.ndarray:
    return np.array(G.graph, dtype=int)
