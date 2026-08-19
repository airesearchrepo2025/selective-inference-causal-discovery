"""Generalized adjustment criterion and the canonical adjustment set."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from graphs import (
    ARROW,
    TAIL,
    adjacent,
    ancestors,
    edge_visible,
    is_directed,
    m_separated,
    poss_an,
    poss_de,
    representative_mag,
)


@dataclass
class AdjustmentResult:
    identifiable: bool
    reason: str
    adjustment_set: List[int] = field(default_factory=list)
    amenable: bool = False
    peo_valid: bool = True


def gac_adjustment(E: np.ndarray, i: int, j: int, kind: str) -> AdjustmentResult:
    """Identifiability and canonical adjustment set for beta_{i->j}."""
    if i == j:
        return AdjustmentResult(False, "i == j")
    d = E.shape[0]

    r1 = poss_de(E, [i])
    r2 = poss_an(E, [j], exclude=[i])

    amenable = True
    for v in adjacent(E, i):
        if E[i, v] != ARROW:
            if v in r2:
                if not (is_directed(E, i, v) and edge_visible(E, i, v, kind)):
                    amenable = False
                    break
    if not amenable:
        return AdjustmentResult(False, "not amenable", amenable=False)

    m_set = (r1 & r2) - {i}
    forb = poss_de(E, sorted(m_set)) if m_set else set()

    adj_set = sorted(poss_an(E, [i, j]) - {i, j} - forb)

    M, peo_valid = representative_mag(E, kind)
    Mno_i = M.copy()
    Mno_i[i, :] = 0
    Mno_i[:, i] = 0
    an_j = ancestors(Mno_i, [j])
    M_pbd = M.copy()
    for v in adjacent(M, i):
        if is_directed(M, i, v) and (v == j or v in an_j):
            M_pbd[i, v] = 0
            M_pbd[v, i] = 0
    blocked = m_separated(M_pbd, i, j, adj_set)
    if not blocked:
        return AdjustmentResult(False, "Adjust set fails blocking",
                                amenable=True, peo_valid=peo_valid)
    return AdjustmentResult(True, "ok", adjustment_set=adj_set,
                            amenable=True, peo_valid=peo_valid)


def identifiable_pairs_random_order(E: np.ndarray, kind: str, rng) -> tuple:
    """Return the first identifiable (i, j) pair in a uniformly random order."""
    d = E.shape[0]
    pairs = [(a, b) for a in range(d) for b in range(d) if a != b]
    order = rng.permutation(len(pairs))
    for idx in order:
        a, b = pairs[idx]
        res = gac_adjustment(E, a, b, kind)
        if res.identifiable:
            return (a, b), res
    return None, None
