"""Run FCI while recording every executed CI test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from graphs import from_causallearn


@dataclass(frozen=True)
class CITest:
    a: int
    b: int
    S: Tuple[int, ...]
    removed: bool
    sign: int
    rho_c: float
    r_obs: float


@dataclass
class FCIRun:
    pag: np.ndarray
    tests: List[CITest]
    alpha0: float
    n: int


def _partial_corr(corr: np.ndarray, a: int, b: int, S) -> float:
    var = [a, b] + list(S)
    sub = corr[np.ix_(var, var)]
    inv = np.linalg.inv(sub)
    r = -inv[0, 1] / np.sqrt(abs(inv[0, 0] * inv[1, 1]))
    if abs(r) >= 1:
        r = (1.0 - np.finfo(float).eps) * np.sign(r)
    return float(r)


def run_fci_recorded(Y: np.ndarray, alpha0: float,
                     verbose: bool = False) -> FCIRun:
    """Standard FCI with CI-test recording."""
    from causallearn.utils import cit as cit_mod
    from causallearn.search.ConstraintBased.FCI import fci

    n, d = Y.shape
    queried = {}
    orig_call = cit_mod.FisherZ.__call__

    def wrapped(self, X, Yv, condition_set=None):
        p = orig_call(self, X, Yv, condition_set)
        S = tuple(sorted(condition_set)) if condition_set else ()
        a, b = (X, Yv) if X < Yv else (Yv, X)
        queried[(a, b, S)] = p
        return p

    cit_mod.FisherZ.__call__ = wrapped
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()) if not verbose \
                else contextlib.nullcontext():
            g, _ = fci(Y, independence_test_method="fisherz", alpha=alpha0,
                       verbose=verbose, show_progress=False)
    finally:
        cit_mod.FisherZ.__call__ = orig_call

    corr = np.corrcoef(Y.T)
    c = norm.ppf(1.0 - alpha0 / 2.0)
    tests = []
    for (a, b, S), p in queried.items():
        r = _partial_corr(corr, a, b, S)
        rho_c = float(np.tanh(c / np.sqrt(n - len(S) - 3)))
        removed = p > alpha0
        if not (removed == (abs(r) < rho_c) or abs(abs(r) - rho_c) < 1e-12):
            raise RuntimeError(
                f"CI-test decision mismatch: {(a, b, S, p, r, rho_c)}")
        tests.append(CITest(a=a, b=b, S=S, removed=removed,
                            sign=int(np.sign(r)) if r != 0 else 1,
                            rho_c=rho_c, r_obs=r))
    return FCIRun(pag=from_causallearn(g), tests=tests, alpha0=alpha0, n=n)
