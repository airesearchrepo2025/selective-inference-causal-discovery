"""Baselines: naive OLS CI and 50/50 sample splitting."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import t as tdist

from truncation import fit_contrast


def naive_ci(Y: np.ndarray, i: int, j: int, S: List[int],
             alpha: float) -> Tuple[float, float, float]:
    """Classical OLS t CI for gamma1. Returns (L, U, gamma1_hat)."""
    con = fit_contrast(Y, i, j, S)
    sd = np.sqrt(con.sigma2_hat * con.norm2)
    q = tdist.ppf(1.0 - alpha / 2.0, con.df)
    return con.gamma1 - q * sd, con.gamma1 + q * sd, con.gamma1


def split_ci_fci(Y: np.ndarray, i: int, j: int, alpha0: float, alpha: float
                 ) -> Optional[Tuple[float, float]]:
    """Split CI: FCI on the first half, classical CI on the second."""
    from adjustment import gac_adjustment
    from fci_recording import run_fci_recorded

    n = Y.shape[0]
    n1 = n // 2
    run1 = run_fci_recorded(Y[:n1], alpha0)
    res = gac_adjustment(run1.pag, i, j, kind="pag")
    if not res.identifiable:
        return None
    L, U, _ = naive_ci(Y[n1:], i, j, res.adjustment_set, alpha)
    return L, U


def split_ci_ges(Y: np.ndarray, i: int, j: int, alpha: float,
                 ges_runner) -> Optional[Tuple[float, float]]:
    """Split CI with GES discovery on the first half."""
    from adjustment import gac_adjustment

    n = Y.shape[0]
    n1 = n // 2
    E1 = ges_runner(Y[:n1])
    res = gac_adjustment(E1, i, j, kind="cpdag")
    if not res.identifiable:
        return None
    L, U, _ = naive_ci(Y[n1:], i, j, res.adjustment_set, alpha)
    return L, U
