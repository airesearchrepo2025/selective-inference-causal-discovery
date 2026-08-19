"""Selective confidence interval after FCI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from adjustment import gac_adjustment
from fci_recording import FCIRun, run_fci_recorded
from pivot import invert_ci
from truncation import Contrast, TruncationDiag, fit_contrast, truncation_set


@dataclass
class SelectiveCIResult:
    ci: Tuple[float, float]
    gamma1: float
    sigma2_hat: float
    intervals: List[Tuple[float, float]]
    diag: TruncationDiag
    n_intervals: int


def selective_ci_fci(Y: np.ndarray, run: FCIRun, i: int, j: int,
                     S: List[int], alpha: float,
                     sigma2: Optional[float] = None) -> SelectiveCIResult:
    """Selective 1-alpha CI for gamma1(S*)."""
    con = fit_contrast(Y, i, j, S)
    intervals, diag = truncation_set(Y, j, con, run.tests)
    if not intervals:
        intervals = [(con.t_obs - 1e-9, con.t_obs + 1e-9)]
    if sigma2 is None:
        sd = float(np.sqrt(con.sigma2_hat * con.norm2))
        ci = invert_ci(con.t_obs, sd, intervals, alpha, df=con.df)
    else:
        sd = float(np.sqrt(sigma2 * con.norm2))
        ci = invert_ci(con.t_obs, sd, intervals, alpha, df=None)
    return SelectiveCIResult(ci=ci, gamma1=con.gamma1,
                             sigma2_hat=con.sigma2_hat,
                             intervals=intervals, diag=diag,
                             n_intervals=len(intervals))


def full_pipeline_fci(Y: np.ndarray, alpha0: float, alpha: float, rng
                      ) -> Optional[dict]:
    """Discovery + pair selection + selective CI."""
    from adjustment import identifiable_pairs_random_order

    run = run_fci_recorded(Y, alpha0)
    pair, res = identifiable_pairs_random_order(run.pag, "pag", rng)
    if pair is None:
        return None
    i, j = pair
    sel = selective_ci_fci(Y, run, i, j, res.adjustment_set, alpha)
    return dict(i=i, j=j, S=res.adjustment_set, run=run, sel=sel)
