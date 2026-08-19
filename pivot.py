"""Truncated-normal/t pivot and selective confidence interval inversion."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t as tdist


def _log_mass(logcdf, logsf, a: float, b: float) -> float:
    if a >= b:
        return -np.inf
    if a > 0:
        la, lb = logsf(a), logsf(b)
        if not np.isfinite(la):
            return -np.inf
        if not np.isfinite(lb):
            return la
        if lb >= la:
            return -np.inf
        return la + np.log1p(-np.exp(lb - la))
    if b < 0:
        la, lb = logcdf(a), logcdf(b)
        if not np.isfinite(lb):
            return -np.inf
        if not np.isfinite(la):
            return lb
        if la >= lb:
            return -np.inf
        return lb + np.log1p(-np.exp(la - lb))
    m = -np.expm1(logcdf(a)) - np.exp(logsf(b))
    return np.log(m) if m > 0 else -np.inf


def truncated_cdf(x: float, mu: float, sd: float,
                  intervals: List[Tuple[float, float]],
                  df: int | None = None) -> float:
    """Truncated CDF at x."""
    if df is None:
        logcdf, logsf = norm.logcdf, norm.logsf
    else:
        logcdf = lambda z: tdist.logcdf(z, df)
        logsf = lambda z: tdist.logsf(z, df)
    z = lambda v: (v - mu) / sd
    num, den = [], []
    for lo, hi in intervals:
        den.append(_log_mass(logcdf, logsf, z(lo), z(hi)))
        num.append(_log_mass(logcdf, logsf, z(lo), z(min(hi, x))))
    lden = logsumexp(den)
    lnum = logsumexp(num)
    if lden == -np.inf:
        return 0.0 if x <= intervals[0][0] else 1.0
    return float(np.exp(min(lnum - lden, 0.0)))


def invert_ci(t_obs: float, sd: float,
              intervals: List[Tuple[float, float]], alpha: float,
              df: int | None = None, expand_cap: float = 1e6,
              tol: float = 1e-8) -> Tuple[float, float]:
    """Selective CI by inverting the truncated pivot."""
    G = lambda mu: truncated_cdf(t_obs, mu, sd, intervals, df=df)

    def solve(target: float, direction: int) -> float:
        step = sd
        mu_in = t_obs
        while direction * (target - G(mu_in)) > 0 and step < expand_cap * sd:
            mu_in -= direction * step
            step *= 2
        step = sd
        mu_out = mu_in
        while step < expand_cap * sd:
            mu_out = mu_in + direction * step
            if direction < 0 and G(mu_out) >= target:
                break
            if direction > 0 and G(mu_out) <= target:
                break
            step *= 2
        else:
            return -np.inf if direction < 0 else np.inf
        lo, hi = (mu_out, mu_in) if direction < 0 else (mu_in, mu_out)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if G(mid) >= target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * sd:
                break
        return 0.5 * (lo + hi)

    L = solve(1.0 - alpha / 2.0, direction=-1)
    U = solve(alpha / 2.0, direction=+1)
    return L, U
