"""Truncation set computation for the FCI selection event."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

BIG = 1e18


@dataclass
class Contrast:
    eta: np.ndarray
    t_obs: float
    norm2: float
    sigma2_hat: float
    df: int
    gamma1: float


def fit_contrast(Y: np.ndarray, i: int, j: int, S: Sequence[int]) -> Contrast:
    """Compute the OLS contrast vector and regression statistics."""
    n = Y.shape[0]
    X = np.column_stack([Y[:, i], Y[:, list(S)], np.ones(n)])
    XtX = X.T @ X
    coefs = np.linalg.solve(XtX, X.T @ Y[:, j])
    eta = X @ np.linalg.solve(XtX, np.eye(X.shape[1])[:, 0])
    resid = Y[:, j] - X @ coefs
    p = X.shape[1]
    sigma2 = float(resid @ resid) / (n - p)
    t_obs = float(eta @ Y[:, j])
    return Contrast(eta=eta, t_obs=t_obs, norm2=float(eta @ eta),
                    sigma2_hat=sigma2, df=n - p, gamma1=float(coefs[0]))


def intersect_unions(A: List[Tuple[float, float]],
                     B: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for (a1, a2) in A:
        for (b1, b2) in B:
            lo, hi = max(a1, b1), min(a2, b2)
            if lo <= hi:
                out.append((lo, hi))
    out.sort()
    merged = []
    for lo, hi in out:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _poly_nonneg_set(coefs: np.ndarray, sign: int,
                     tol: float) -> List[Tuple[float, float]]:
    c = np.asarray(coefs, dtype=float)
    scale = np.max(np.abs(c)) if np.max(np.abs(c)) > 0 else 1.0
    c = c / scale
    nz = np.flatnonzero(np.abs(c) > 1e-13)
    if len(nz) == 0:
        return [(-BIG, BIG)]
    c = c[nz[0]:]
    if len(c) == 1:
        return [(-BIG, BIG)] if sign * c[0] >= -tol else []
    roots = np.roots(c)
    real = np.sort(roots[np.abs(roots.imag) < 1e-8].real)
    pts = np.concatenate([[-BIG], real, [BIG]])
    out = []
    for k in range(len(pts) - 1):
        lo, hi = pts[k], pts[k + 1]
        mid = (np.clip(lo, -1e6, 1e6) + np.clip(hi, -1e6, 1e6)) / 2.0
        if sign * np.polyval(c, mid) >= -tol:
            out.append((lo, hi))
    return intersect_unions(out, [(-BIG, BIG)])


def _quad_from_3pts(f0: float, fp: float, fm: float) -> np.ndarray:
    c0 = f0
    c1 = (fp - fm) / 2.0
    c2 = (fp + fm) / 2.0 - f0
    return np.array([c2, c1, c0])


@dataclass
class TruncationDiag:
    n_moving: int = 0
    n_constant: int = 0
    n_obs_violations: int = 0
    violations: list = field(default_factory=list)


def truncation_set(Y: np.ndarray, j: int, contrast: Contrast, tests,
                   tol: float = 1e-9) -> Tuple[List[Tuple[float, float]],
                                               TruncationDiag]:
    """Compute the truncation set as a union of intervals."""
    n, d = Y.shape
    Yt = Y - Y.mean(axis=0, keepdims=True)
    eta = contrast.eta
    norm2 = contrast.norm2
    g = (Yt.T @ eta) / norm2
    h = float(np.linalg.norm(Yt[:, j]) * np.sqrt(norm2))
    if h == 0:
        h = 1.0
    diag = TruncationDiag()
    T = [(-BIG, BIG)]
    for tst in tests:
        V = [tst.a, tst.b] + list(tst.S)
        if j not in V:
            diag.n_constant += 1
            continue
        diag.n_moving += 1
        l = V.index(j)
        m = len(V)
        K0 = Yt[:, V].T @ Yt[:, V]
        u = g[V].copy()
        dsc = 1.0 / np.sqrt(np.diag(K0))

        def K_at(ts: float) -> np.ndarray:
            tau = h * ts
            K = K0.copy()
            K[l, :] += tau * u
            K[:, l] += tau * u
            K[l, l] += tau * tau / norm2
            return K * np.outer(dsc, dsc)

        Ks = [K_at(0.0), K_at(1.0), K_at(-1.0)]
        sgn0, ld0 = np.linalg.slogdet(Ks[0])
        cscale = np.exp(-ld0 / max(m, 1)) if np.isfinite(ld0) else 1.0
        Ks = [cscale * K for K in Ks]

        def cof(K: np.ndarray, r: int, cidx: int) -> float:
            sub = np.delete(np.delete(K, r, axis=0), cidx, axis=1)
            if sub.size == 0:
                return 1.0
            return ((-1) ** (r + cidx)) * np.linalg.det(sub)

        C01 = _quad_from_3pts(*[cof(K, 0, 1) for K in Ks])
        C00 = _quad_from_3pts(*[cof(K, 0, 0) for K in Ks])
        C11 = _quad_from_3pts(*[cof(K, 1, 1) for K in Ks])
        q = np.polysub(np.polymul(C01, C01),
                       (tst.rho_c ** 2) * np.polymul(C00, C11))
        if tst.removed:
            sets = _poly_nonneg_set(q, sign=-1, tol=tol)
        else:
            s1 = _poly_nonneg_set(C01, sign=-int(tst.sign), tol=tol)
            s2 = _poly_nonneg_set(q, sign=+1, tol=tol)
            sets = intersect_unions(s1, s2)
        if not any(lo - 1e-9 <= 0.0 <= hi + 1e-9 for lo, hi in sets):
            diag.n_obs_violations += 1
            diag.violations.append((tst.a, tst.b, tst.S))
            sets = intersect_unions(sets + [(-1e-9, 1e-9)], [(-BIG, BIG)])
        T = intersect_unions(T, sets)
        if not T:
            break
    Tt = [(contrast.t_obs + h * lo, contrast.t_obs + h * hi) for lo, hi in T]
    return Tt, diag
