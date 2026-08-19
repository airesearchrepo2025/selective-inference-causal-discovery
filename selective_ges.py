"""Monte-Carlo selective CI after GES, and data-carving variant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ges import GESRun
from truncation import Contrast, fit_contrast


@dataclass
class GESSelectiveResult:
    ci: Tuple[float, float]
    gamma1: float
    min_accept: float
    min_ess: float
    grid: np.ndarray
    Fhat: np.ndarray
    n_moving_dets: int
    obs_violations: int


def _det_quads(Y: np.ndarray, j: int, con: Contrast, run: GESRun,
               h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quadratic coefficients of each registry logdet's determinant."""
    rows = run.rows
    if rows is not None:
        Ysel = Y[rows]
        mv = (con.eta / con.norm2)[rows]
    else:
        Ysel = Y
        mv = con.eta / con.norm2
    Yt = Ysel - Ysel.mean(axis=0, keepdims=True)
    mvt = mv - mv.mean()
    scale = 1.0 / np.linalg.norm(Yt, axis=0)
    Yt = Yt * scale
    mvt = mvt * scale[j]
    n_dets = len(run.registry.sets)
    coeffs = np.zeros((n_dets, 3))
    moving = np.zeros(n_dets, dtype=bool)
    for k, S in enumerate(run.registry.sets):
        V = sorted(S)
        sub = Yt[:, V]
        if j not in S:
            K = sub.T @ sub
            coeffs[k, 2] = np.linalg.det(K) if len(V) else 1.0
            continue
        moving[k] = True
        l = V.index(j)
        vals = []
        for ts in (0.0, 1.0, -1.0):
            col = sub.copy()
            col[:, l] = col[:, l] + (h * ts) * mvt
            K = col.T @ col
            vals.append(np.linalg.det(K))
        f0, fp, fm = vals
        coeffs[k] = [(fp + fm) / 2.0 - f0, (fp - fm) / 2.0, f0]
    return coeffs, moving


def _constraint_matrix(run: GESRun):
    rows, cols, data, consts, geqs = [], [], [], [], []
    for r, c in enumerate(run.constraints):
        for idx, cf in c.terms:
            rows.append(r)
            cols.append(idx)
            data.append(cf)
        consts.append(c.const)
        geqs.append(c.geq)
    n_dets = len(run.registry.sets)
    A = sp.csr_matrix((data, (rows, cols)),
                      shape=(len(run.constraints), max(n_dets, 1)))
    return A, np.array(consts), np.array(geqs, dtype=bool)


def _accept_mask(tau_scaled: np.ndarray, coeffs, moving, A, consts, geqs,
                 tol: float) -> np.ndarray:
    NT = len(tau_scaled)
    dets = np.empty((coeffs.shape[0], NT))
    dets[~moving] = coeffs[~moving, 2:3]
    if moving.any():
        c = coeffs[moving]
        dets[moving] = (c[:, [0]] * tau_scaled ** 2 + c[:, [1]] * tau_scaled
                        + c[:, [2]])
    ld = np.log(np.clip(dets, 1e-300, None))
    vals = A @ ld + consts[:, None]
    sat = np.where(geqs[:, None], vals >= -tol, vals <= tol)
    return sat.all(axis=0)


def _reduced_system(run: GESRun, coeffs: np.ndarray, moving: np.ndarray):
    ld_const = np.log(np.clip(coeffs[:, 2], 1e-300, None))
    mov_idx = np.flatnonzero(moving)
    col_of = {int(k): c for c, k in enumerate(mov_idx)}
    rows, cols, data, consts, geqs = [], [], [], [], []
    r = 0
    for c in run.constraints:
        const = c.const
        terms = []
        for idx, cf in c.terms:
            if moving[idx]:
                terms.append((col_of[idx], cf))
            else:
                const += cf * ld_const[idx]
        if not terms:
            continue
        for cc, cf in terms:
            rows.append(r)
            cols.append(cc)
            data.append(cf)
        consts.append(const)
        geqs.append(c.geq)
        r += 1
    A = sp.csr_matrix((data, (rows, cols)), shape=(r, max(len(mov_idx), 1)))
    return A, np.array(consts), np.array(geqs, dtype=bool), mov_idx


def _accept_mask_reduced(tau_scaled: np.ndarray, coeffs_mov: np.ndarray,
                         A, consts, geqs, tol: float) -> np.ndarray:
    dets = (coeffs_mov[:, [0]] * tau_scaled ** 2
            + coeffs_mov[:, [1]] * tau_scaled + coeffs_mov[:, [2]])
    ld = np.log(np.clip(dets, 1e-300, None))
    vals = A @ ld + consts[:, None]
    sat = np.where(geqs[:, None], vals >= -tol, vals <= tol)
    return sat.all(axis=0)


def selective_ci_ges_mc(Y: np.ndarray, run: GESRun, i: int, j: int,
                        S: List[int], alpha: float, rng,
                        B: int = 10000, G: int = 81,
                        grid_mult: float = 8.0, max_expand: int = 2,
                        estimator: str = "is") -> GESSelectiveResult:
    """Monte Carlo selective CI after GES."""
    con = fit_contrast(Y, i, j, S)
    sd = float(np.sqrt(con.sigma2_hat * con.norm2))
    h = sd if sd > 0 else 1.0
    coeffs, moving = _det_quads(Y, j, con, run, h)
    tol = 1e-6 * max(run.n, 1)

    A_full, c_full, g_full = _constraint_matrix(run)
    obs_ok = _accept_mask(np.zeros(1), coeffs, moving, A_full, c_full,
                          g_full, tol)
    obs_violations = int(~obs_ok[0])

    A, consts, geqs, mov_idx = _reduced_system(run, coeffs, moving)
    coeffs_mov = coeffs[mov_idx]

    xi = rng.standard_normal(B)

    if estimator == "is":
        t = con.t_obs + sd * xi
        tau_sc = (t - con.t_obs) / h
        ok = _accept_mask_reduced(tau_sc, coeffs_mov, A, consts, geqs, tol)
        t_acc = t[ok]
        below = t_acc <= con.t_obs
        base_acc = float(ok.mean())

        def scan(grid):
            F = np.full(len(grid), np.nan)
            ess = np.zeros(len(grid))
            if len(t_acc) == 0:
                return F, ess
            for g, mu in enumerate(grid):
                lw = (-(t_acc - mu) ** 2 + (t_acc - con.t_obs) ** 2) / (2 * sd * sd)
                lw -= lw.max()
                w = np.exp(lw)
                sw = w.sum()
                ess[g] = sw * sw / (w @ w)
                F[g] = float(w[below].sum() / sw)
            return F, ess
    else:
        def scan(grid):
            F = np.full(len(grid), np.nan)
            acc = np.zeros(len(grid))
            for g, mu in enumerate(grid):
                t = mu + sd * xi
                tau_sc = (t - con.t_obs) / h
                ok = _accept_mask_reduced(tau_sc, coeffs_mov, A, consts,
                                          geqs, tol)
                na = int(ok.sum())
                acc[g] = na / B
                if na:
                    F[g] = float(np.mean(t[ok] <= con.t_obs))
            return F, acc
        base_acc = None

    lo_m, hi_m = -grid_mult, grid_mult
    for _ in range(max_expand + 1):
        grid = con.gamma1 + np.linspace(lo_m, hi_m, G) * sd
        F, aux = scan(grid)
        sel = np.flatnonzero((F >= alpha / 2) & (F <= 1 - alpha / 2))
        if len(sel) == 0:
            ci = (con.gamma1, con.gamma1)
            break
        touching = sel[0] == 0 or sel[-1] == len(grid) - 1
        k = sel[0]
        if k > 0 and np.isfinite(F[k - 1]) and F[k - 1] > F[k]:
            frac = (F[k - 1] - (1 - alpha / 2)) / (F[k - 1] - F[k])
            L = float(grid[k - 1] + np.clip(frac, 0, 1) * (grid[k] - grid[k - 1]))
        else:
            L = float(grid[k])
        k = sel[-1]
        if k < len(grid) - 1 and np.isfinite(F[k + 1]) and F[k] > F[k + 1]:
            frac = (F[k] - alpha / 2) / (F[k] - F[k + 1])
            U = float(grid[k] + np.clip(frac, 0, 1) * (grid[k + 1] - grid[k]))
        else:
            U = float(grid[k])
        ci = (L, U)
        if not touching:
            break
        lo_m *= 2.0
        hi_m *= 2.0
    if estimator == "is":
        min_acc = base_acc
        min_ess = float(np.min(aux)) if len(aux) and np.any(np.isfinite(aux)) else 0.0
    else:
        min_acc = float(np.min(aux[np.isfinite(aux)])) if len(aux) else 0.0
        min_ess = 0.0
    return GESSelectiveResult(ci=ci, gamma1=con.gamma1, min_accept=min_acc,
                              min_ess=min_ess, grid=grid, Fhat=F,
                              n_moving_dets=int(moving.sum()),
                              obs_violations=obs_violations)


def run_ges_carving(Y: np.ndarray, frac: float, rng) -> GESRun:
    """GES selection on a fixed row subset for data carving."""
    from ges import run_ges_recorded
    n = Y.shape[0]
    n_sel = int(round(frac * n))
    rows = rng.permutation(n)[:n_sel]
    rows.sort()
    return run_ges_recorded(Y, rows=rows)
