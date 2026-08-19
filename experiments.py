"""Seeded replication loops for coverage experiments."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from adjustment import identifiable_pairs_random_order
from baselines import naive_ci, split_ci_fci
from dgp import random_er_sem, sachs_sem, sample_data
from fci_recording import run_fci_recorded
from selective_fci import selective_ci_fci

NOISE_ID = {"gaussian": 0, "laplace": 1, "student_t5": 2}


def _select_pair(E, sem, rng, pair_rule: str):
    from adjustment import gac_adjustment
    if pair_rule == "uniform":
        return identifiable_pairs_random_order(E, "pag", rng)
    d = E.shape[0]
    if pair_rule == "true_graph":
        Et = sem.true_pag()
        pairs = [(a, b) for a in range(d) for b in range(d) if a != b]
        for k in rng.permutation(len(pairs)):
            a, b = pairs[k]
            if gac_adjustment(Et, a, b, "pag").identifiable:
                res = gac_adjustment(E, a, b, "pag")
                return ((a, b), res) if res.identifiable else (None, None)
        return None, None
    if pair_rule == "nonzero":
        pairs = [(a, b) for a in range(d) for b in range(d)
                 if a != b and abs(sem.true_effect(a, b)) > 1e-12]
        for k in rng.permutation(len(pairs)):
            a, b = pairs[k]
            res = gac_adjustment(E, a, b, "pag")
            if res.identifiable:
                return (a, b), res
        return None, None
    raise ValueError(pair_rule)


@dataclass
class FCISetting:
    d: int
    s: float
    L0: int
    n: int
    alpha: float = 0.05
    alpha0: Optional[float] = None
    noise: str = "gaussian"
    sachs: bool = False
    pair_rule: str = "uniform"
    latent_rule: str = "children2"
    name: str = ""

    def resolved_alpha0(self) -> float:
        if self.alpha0 is not None:
            return self.alpha0
        return 0.01 if self.n <= 1000 else 0.001


def _rng_for(st, rep: int, base_seed: int) -> np.random.Generator:
    ss = np.random.SeedSequence([base_seed, st.d, st.n,
                                 int(round(10 * st.s)), st.L0,
                                 NOISE_ID[st.noise], int(st.sachs), rep])
    return np.random.default_rng(ss)


def one_fci_rep(st: FCISetting, rep: int, base_seed: int) -> Optional[dict]:
    rng = _rng_for(st, rep, base_seed)
    sem = (sachs_sem(rng) if st.sachs
           else random_er_sem(st.d, st.s, st.L0, rng,
                              latent_rule=st.latent_rule))
    Y = sample_data(sem, st.n, rng, st.noise)
    alpha0 = st.resolved_alpha0()

    t_start = time.perf_counter()
    run = run_fci_recorded(Y, alpha0)
    pair, res = _select_pair(run.pag, sem, rng, st.pair_rule)
    if pair is None:
        return None
    i, j = pair
    S = res.adjustment_set
    beta = sem.true_effect(i, j)

    nL, nU, _ = naive_ci(Y, i, j, S, st.alpha)
    sel = selective_ci_fci(Y, run, i, j, S, st.alpha)
    t_sel = time.perf_counter() - t_start
    sp = split_ci_fci(Y, i, j, alpha0, st.alpha)

    sig = sem.sigma_observed()
    idx = [i] + list(S)
    g1 = float(np.linalg.solve(sig[np.ix_(idx, idx)],
                                sig[np.ix_(idx, [j])])[0, 0])

    row = dict(rep=rep, i=i, j=j, S=tuple(S), beta=beta,
               d=st.d, n=st.n, s=st.s, L0=st.L0, noise=st.noise,
               gamma1_pop=g1,
               valid_S=int(abs(g1 - beta) < 1e-8),
               naive_L=nL, naive_U=nU,
               naive_cover=int(nL <= beta <= nU),
               naive_cover_g1=int(nL <= g1 <= nU),
               sel_L=sel.ci[0], sel_U=sel.ci[1],
               sel_cover=int(sel.ci[0] <= beta <= sel.ci[1]),
               sel_cover_g1=int(sel.ci[0] <= g1 <= sel.ci[1]),
               sel_time=t_sel, n_intervals=sel.n_intervals,
               n_tests=len(run.tests),
               obs_violations=sel.diag.n_obs_violations,
               split_L=np.nan, split_U=np.nan, split_cover=np.nan,
               split_cover_g1=np.nan)
    if sp is not None:
        row.update(split_L=sp[0], split_U=sp[1],
                   split_cover=int(sp[0] <= beta <= sp[1]),
                   split_cover_g1=int(sp[0] <= g1 <= sp[1]))
    return row


def run_fci_setting(st: FCISetting, reps: int, base_seed: int,
                    progress_every: int = 0) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    excluded = 0
    for rep in range(reps):
        out = one_fci_rep(st, rep, base_seed)
        if out is None:
            excluded += 1
        else:
            rows.append(out)
        if progress_every and (rep + 1) % progress_every == 0:
            print(f"  [{st.name or st}] rep {rep + 1}/{reps} "
                  f"(excluded so far: {excluded})", flush=True)
    df = pd.DataFrame(rows)
    df.attrs["excluded"] = excluded
    df.attrs["R"] = len(rows)
    return df


def coverage_summary(df) -> dict:
    R = len(df)
    out = {"R": R}
    for m in ("naive", "split", "sel"):
        col = df[f"{m}_cover"].dropna()
        p = float(col.mean()) if len(col) else np.nan
        out[m] = 100 * p
        out[f"{m}_se"] = 100 * float(np.sqrt(p * (1 - p) / len(col))) if len(col) else np.nan
        out[f"{m}_R"] = int(len(col))
    for m in ("naive", "sel"):
        col = f"{m}_cover_g1"
        if col in df:
            v = df[col].dropna()
            p = float(v.mean()) if len(v) else np.nan
            out[f"{m}_g1"] = 100 * p
            out[f"{m}_g1_se"] = 100 * float(np.sqrt(p * (1 - p) / len(v))) if len(v) else np.nan
    if "valid_S" in df:
        out["valid_S_pct"] = 100 * float(df.valid_S.mean())
    out["sel_width_med"] = float(np.nanmedian(df.sel_U - df.sel_L))
    out["naive_width_med"] = float(np.nanmedian(df.naive_U - df.naive_L))
    out["sel_time_med"] = float(np.nanmedian(df.sel_time))
    return out


@dataclass
class GESSetting:
    d: int
    s: float
    n: int
    alpha: float = 0.05
    B: int = 10000
    G: int = 81
    grid_mult: float = 8.0
    bic_lambda: float = 2.0
    carve_frac: float = 0.7
    noise: str = "gaussian"
    name: str = ""


def one_ges_rep(st: GESSetting, rep: int, base_seed: int) -> Optional[dict]:
    from ges import run_ges_recorded
    from selective_ges import run_ges_carving, selective_ci_ges_mc
    from adjustment import gac_adjustment
    from baselines import split_ci_ges

    ss = np.random.SeedSequence([base_seed, st.d, st.n,
                                 int(round(10 * st.s)), 99,
                                 NOISE_ID[st.noise], rep])
    rng = np.random.default_rng(ss)
    sem = random_er_sem(st.d, st.s, 0, rng)
    Y = sample_data(sem, st.n, rng, st.noise)

    run = run_ges_recorded(Y, bic_lambda=st.bic_lambda)
    pair, res = identifiable_pairs_random_order(run.cpdag, "cpdag", rng)
    if pair is None:
        return None
    i, j = pair
    S = res.adjustment_set
    beta = sem.true_effect(i, j)

    nL, nU, _ = naive_ci(Y, i, j, S, st.alpha)
    t0 = time.perf_counter()
    sel = selective_ci_ges_mc(Y, run, i, j, S, st.alpha, rng,
                              B=st.B, G=st.G, grid_mult=st.grid_mult)
    t_sel = time.perf_counter() - t0

    sig = sem.sigma_observed()
    idx = [i] + list(S)
    g1 = float(np.linalg.solve(sig[np.ix_(idx, idx)],
                                sig[np.ix_(idx, [j])])[0, 0])

    row = dict(rep=rep, i=i, j=j, S=tuple(S), beta=beta,
               d=st.d, n=st.n, s=st.s, noise=st.noise,
               gamma1_pop=g1,
               valid_S=int(abs(g1 - beta) < 1e-8),
               naive_L=nL, naive_U=nU, naive_cover=int(nL <= beta <= nU),
               naive_cover_g1=int(nL <= g1 <= nU),
               sel_L=sel.ci[0], sel_U=sel.ci[1],
               sel_cover=int(sel.ci[0] <= beta <= sel.ci[1]),
               sel_cover_g1=int(sel.ci[0] <= g1 <= sel.ci[1]),
               sel_time=t_sel, min_accept=sel.min_accept,
               min_ess=sel.min_ess, obs_violations=sel.obs_violations,
               carve_L=np.nan, carve_U=np.nan, carve_cover=np.nan,
               split_L=np.nan, split_U=np.nan, split_cover=np.nan)

    run_c = run_ges_carving(Y, st.carve_frac, rng)
    res_c = gac_adjustment(run_c.cpdag, i, j, kind="cpdag")
    if res_c.identifiable:
        selc = selective_ci_ges_mc(Y, run_c, i, j, res_c.adjustment_set,
                                   st.alpha, rng, B=st.B, G=st.G,
                                   grid_mult=st.grid_mult)
        row.update(carve_L=selc.ci[0], carve_U=selc.ci[1],
                   carve_cover=int(selc.ci[0] <= beta <= selc.ci[1]))

    sp = split_ci_ges(Y, i, j, st.alpha,
                      lambda Yh: run_ges_recorded(
                          Yh, bic_lambda=st.bic_lambda).cpdag)
    if sp is not None:
        row.update(split_L=sp[0], split_U=sp[1],
                   split_cover=int(sp[0] <= beta <= sp[1]))
    return row


def run_ges_setting(st: GESSetting, reps: int, base_seed: int,
                    progress_every: int = 0):
    import pandas as pd
    rows, excluded = [], 0
    for rep in range(reps):
        out = one_ges_rep(st, rep, base_seed)
        if out is None:
            excluded += 1
        else:
            rows.append(out)
        if progress_every and (rep + 1) % progress_every == 0:
            print(f"  [{st.name or st}] rep {rep + 1}/{reps}", flush=True)
    df = pd.DataFrame(rows)
    df.attrs["excluded"] = excluded
    df.attrs["R"] = len(rows)
    return df
