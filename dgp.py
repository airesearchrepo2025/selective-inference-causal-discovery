"""Synthetic data generation for linear Gaussian SEMs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

SACHS_NODES = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA",
               "PKC", "P38", "Jnk"]
SACHS_EDGES = [
    ("PKC", "Raf"), ("PKC", "Mek"), ("PKC", "Jnk"), ("PKC", "P38"),
    ("PKC", "PKA"),
    ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"), ("PKA", "Akt"),
    ("PKA", "Jnk"), ("PKA", "P38"),
    ("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt"),
    ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
]


@dataclass
class SEM:
    B: np.ndarray
    observed: np.ndarray
    latent: np.ndarray
    total: np.ndarray

    @property
    def d(self) -> int:
        return len(self.observed)

    def true_effect(self, i_obs: int, j_obs: int) -> float:
        return float(self.total[self.observed[i_obs], self.observed[j_obs]])

    def true_pag(self) -> np.ndarray:
        from causallearn.graph.Dag import Dag
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.DAG2PAG import dag2pag
        D = self.B.shape[0]
        nodes = [GraphNode(f"X{k}") for k in range(D)]
        dag = Dag(nodes)
        for u in range(D):
            for v in range(D):
                if self.B[u, v] != 0:
                    dag.add_directed_edge(nodes[u], nodes[v])
        pag = dag2pag(dag, [nodes[k] for k in self.latent])
        obs_names = [f"X{k}" for k in self.observed]
        node_map = {nd.get_name(): idx for idx, nd in enumerate(pag.nodes)}
        order = [node_map[nm] for nm in obs_names]
        return np.array(pag.graph, dtype=int)[np.ix_(order, order)]

    def sigma_observed(self) -> np.ndarray:
        C = self.total
        S = C.T @ C
        return S[np.ix_(self.observed, self.observed)]

    def observed_dag_endpoint(self) -> np.ndarray:
        from graphs import dag_to_endpoint
        adj = (self.B != 0).astype(int)
        sub = adj[np.ix_(self.observed, self.observed)]
        return dag_to_endpoint(sub)


def random_er_sem(d: int, s: float, L0: int, rng: np.random.Generator,
                  latent_rule: str = "children2") -> SEM:
    """Random Erdos-Renyi DAG over d + L0 nodes."""
    D = d + L0
    p = min(1.0, s / max(D - 1, 1))
    perm = rng.permutation(D)
    B = np.zeros((D, D))
    for a in range(D):
        for b in range(a + 1, D):
            if rng.random() < p:
                w = rng.uniform(0.3, 1.0) * (1 if rng.random() < 0.5 else -1)
                B[perm[a], perm[b]] = w
    if L0 > 0 and latent_rule == "uniform":
        latent = np.sort(rng.choice(D, size=L0, replace=False))
    elif L0 > 0:
        n_children = (B != 0).sum(axis=1)
        cands = np.flatnonzero(n_children >= 2)
        if len(cands) >= L0:
            latent = np.sort(rng.choice(cands, size=L0, replace=False))
        else:
            extra = np.setdiff1d(np.arange(D), cands)
            fill = rng.choice(extra, size=L0 - len(cands), replace=False)
            latent = np.sort(np.concatenate([cands, fill]))
    else:
        latent = np.array([], dtype=int)
    observed = np.setdiff1d(np.arange(D), latent)
    total = np.linalg.inv(np.eye(D) - B)
    return SEM(B=B, observed=observed, latent=latent, total=total)


def sachs_sem(rng: np.random.Generator) -> SEM:
    """Sachs consensus topology with random weights."""
    D = len(SACHS_NODES)
    idx = {v: k for k, v in enumerate(SACHS_NODES)}
    B = np.zeros((D, D))
    for u, v in SACHS_EDGES:
        w = rng.uniform(0.3, 1.0) * (1 if rng.random() < 0.5 else -1)
        B[idx[u], idx[v]] = w
    observed = np.arange(D)
    total = np.linalg.inv(np.eye(D) - B)
    return SEM(B=B, observed=observed, latent=np.array([], dtype=int),
               total=total)


def sample_data(sem: SEM, n: int, rng: np.random.Generator,
                noise: str = "gaussian") -> np.ndarray:
    """Sample n rows of the observed coordinates."""
    D = sem.B.shape[0]
    if noise == "gaussian":
        eps = rng.standard_normal((n, D))
    elif noise == "laplace":
        eps = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(n, D))
    elif noise == "student_t5":
        eps = rng.standard_t(5, size=(n, D)) * np.sqrt(3.0 / 5.0)
    else:
        raise ValueError(noise)
    X = eps @ sem.total
    return X[:, sem.observed]
