# Selective Inference After Causal Discovery

Selective confidence intervals for causal effects after graph discovery, conditioning on the discovery algorithm's execution path.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Main files:

| File | Description |
|---|---|
| `selective_fci.py` | Selective confidence interval after FCI |
| `selective_ges.py` | Monte-Carlo selective CI after GES and data-carving variant |
| `fci_recording.py` | FCI with CI-test recording |
| `ges.py` | GES with score-comparison recording |
| `truncation.py` | Truncation-set algebra |
| `pivot.py` | Truncated CDF and CI inversion |
| `adjustment.py` | Generalized adjustment criterion |
| `graphs.py` | Graph utilities (PAG, CPDAG, MAG, DAG) |
| `dgp.py` | Data generation (random ER-DAG SEMs, Sachs topology) |
| `baselines.py` | Naive and sample-splitting baselines |
| `experiments.py` | Replication loops for coverage experiments |
