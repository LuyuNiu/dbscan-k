# DBSCAN+K: Cost-Aware Marker Detection for Intraday Trading

Code and data release for the ICAIF 2026 paper *DBSCAN+K: Cost-Aware Marker
Detection for Intraday Trading* by Luyu Niu and Henry Han (Baylor University).

DBSCAN+K detects intraday trading markers on a 2-D SR-tSNE embedding of 5-minute
bars: DBSCAN noise points are candidate markers, each candidate is scored by
geometric isolation plus high-/low-dimensional neighbor mismatch (DLP), and only
the top-*P*% per direction is kept, with *P* set by a single global
cost-to-filter rule shared by all stocks.

## Layout

```
data/      35 U.S. equities, 5-min bars, ~60 trading days (2026), one CSV per ticker
cache/     precomputed 2-D embeddings per ticker (SR-tSNE lambda=1.0, and t-SNE
           for the original detector). Walk-forward windows are built on demand
           by src/embed_walkforward.py.
results/   result CSVs reported in the paper; scripts also write here
src/       all code (see below)
```

## Environment

Python 3.10+, see `requirements.txt` (tested versions pinned). PyTorch is only
needed to recompute embeddings; everything downstream of `cache/` is CPU-only.

```
pip install -r requirements.txt
```

## Reproduce the paper

Run everything **from the repository root**.

**Main results** (from the shipped result CSVs, seconds):

```
python src/compute_baselines_bps.py    # timing-baseline table across the cost grid
python src/compute_risk_ablation.py    # risk metrics at 0.5 bp and the filter ablation
python src/compute_significance.py     # paired Wilcoxon and t-tests against each baseline
python src/select_pstar_pilot.py       # P*(c) schedule selection on the pilot grid
python src/execution_nextopen.py       # next-bar-open fill robustness -> results/nextopen_results.csv
```

Per-stock net returns for every method and cost level are shipped directly in
`results/panel_35.csv` and `results/baselines_perstock35.csv`.

The cost-to-filter schedule `P*(c) = 80/80/50/40/10` was fixed on a
sixteen-stock pilot (one-minute bars, 2019) that shares no tickers and no
time period with the thirty-five-stock panel. The pilot grid ships as
`results/pilot_grid16.csv`; `src/select_pstar_pilot.py` reproduces the
selection and shows that the paper value attains the maximum win count
against buy-and-hold at every cost level.

**Full recomputation from data + cached embeddings** (CPU, ~10-25 min each):

```
python src/grid_panel.py 0 1               # trading grid over all P and cost levels
python src/ablation_w.py                   # DLP-weight ablation grid
python src/compute_baselines_perstock.py   # timing baselines per stock
python src/compute_risk_multi.py           # risk metrics at 0, 0.5 and 2 bp
```

**Out-of-sample walk-forward** (two steps; step 1 is the slow one):

```
python src/embed_walkforward.py   # 40/20 day windows embedded independently -> cache/
python src/walkforward_eval.py    # P* chosen on the fitting window, frozen, applied to the test window
```

Step 1 re-embeds every window from scratch because t-SNE is transductive:
slicing a full-window embedding in time would let test-period bars take part in
the embedding. Expect roughly 2-3 minutes per ticker on CPU.

**Supporting analyses** (each writes to `results/` or prints to stdout):

```
python src/cluster_ablation.py    # HDBSCAN / OPTICS in place of DBSCAN
python src/bootstrap_tests.py     # block bootstrap over trading days
python src/dlp_independence.py    # rank correlation of the two score components
```

**Recompute embeddings from scratch** (optional, GPU, ~45 s per stock):

```
python src/embed_stocks.py     # t-SNE + SR-tSNE for every CSV in data/ -> cache/
```

> Reproduce from the shipped `cache/`. All reported numbers come from these
> embeddings, so recomputing them is only needed to verify the embedding step
> itself.

## Code map

| File | Role |
|---|---|
| `src/sr_tsne.py` | SR-tSNE: t-SNE with a differentiable rank-preservation (DLP) regularizer |
| `src/dbscan_k_v2.py` | DBSCAN+K detector: adaptive eps, noise scoring (isolation + DLP), top-*P*% filter |
| `src/insample_fixP.py` | trading simulator (`run_trading`), original detector, shared helpers |
| `src/compute_baselines*.py` | buy-and-hold and classical timing baselines (MA, RSI, momentum, random) |
| `src/compute_risk_ablation.py`, `src/compute_risk_multi.py` | return, Sharpe, volatility and max drawdown |
| `src/compute_significance.py` | paired Wilcoxon and t-tests against each baseline |
| `src/grid_panel.py`, `src/ablation_w.py` | result grids over *P*, cost, and DLP weight |
| `src/select_pstar_pilot.py` | P*(c) schedule selection on the pilot grid (`results/pilot_grid16.csv`) |
| `src/execution_nextopen.py` | next-bar-open fill convention robustness check |
| `src/embed_stocks.py` | t-SNE and SR-tSNE embeddings for the panel |
| `src/embed_walkforward.py`, `src/walkforward_eval.py` | out-of-sample 40/20 walk-forward |
| `src/cluster_ablation.py` | candidate-generator ablation (HDBSCAN, OPTICS) |
| `src/bootstrap_tests.py` | block bootstrap and multiple-comparison correction |
| `src/dlp_independence.py` | correlation of the geometric and DLP components |

## Data note

The per-ticker CSVs are normalized 5-minute feature files derived from publicly
available Yahoo Finance data and are provided solely for research
reproducibility. They are not investment advice and must not be redistributed
for commercial use.
