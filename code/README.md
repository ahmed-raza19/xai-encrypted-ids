# Fidelity and Computational Cost of XAI for Encrypted-Traffic Intrusion Detection — Phase 3 code

Implementation of the Phase 2 methodology: three feature-visibility tiers (T1 plaintext-era,
T2 TLS 1.3-over-TCP, T3 QUIC-strict metadata), five detectors, nine post-hoc / native explainers,
fidelity and cost measurement, and the **budget-aware selective XAI trigger** evaluated in a
trace-driven streaming replay.

## Layout

```
xai-encrypted-ids/
  code/
    xaiids/            library (one module per Phase 2 methodology step)
      config.py        paths, seeds, row caps, explainer budgets, trigger defaults
      data.py          loaders, cleaning, label harmonisation, water-filling cap, time-ordered split, leakage control
      tiers.py         T1/T2/T3 visibility tiers and semantic feature groups (regex over column names)
      models.py        LR, RF, XGBoost, MLP (PyTorch), EBM — all on raw features (scaling inside the model)
      explainers.py    TreeSHAP (path / interventional), KernelSHAP, LIME, IG, DeepSHAP, LinearSHAP,
                       EBM-native, PartitionSHAP (Owen), amortised surrogate
      metrics.py       detection, faithfulness, deletion AOPC, infidelity, max-sensitivity, Gini, Jaccard@k
      profiling.py     single-core pinning, peak-RSS sampler
      trigger.py       selective trigger (Algorithm 1), policies B1-B6, ablations A1-A6
      stream.py        discrete-event replay (Poisson arrivals, main + deferred + deep lanes)
      stats.py         bootstrap CIs, Holm, Wilcoxon + rank-biserial, Friedman-Nemenyi, TOST, McNemar
    scripts/           numbered experiment entry points (below)
  data/raw/            downloaded datasets (not versioned)
  data/processed/      cleaned parquet files (not versioned)
  results/             all CSV/JSON outputs and raw attributions (results/e1/*.npz); trained models are not versioned
  figures/             all figures (PDF + PNG)
  report/              Phase 3 LaTeX report
  docs/                PDF reports of Phases 1-3
```

## Environment

* Windows 11, Intel Core i5-1245U (10 cores / 12 threads), 15.6 GB RAM, no GPU; Python 3.12.7.
* `python -m venv .venv && .venv\Scripts\python -m pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
* Packet-level validation runs in WSL Ubuntu 24.04 with NFStream 6.6.0: `python3 -m venv ~/nfsenv && ~/nfsenv/bin/pip install nfstream pandas pyarrow`
* All timing runs are pinned to one logical CPU (`XAI_CORE`, default 2) with single-threaded BLAS/OpenMP/PyTorch.

## Data

| Dataset | Source | Notes |
|---|---|---|
| HIKARI-2021 | Zenodo record 6463389 (CC-BY 4.0) | flow CSV + two attack PCAPs + anonymised benign PCAPs |
| CIC-IDS2017 (improved) | Engelen et al., distrinet-research.be/CNS2022 | corrected labels; "Attempted" relabelled benign |
| CSE-CIC-IDS2018 (improved) | Liu, Engelen et al., same site | read selectively from the 10.4 GB zip |
| CIRA-CIC-DoHBrw-2020 | unb.ca/cic/datasets/dohbrw-2020.html | requires the CIC download form; place CSVs in `data/raw/dohbrw/` |

`bash code/scripts/download_data.sh` and `bash code/scripts/download_2018.sh` fetch the public files.

## Reproducing every result (run from `code/`)

| Step | Command | Output |
|---|---|---|
| Data preparation | `python scripts/01_prepare_data.py hikari cic2017 cic2018 [dohbrw]` | `data/processed/*.parquet`, `results/data_summary_*.json` |
| Detector training (all tiers) | `python scripts/02_train_models.py hikari cic2017 [dohbrw]` | `results/detection.csv`, `results/models/` |
| Second operating point (val-best F1) | `python scripts/02b_operating_points.py` | `results/detection_valbestf1.csv` |
| Detection latency on an idle, pinned core | `python scripts/02c_detection_latency.py` | `results/detection_latency_isolated.csv` (used for every overhead ratio) |
| E1 cost profiling (RQ1) | `python scripts/03_e1_cost.py run cic2017 hikari --tiers T3,T1,T2` | `results/e1/*.json|npz` (attributions used by E2) |
| E1 re-timing on an idle machine | `XAI_E1_DIR=../results/e1_isolated python scripts/03_e1_cost.py run cic2017 hikari --tiers T3,T1,T2` | `results/e1_isolated/` (timings used in the report) |
| E2 fidelity (RQ2) | `python scripts/04_e2_fidelity.py` | `results/e2_*.csv` |
| E2 normalised subset-removal infidelity | `python scripts/04b_infidelity.py` | `results/e2_infidelity.csv`, merged into `e2_fidelity.csv` |
| E3 drift (RQ3) | `python scripts/05_e3_drift.py crossnet temporal hikari cic2017 p9` | `results/e3_*.csv` |
| E4 streaming trigger (RQ4) | `python scripts/06_e4_stream.py cic2017 hikari` | `results/e4_*.csv` |
| E4 rule-parameter sensitivity (A6) | `python scripts/06b_e4_sensitivity.py cic2017 hikari` | `results/e4sens_*.csv` |
| HIKARI diagnostics | `python scripts/diag_hikari.py`, `python scripts/diag_hikari_index.py` | console, `results/diag_hikari_index.json` |
| Packet-level payload stripping | (WSL) `~/nfsenv/bin/python code/scripts/07_pcap_pipeline.py` | `results/pcap_validation.json` |
| Robustness | `python scripts/08_robustness.py perturb cic2017 hikari` and `... repeat` | `results/robustness_*.csv` |
| Extra ablations A8/A10/A11 | `python scripts/10_ablation_extra.py cic2017 hikari` | `results/ablation_extra.csv` |
| Known-answer sanity test (planted feature) | `python scripts/12_sanity_planted.py` | `results/sanity_planted.csv` |
| Statistics + figures + tables | `python scripts/09_figures.py` and `python scripts/11_tables.py` | `figures/`, `report/generated/` |
| Report PDF | in `report/`: `tectonic -X compile Phase3_Report.tex` (or pdfLaTeX, two passes) | `report/Phase3_Report.pdf` |

Every script is deterministic for a given seed (`config.SEED = 42`); E1 is resumable (finished
configurations are skipped). Timings depend on the machine; relative orderings are what the
report interprets.

**Timing hygiene.** On laptops the cores share a power budget, so load on *other* cores slows the
timing core. Run timing-sensitive steps (E1, E4, A6 sensitivity, A10/A11, timing repeat) alone:
`bash scripts/run_isolated.sh` runs them one after another, and `scripts/run_after_batch.sh`
queues the sensitivity and infidelity steps behind it. Non-timing work can be confined to other
cores with `XAI_BG_CORES=4-11`.
