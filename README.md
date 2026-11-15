# Explainable AI for Encrypted-Traffic Intrusion Detection: Fidelity and Cost

Research project for **CS3001 Computer Networks** at the National University of Computer & Emerging Sciences (FAST-NUCES), by **Ahmed Raza** (24i-0026) and **Huzaifa Mudassar** (24i-0050), Department of AI & DS.

Most network traffic is now encrypted, so an intrusion detection system (IDS) only sees metadata such as packet sizes, timing and direction. Security analysts also need to know *why* a flow was flagged, which explainable-AI (XAI) methods such as SHAP and LIME provide, but these methods can be slow and their explanations are rarely checked. This project measures, on the same flows and the same machine:

1. **Cost:** how much time and memory each explainer adds to a real-time IDS;
2. **Quality:** whether the explanations are faithful, and whether encryption changes that;
3. **Consistency:** whether explanations stay the same when the network changes;
4. **A selective trigger:** a budget-aware policy that explains only the flows that matter, without slowing detection down.

## Main results

- **Encryption costs little detection accuracy.** XGBoost on the encrypted-traffic view of CIC-IDS2017 reaches F1 0.981 (precision 0.993, recall 0.969). HIKARI-2021 is much harder (F1 0.12–0.26 at 1% false positives).
- **Cost depends on the explainer–detector pair.** TreeSHAP explains an XGBoost flow in about 1 ms, close to the detection time; KernelSHAP and LIME need 7–524 ms, up to 1,500 times the detection time.
- **Paying more does not buy quality.** In 8 of 10 dataset–detector pairs the most faithful explainer is a fast, model-specific one.
- **Encryption removed artefacts, not explainability.** On CIC-IDS2017, explanations became *more* faithful when ports and TCP headers were hidden (7 of 8 explainer–model pairs).
- **Explanations drift when the network changes,** and the drift follows the loss in recall (Spearman ρ = 0.63).
- **The selective trigger works within detection capacity:** explanation overhead 0.10–0.11 of detection time, every true alert explained within 1 s, and a 35 ms verdict delay during a port-scan flood (2.7 s for "explain every alert").
- **Known-answer check:** TreeSHAP, KernelSHAP-500 and PartitionSHAP always find a planted cause; LIME often blames it when it played no role, and DeepSHAP never finds it.

Full details are in the reports in [`docs/`](docs). For a quick tour, open the demo notebook [`code/demo.ipynb`](code/demo.ipynb); its outputs are already included.

## Repository layout

```
code/      Python package `xaiids`, numbered experiment scripts and demo.ipynb (see code/README.md)
results/   all result files (CSV, JSON, attributions) used in the reports, plus the trained detector the demo loads
figures/   all figures (PDF and PNG)
report/    LaTeX source of the Phase 3 report
docs/      PDF reports: Phase 1 literature review, Phase 2 methodology, Phase 3 implementation and evaluation
```

## Reproducing the results

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Datasets are not included because of their size. Download them into `data/raw/` with `code/scripts/download_data.sh` and `code/scripts/download_2018.sh`:

- **HIKARI-2021** (Zenodo record 6463389, CC BY 4.0)
- **CIC-IDS2017** and **CSE-CIC-IDS2018**, improved releases by Engelen et al. and Liu et al.

The step-by-step commands for every table and figure are in [`code/README.md`](code/README.md). All timings were measured on one CPU core of an Intel Core i5-1245U laptop (Python 3.12.7, exact library versions in `code/requirements.txt`), with random seed 42.

## License and acknowledgement

The code is released under the [MIT License](LICENSE). Parts of the code and the reports were prepared with the help of an AI assistant (Claude, Anthropic).
