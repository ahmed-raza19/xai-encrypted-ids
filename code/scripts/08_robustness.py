"""Robustness tests relevant to encrypted traffic (Phase 3 validation).

R1 timing jitter : every timing feature of a test flow is scaled by (1 + e), e ~ U(-a, a),
                   emulating queueing / path-delay variation; a in {0.05, 0.10, 0.25, 0.50}.
R2 size padding  : every size feature is scaled by (1 + u), u ~ U(0, b), emulating TLS record /
                   QUIC padding that inflates observed sizes; b in {0.05, 0.10, 0.25, 0.50}.
For each perturbation level: detection F1/recall/FPR at the clean threshold, and explanation
stability = mean Jaccard@5 between the attribution of the clean and the perturbed flow.
R3 timing repeatability: selected E1 configurations re-timed three times (coefficient of
                   variation of the mean per-flow latency).

Usage:  python scripts/08_robustness.py perturb <ds> [...]  |  python scripts/08_robustness.py repeat
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os as _os
if _os.environ.get("XAI_BG_CORES"):   # keep background analysis off the timing core
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from xaiids import data, explainers, metrics, models   # noqa: E402
from xaiids.config import RESULTS, SEED                # noqa: E402
from xaiids.tiers import feature_group, tier_columns   # noqa: E402

LEVELS = [0.0, 0.05, 0.10, 0.25, 0.50]


def perturb(ds, tier="T3"):
    df = data.load_processed(ds)
    feats = tier_columns([c for c in df.columns if c not in data.META], tier)
    groups = np.array([feature_group(c) for c in feats])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    Xtr = tr[feats].to_numpy(np.float32)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(te), size=min(20000, len(te)), replace=False)
    X, y = te[feats].to_numpy(np.float32)[sel], te.label.values[sel]
    thresholds = json.loads((RESULTS / "thresholds.json").read_text())
    bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
    median = np.median(Xtr, axis=0)
    rows = []
    for mname, method, budget in (("XGB", "TreeSHAP-interv", None), ("MLP", "IG", 32)):
        m = models.load(ds, tier, mname)
        tau = thresholds[f"{ds}|{tier}|{mname}"]
        ex = explainers.build(m, method, budget, bg, None, feats, median)
        ex_idx = rng.choice(len(X), size=200, replace=False)
        phi_clean = [ex.explain(X[i]) for i in ex_idx]
        for kind, grp, lo in (("timing_jitter", "timing", -1), ("size_padding", "size", 0)):
            mask = groups == grp
            for a in LEVELS:
                r = np.random.default_rng(SEED + int(a * 1000))
                factor = np.ones_like(X)
                if a > 0:
                    factor[:, mask] = 1.0 + r.uniform(lo * a, a, size=(len(X), mask.sum()))
                Xp = X * factor
                met = metrics.detection_metrics(y, m.predict_proba(Xp), tau)
                j5 = [metrics.jaccard_at_k(phi_clean[k], ex.explain(Xp[i]), 5) for k, i in enumerate(ex_idx)]
                rows.append({"dataset": ds, "tier": tier, "model": mname, "explainer": method, "perturbation": kind,
                             "level": a, "features_perturbed": int(mask.sum()), "f1": met["f1"],
                             "recall": met["recall"], "fpr": met["fpr"], "expl_jaccard5": float(np.mean(j5))})
                print(f"  {ds} {mname} {kind:14s} a={a:<4} F1={met['f1']:.4f} FPR={met['fpr']:.4f} "
                      f"J5={np.mean(j5):.3f}", flush=True)
    out = RESULTS / "robustness_perturbation.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[old.dataset != ds], new], ignore_index=True)
    new.to_csv(out, index=False)


REPEAT = [("cic2017", "T3", "XGB", "TreeSHAP-path", "None"), ("cic2017", "T3", "XGB", "TreeSHAP-interv", "None"),
          ("cic2017", "T3", "XGB", "KernelSHAP", "500"), ("cic2017", "T3", "XGB", "LIME", "1000"),
          ("cic2017", "T3", "MLP", "IG", "32"), ("cic2017", "T3", "MLP", "DeepSHAP", "None"),
          ("hikari", "T3", "RF", "KernelSHAP", "500"), ("hikari", "T3", "XGB", "TreeSHAP-interv", "None")]


def repeat(n=3):
    """Re-time selected configurations in fresh processes (results/e1 untouched)."""
    script = ROOT / "scripts" / "03_e1_cost.py"
    rows = []
    for c in REPEAT:
        for r in range(n):
            out = subprocess.run([sys.executable, str(script), "worker", *c], capture_output=True, text=True,
                                 env={**__import__("os").environ, "XAI_E1_DIR": str(RESULTS / "e1_repeat")})
            res = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append({"config": "|".join(c), "rep": r, "mean_ms": res["mean_ms"], "p95_ms": res["p95_ms"],
                         "peak_mem_increment_mb": res["peak_mem_increment_mb"]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "robustness_timing_repeat.csv", index=False)
    s = df.groupby("config").mean_ms.agg(["mean", "std"])
    s["cv"] = s["std"] / s["mean"]
    print(s)


if __name__ == "__main__":
    if sys.argv[1] == "perturb":
        for ds in sys.argv[2:]:
            perturb(ds)
    elif sys.argv[1] == "repeat":
        repeat()
