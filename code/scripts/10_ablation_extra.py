"""Component ablations outside the streaming trigger (Phase 2 Table 7: A8, A10, A11).

A8  feature groups within T3: XGBoost retrained on timing-only, size-only and
    volume/direction-only subsets -> detection and TreeSHAP faithfulness.
A10 grouped (Owen/PartitionSHAP) vs independent (KernelSHAP) attributions at a matched number
    of model evaluations -> fidelity and stability (max-sensitivity) under collinear metadata.
A11 KernelSHAP background size k in {10, 50, 100} -> cost and fidelity.

Usage: python scripts/10_ablation_extra.py <ds> [...]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xaiids.profiling import pin_single_core  # noqa: E402
pin_single_core()

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from xaiids import data, explainers, metrics, models   # noqa: E402
from xaiids.config import RESULTS, SEED                # noqa: E402
from xaiids.metrics import threshold_for_fpr           # noqa: E402
from xaiids.tiers import feature_group, tier_columns   # noqa: E402

N = 100


def fidelity_block(model, ex, X, median, tau, scale, rng, sens=20):
    faith, aopc, sensv, times = [], [], [], []
    for j, x in enumerate(X):
        t0 = time.perf_counter()
        phi = ex.explain(x)
        times.append(time.perf_counter() - t0)
        faith.append(metrics.faithfulness_correlation(model.predict_proba, x, phi, median, tau, rng))
        a, ar = metrics.deletion_aopc(model.predict_proba, x, phi, median, tau, rng)
        aopc.append(a - ar)
        if j < sens:
            sensv.append(metrics.max_sensitivity(ex.explain, x, phi, scale, rng, n=3))
    return {"faith": float(np.mean(faith)), "aopc_gain": float(np.mean(aopc)),
            "max_sensitivity": float(np.median(sensv)), "mean_ms": float(np.mean(times) * 1e3)}


def run(ds):
    df = data.load_processed(ds)
    feats = tier_columns([c for c in df.columns if c not in data.META], "T3")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    rng = np.random.default_rng(SEED)
    sel = np.concatenate([rng.permutation(np.flatnonzero(te.label.values == c))[:N // 2] for c in (1, 0)])
    rows = []

    # ---- A8: feature groups
    groups = {g: [c for c in feats if feature_group(c) == g] for g in ("timing", "size", "volume_direction")}
    groups["all_T3"] = feats
    for g, cols in groups.items():
        if len(cols) < 2:
            continue
        Xtr = tr[cols].to_numpy(np.float32)
        m = models.make("XGB")
        m.fit(Xtr, tr.label.values, w=models.class_weights(tr.label.values))
        m.est.set_params(n_jobs=1)
        tau = threshold_for_fpr(va.label.values, m.predict_proba(va[cols].to_numpy(np.float32)), 0.01)
        det = metrics.detection_metrics(te.label.values, m.predict_proba(te[cols].to_numpy(np.float32)), tau)
        bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
        ex = explainers.TreeSHAP(m, bg, "interventional")
        fb = fidelity_block(m, ex, te[cols].to_numpy(np.float32)[sel], np.median(Xtr, 0), tau, Xtr.std(0) + 1e-9, rng)
        rows.append({"dataset": ds, "ablation": "A8 feature group", "variant": g, "features": len(cols),
                     "f1": det["f1"], "pr_auc": det.get("pr_auc"), "fpr": det["fpr"], **fb})
        print(rows[-1], flush=True)

    # ---- A10 / A11 on the T3 XGBoost detector
    m = models.load(ds, "T3", "XGB")
    m.est.set_params(n_jobs=1)
    Xtr = tr[feats].to_numpy(np.float32)
    X = te[feats].to_numpy(np.float32)[sel]
    tau = json.loads((RESULTS / "thresholds.json").read_text())[f"{ds}|T3|XGB"]
    median, scale = np.median(Xtr, 0), Xtr.std(0) + 1e-9
    for k in (10, 50, 100):
        bg = explainers.background_kmeans(Xtr, k, seed=SEED)
        fb = fidelity_block(m, explainers.KernelSHAP(m, bg, 500), X, median, tau, scale, rng)
        rows.append({"dataset": ds, "ablation": "A11 background size", "variant": f"KernelSHAP-500, k={k}", **fb})
        print(rows[-1], flush=True)
    bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
    for name, ex in (("independent KernelSHAP-500", explainers.KernelSHAP(m, bg, 500)),
                     ("grouped PartitionSHAP-500 (Owen)", explainers.PartitionOwen(m, bg, 500))):
        fb = fidelity_block(m, ex, X, median, tau, scale, rng)
        rows.append({"dataset": ds, "ablation": "A10 grouped vs independent", "variant": name, **fb})
        print(rows[-1], flush=True)

    out = RESULTS / "ablation_extra.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[old.dataset != ds], new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
