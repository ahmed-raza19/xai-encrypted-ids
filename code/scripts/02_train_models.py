"""Step 5-6: train every detector on every visibility tier and evaluate detection.

Usage:  python scripts/02_train_models.py hikari cic2017 [...]
Outputs: results/models/<ds>_<tier>_<model>.joblib, results/detection.csv (appended),
         results/thresholds.json (alert thresholds per model, chosen on validation).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402

from xaiids import data, models                       # noqa: E402
from xaiids.config import RESULTS, SEED               # noqa: E402
from xaiids.metrics import detection_metrics, threshold_for_fpr  # noqa: E402
from xaiids.tiers import tier_columns                 # noqa: E402

TIERS = ["T1", "T2", "T3"]
MODELS = ["LR", "RF", "XGB", "MLP"]
EBM_MAX_ROWS = 100_000


def single_thread(model):
    """Restrict a fitted model to one thread so latency is a per-core cost."""
    if model.name in ("RF", "XGB"):
        model.est.set_params(n_jobs=1)
    torch.set_num_threads(1)


def multi_thread(model):
    if model.name in ("RF", "XGB"):
        model.est.set_params(n_jobs=-1)
    torch.set_num_threads(8)


def run(ds):
    df = data.load_processed(ds)
    feats = [c for c in df.columns if c not in data.META]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    thresholds = json.loads((RESULTS / "thresholds.json").read_text()) if (RESULTS / "thresholds.json").exists() else {}
    rows = []
    for tier in TIERS:
        cols = tier_columns(feats, tier)
        Xtr, Xva, Xte = (d[cols].to_numpy(np.float32) for d in (tr, va, te))
        ytr, yva, yte = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()
        names = MODELS + (["EBM"] if tier == "T3" else [])
        for name in names:
            t0 = time.time()
            m = models.make(name)
            if name == "XGB":
                m.fit(Xtr, ytr, w=models.class_weights(ytr))
            elif name == "EBM":
                rng = np.random.default_rng(SEED)
                idx = rng.choice(len(Xtr), size=min(len(Xtr), EBM_MAX_ROWS), replace=False)
                m.fit(Xtr[idx], ytr[idx], feature_names=cols)
            else:
                m.fit(Xtr, ytr)
            fit_s = time.time() - t0
            p_va, p_te = m.predict_proba(Xva), m.predict_proba(Xte)
            tau = threshold_for_fpr(yva, p_va, 0.01)
            met = detection_metrics(yte, p_te, tau)
            single_thread(m)
            lat = models.time_batch1(m, Xte, n=500, warmup=50)
            multi_thread(m)
            models.save(m, ds, tier)
            thresholds[f"{ds}|{tier}|{name}"] = tau
            rows.append({"dataset": ds, "tier": tier, "model": name, "features": len(cols), "tau": tau,
                         "fit_s": round(fit_s, 1), "lat_p50_ms": float(np.median(lat) * 1e3),
                         "lat_p95_ms": float(np.quantile(lat, 0.95) * 1e3), **met})
            print(f"  {ds} {tier} {name:4s} d={len(cols):3d} F1={met['f1']:.4f} AUC={met.get('roc_auc', np.nan):.4f} "
                  f"FPR={met['fpr']:.4f} MCC={met['mcc']:.4f} fit={fit_s:.0f}s lat={np.median(lat) * 1e3:.3f}ms",
                  flush=True)
    (RESULTS / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    out = RESULTS / "detection.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        old = old[old.dataset != ds]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
