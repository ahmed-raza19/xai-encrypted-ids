"""Second operating point for comparison with published work: the threshold that maximises F1
on the validation split (most papers report F1 without fixing a false-positive budget).

Usage: python scripts/02b_operating_points.py hikari cic2017 [...]
Writes results/detection_valbestf1.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os as _os
if _os.environ.get("XAI_BG_CORES"):   # keep background analysis off the timing core
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from sklearn.metrics import precision_recall_curve    # noqa: E402

from xaiids import data, models                       # noqa: E402
from xaiids.config import RESULTS                     # noqa: E402
from xaiids.metrics import detection_metrics          # noqa: E402
from xaiids.tiers import tier_columns                 # noqa: E402


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return float(thr[k])


rows = []
for ds in sys.argv[1:]:
    df = data.load_processed(ds)
    feats = [c for c in df.columns if c not in data.META]
    va, te = df[df.split == "val"], df[df.split == "test"]
    for tier in ("T1", "T2", "T3"):
        cols = tier_columns(feats, tier)
        for m in ("LR", "EBM", "RF", "XGB", "MLP"):
            try:
                model = models.load(ds, tier, m)
            except FileNotFoundError:
                continue
            thr = best_f1_threshold(va.label.values, model.predict_proba(va[cols].to_numpy(np.float32)))
            met = detection_metrics(te.label.values, model.predict_proba(te[cols].to_numpy(np.float32)), thr)
            rows.append({"dataset": ds, "tier": tier, "model": m, "threshold": thr, **met})
            print(f"{ds} {tier} {m:4s} F1={met['f1']:.4f} prec={met['precision']:.4f} rec={met['recall']:.4f} "
                  f"FPR={met['fpr']:.4f} acc={met['accuracy']:.4f}", flush=True)
out = RESULTS / "detection_valbestf1.csv"
new = pd.DataFrame(rows)
if out.exists():
    old = pd.read_csv(out)
    new = pd.concat([old[~old.dataset.isin(new.dataset.unique())], new], ignore_index=True)
new.to_csv(out, index=False)
