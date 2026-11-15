"""Per-flow detection latency re-measured on an idle machine, pinned to one core (as E1).

02_train_models.py times detection right after multi-threaded training, without core pinning,
which leaves the MLP latency noisy (0.09-0.36 ms for the same network). This script re-times every
saved detector with batch size 1 on 1,000 random test flows (100 warm-ups) and writes
results/detection_latency_isolated.csv, which the tables use for every overhead ratio.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xaiids.profiling import pin_single_core   # noqa: E402
pin_single_core()

import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402

from xaiids import data, models                # noqa: E402
from xaiids.config import RESULTS, SEED        # noqa: E402
from xaiids.tiers import tier_columns          # noqa: E402


def main(datasets):
    rows = []
    for ds in datasets:
        df = data.load_processed(ds)
        te = df[df.split == "test"]
        allf = [c for c in df.columns if c not in data.META]
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(te))[:1100]
        for tier in ("T1", "T2", "T3"):
            X = te[tier_columns(allf, tier)].to_numpy(np.float32)[idx]
            for name in ("LR", "RF", "XGB", "MLP", "EBM"):
                try:
                    m = models.load(ds, tier, name)
                except FileNotFoundError:
                    continue
                if name in ("RF", "XGB"):
                    m.est.set_params(n_jobs=1)
                lat = models.time_batch1(m, X, n=1000, warmup=100) * 1e3
                rows.append({"dataset": ds, "tier": tier, "model": name, "n": len(lat),
                             "lat_p50_ms": float(np.median(lat)), "lat_p95_ms": float(np.quantile(lat, 0.95)),
                             "lat_p99_ms": float(np.quantile(lat, 0.99)), "lat_mean_ms": float(lat.mean())})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "detection_latency_isolated.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["hikari", "cic2017"])
