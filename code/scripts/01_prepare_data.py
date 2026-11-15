"""Step 2-3: load, clean, harmonise, cap and split every available dataset.

Usage:  python scripts/01_prepare_data.py hikari cic2017 [cic2018] [dohbrw]
Writes data/processed/<name>.parquet and results/data_summary_<name>.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os as _os
if _os.environ.get("XAI_BG_CORES"):
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

from xaiids import data                      # noqa: E402
from xaiids.config import RESULTS            # noqa: E402
from xaiids.tiers import feature_group, tier_columns  # noqa: E402

# CSE-CIC-IDS2018 days sharing attack families with CIC-IDS2017 (cross-network experiment)
CIC2018_DAYS = ["Wednesday-14-02", "Thursday-15-02", "Friday-16-02", "Tuesday-20-02",
                "Thursday-22-02", "Friday-23-02", "Friday-02-03"]

LOADERS = {
    "hikari": data.load_hikari,
    "cic2017": lambda: data.load_cic(2017),
    "cic2018": lambda: data.load_cic_sampled(2018, CIC2018_DAYS, cap=400_000),
    "dohbrw": data.load_dohbrw,
}


def summarise(name, df):
    feats = [c for c in df.columns if c not in data.META]
    out = {
        "rows": len(df),
        "features_T1": len(tier_columns(feats, "T1")),
        "features_T2": len(tier_columns(feats, "T2")),
        "features_T3": len(tier_columns(feats, "T3")),
        "split_rows": df.split.value_counts().to_dict(),
        "malicious_share_by_split": df.groupby("split").label.mean().round(4).to_dict(),
        "family_by_split": df.groupby(["family", "split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "T3_feature_groups": {c: feature_group(c) for c in tier_columns(feats, "T3")},
        "T1_only": sorted(set(tier_columns(feats, "T1")) - set(tier_columns(feats, "T2"))),
        "T2_only": sorted(set(tier_columns(feats, "T2")) - set(tier_columns(feats, "T3"))),
    }
    (RESULTS / f"data_summary_{name}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        t0 = time.time()
        raw = LOADERS[name]()
        df = data.finalise(raw, name)
        s = summarise(name, df)
        print(f"[{name}] {s['rows']} rows | T1/T2/T3 features = {s['features_T1']}/{s['features_T2']}/"
              f"{s['features_T3']} | splits {s['split_rows']} | malicious share {s['malicious_share_by_split']} "
              f"| {time.time() - t0:.0f}s")
        print("   families:", {k: sum(v.values()) for k, v in s["family_by_split"].items()})
