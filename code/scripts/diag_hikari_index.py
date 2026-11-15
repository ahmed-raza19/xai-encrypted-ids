"""Diagnostic: does the row index of the public HIKARI-2021 CSV leak the label?

The CSV carries two index columns ('' and 'Unnamed: 0'). If rows are grouped by traffic category,
a pipeline that keeps them can separate the classes by row number alone. Our pipeline drops both
(tiers.ID_PATTERNS: ^unnamed). Writes results/diag_hikari_index.json.
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xaiids.config import DATA_RAW, RESULTS  # noqa: E402

z = zipfile.ZipFile(DATA_RAW / "hikari" / "ALLFLOWMETER_HIKARI2021.csv.zip")
df = pd.read_csv(z.open("ALLFLOWMETER_HIKARI2021.csv"), usecols=[0, 1, 86, 87], low_memory=False)
df.columns = ["idx0", "idx1", "traffic_category", "Label"]
cat = df.traffic_category.astype(str)
runs = int((cat != cat.shift()).sum())           # number of contiguous blocks of one category
out = {"rows": int(len(df)), "attack_rows": int(df.Label.sum()),
       "category_counts": cat.value_counts().to_dict(),
       "contiguous_category_blocks": runs,
       "auc_idx0": float(roc_auc_score(df.Label, df.idx0)), "auc_idx1": float(roc_auc_score(df.Label, df.idx1)),
       "idx0_equals_row": bool((df.idx0.values == np.arange(len(df))).all())}
auc = max(out["auc_idx0"], 1 - out["auc_idx0"])
out["auc_idx0_best_direction"] = float(auc)
# per-category index ranges
out["index_range_by_category"] = {k: [int(g.idx1.min()), int(g.idx1.max())] for k, g in df.groupby(cat)}
print(json.dumps(out, indent=1))
(RESULTS / "diag_hikari_index.json").write_text(json.dumps(out, indent=1))
