"""Diagnostic: per-family recall and the effect of the split protocol on HIKARI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from xaiids import data, models
from xaiids.metrics import threshold_for_fpr
from xaiids.tiers import tier_columns

df = data.load_processed("hikari")
feats = tier_columns([c for c in df.columns if c not in data.META], "T1")
m = models.load("hikari", "T1", "XGB")
for s in ("val", "test"):
    d = df[df.split == s]
    p = m.predict_proba(d[feats].to_numpy(np.float32))
    print(s, "AUC", round(roc_auc_score(d.label, p), 4))
    if s == "val":
        tau = threshold_for_fpr(d.label.values, p, 0.01)
    yhat = p >= tau
    print(pd.DataFrame({"attack": d.attack, "y": d.label, "det": yhat}).groupby("attack").agg(
        n=("y", "size"), detected=("det", "mean")).round(3))

# same model class, random stratified split (the protocol most HIKARI papers use)
raw = data.load_hikari()
raw = data.clean_numeric(raw, feats)
from sklearn.model_selection import train_test_split
tr, te = train_test_split(raw, test_size=0.3, stratify=raw.label, random_state=0)
clf = XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist", n_jobs=-1)
clf.fit(tr[feats], tr.label)
p = clf.predict_proba(te[feats])[:, 1]
print("random split AUC", round(roc_auc_score(te.label, p), 4))
tau = threshold_for_fpr(te.label.values, p, 0.01)
print(pd.DataFrame({"attack": te.traffic_category, "det": p >= tau}).groupby("attack").det.mean().round(3))
dup = raw.duplicated(subset=feats).mean()
print("duplicate feature-vector share in raw HIKARI:", round(dup, 3))
print(raw.groupby("traffic_category").apply(lambda g: g.duplicated(subset=feats).mean()).round(3))
