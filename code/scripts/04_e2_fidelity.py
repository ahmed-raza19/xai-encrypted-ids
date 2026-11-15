"""E2 - explanation fidelity under encryption-induced visibility loss (RQ2).

Consumes the attributions saved by E1 (results/e1/*.npz) and evaluates, per flow:
  faithfulness correlation (Bhatt 2020)     with two removal baselines: train median / benign marginal
  deletion AOPC (Samek 2017) and its random-order reference
  infidelity (Yeh 2019), max-sensitivity (Yeh 2019, first 30 flows), Gini sparsity
plus pairwise agreement (Jaccard@5, Spearman) between explainers on identical flows.

Usage:  python scripts/04_e2_fidelity.py [dataset ...]
Writes results/e2_fidelity.csv (per config summary), results/e2_perflow.parquet,
       results/e2_agreement.csv, results/e2_collinearity.csv
"""
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os as _os
if _os.environ.get("XAI_BG_CORES"):   # keep background analysis off the timing core
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from scipy.stats import spearmanr                    # noqa: E402

from xaiids import data, explainers, metrics, models  # noqa: E402
from xaiids.config import RESULTS, SEED              # noqa: E402
from xaiids.tiers import tier_columns                # noqa: E402

E1 = RESULTS / "e1"
SENS_FLOWS = 30
_cache = {}


def context(ds, tier):
    key = (ds, tier)
    if key not in _cache:
        df = data.load_processed(ds)
        feats = tier_columns([c for c in df.columns if c not in data.META], tier)
        tr = df[df.split == "train"]
        te = df[df.split == "test"]
        Xtr = tr[feats].to_numpy(np.float32)
        ben = Xtr[tr.label.values == 0]
        _cache[key] = dict(feats=feats, Xte=te[feats].to_numpy(np.float32), median=np.median(Xtr, axis=0),
                           ben=ben, scale=Xtr.std(axis=0) + 1e-9, Xtr=Xtr)
    return _cache[key]


def collinearity(ds, tier, n=20000):
    c = context(ds, tier)
    rng = np.random.default_rng(SEED)
    X = c["Xtr"][rng.choice(len(c["Xtr"]), size=min(n, len(c["Xtr"])), replace=False)]
    X = X[:, X.std(axis=0) > 0]
    rho = spearmanr(X).correlation
    iu = np.triu_indices_from(rho, 1)
    a = np.abs(rho[iu])
    return {"dataset": ds, "tier": tier, "features": X.shape[1], "mean_abs_spearman": float(np.nanmean(a)),
            "share_pairs_abs_rho_gt_0.9": float(np.nanmean(a > 0.9))}


def evaluate(path, thresholds):
    meta = json.loads(path.with_suffix(".json").read_text())
    ds, tier, mname, method, budget = meta["dataset"], meta["tier"], meta["model"], meta["method"], meta["budget"]
    z = np.load(path, allow_pickle=True)
    c = context(ds, tier)
    model = models.load(ds, tier, mname)
    tau = thresholds[f"{ds}|{tier}|{mname}"]
    f = model.predict_proba
    rng = np.random.default_rng(SEED)
    ben = c["ben"]

    def benign_sampler(r):
        return ben[r.integers(len(ben))]

    X = c["Xte"][z["idx"]]
    rows = []
    sens_ex = None
    for j, (x, phi) in enumerate(zip(X, z["phi"])):
        r = {"dataset": ds, "tier": tier, "model": mname, "method": method, "budget": budget,
             "flow": int(z["idx"][j]), "y": int(z["y"][j]), "family": str(z["family"][j])}
        r["faith_median"] = metrics.faithfulness_correlation(f, x, phi, c["median"], tau, rng)
        r["faith_benign"] = metrics.faithfulness_correlation(f, x, phi, benign_sampler, tau, rng)
        a, ar = metrics.deletion_aopc(f, x, phi, c["median"], tau, rng)
        r["aopc_median"], r["aopc_median_rand"] = a, ar
        a, ar = metrics.deletion_aopc(f, x, phi, benign_sampler, tau, rng)
        r["aopc_benign"], r["aopc_benign_rand"] = a, ar
        r["infidelity"] = metrics.infidelity(f, x, phi, c["scale"], tau, rng)
        r["gini"] = metrics.gini_sparsity(phi)
        if j < SENS_FLOWS:
            if sens_ex is None:
                bg = explainers.background_kmeans(c["Xtr"], 50, seed=SEED)
                lime_sample = c["Xtr"][np.random.default_rng(SEED).choice(len(c["Xtr"]), 5000, replace=False)]
                if mname in ("RF", "XGB"):
                    model.est.set_params(n_jobs=-1)
                sens_ex = explainers.build(model, method, budget, bg, lime_sample, c["feats"], c["median"])
            r["max_sensitivity"] = metrics.max_sensitivity(sens_ex.explain, x, phi, c["scale"], rng, n=3)
        rows.append(r)
    return rows


def agreement(perflow_index):
    """Pairwise agreement between explainers that explained the same flows of one model."""
    out = []
    for (ds, tier, mname), group in perflow_index.items():
        for (la, pa), (lb, pb) in combinations(group.items(), 2):
            common = np.intersect1d(pa["idx"], pb["idx"])
            if len(common) < 20:
                continue
            ia = {v: k for k, v in enumerate(pa["idx"])}
            ib = {v: k for k, v in enumerate(pb["idx"])}
            j5 = [metrics.jaccard_at_k(pa["phi"][ia[v]], pb["phi"][ib[v]], 5) for v in common]
            sp = [metrics.rank_spearman(pa["phi"][ia[v]], pb["phi"][ib[v]]) for v in common]
            out.append({"dataset": ds, "tier": tier, "model": mname, "a": la, "b": lb, "n": len(common),
                        "jaccard5": float(np.mean(j5)), "spearman": float(np.mean(sp))})
    return pd.DataFrame(out)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    thresholds = json.loads((RESULTS / "thresholds.json").read_text())
    files = sorted(E1.glob("*.npz"))
    if want:
        files = [p for p in files if p.name.split("__")[0] in want]
    rows, index = [], {}
    for k, p in enumerate(files, 1):
        rows += evaluate(p, thresholds)
        meta = json.loads(p.with_suffix(".json").read_text())
        z = np.load(p, allow_pickle=True)
        label = meta["method"] + ("" if meta["budget"] is None else f"-{meta['budget']}")
        index.setdefault((meta["dataset"], meta["tier"], meta["model"]), {})[label] = {"idx": z["idx"], "phi": z["phi"]}
        print(f"[{k}/{len(files)}] {p.stem}", flush=True)
    per = pd.DataFrame(rows)
    per.to_parquet(RESULTS / "e2_perflow.parquet", index=False)
    agg = per.groupby(["dataset", "tier", "model", "method", "budget"], dropna=False).agg(
        n=("flow", "size"),
        faith_median=("faith_median", "mean"), faith_benign=("faith_benign", "mean"),
        aopc_median=("aopc_median", "mean"), aopc_median_rand=("aopc_median_rand", "mean"),
        aopc_benign=("aopc_benign", "mean"), aopc_benign_rand=("aopc_benign_rand", "mean"),
        infidelity=("infidelity", "median"), max_sensitivity=("max_sensitivity", "median"),
        gini=("gini", "mean")).reset_index()
    agg.to_csv(RESULTS / "e2_fidelity.csv", index=False)
    agreement(index).to_csv(RESULTS / "e2_agreement.csv", index=False)
    pairs = sorted({(k[0], k[1]) for k in index})
    pd.DataFrame([collinearity(ds, t) for ds, t in pairs]).to_csv(RESULTS / "e2_collinearity.csv", index=False)
    print(agg.to_string())
