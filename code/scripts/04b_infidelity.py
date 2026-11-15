"""Optimally-scaled infidelity with subset-removal perturbations (Yeh et al., 2019).

The Gaussian-noise infidelity first computed in E2 is not meaningful for SHAP-style attributions:
they are contributions in output units, not per-unit gradients, and raw-scale noise on byte-rate
features (std ~1e7) makes the product explode. Here perturbations remove a random subset S of
features (replaced by the training median), the linear prediction is s_S = sum_{i in S} phi_i and
the true effect is d_S = g(x) - g(x_[S <- median]). Infidelity is evaluated after the optimal
rescaling beta* of the attributions (explainers output probabilities, log-odds or logits), and
normalised by E[d^2]:
    INFD_norm = min_beta E[(beta s - d)^2] / E[d^2] = 1 - E[s d]^2 / (E[s^2] E[d^2])   in [0, 1].
Writes results/e2_infidelity.csv and merges 'infid_norm' into results/e2_fidelity.csv.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os as _os
if _os.environ.get("XAI_BG_CORES"):
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from xaiids import data, models                    # noqa: E402
from xaiids.config import RESULTS, SEED            # noqa: E402
from xaiids.tiers import tier_columns              # noqa: E402

N_SUB = 50
FRAC = 0.2


def infid_norm(f, x, phi, median, tau, rng):
    d = x.shape[0]
    k = max(1, int(round(FRAC * d)))
    p_x = float(f(x[None])[0])
    sphi = phi if p_x >= tau else -phi
    X = np.repeat(x[None], N_SUB, axis=0)
    s = np.empty(N_SUB)
    for j in range(N_SUB):
        S = rng.choice(d, size=k, replace=False)
        X[j, S] = median[S]
        s[j] = sphi[S].sum()
    g = f(X)
    g0 = p_x
    if p_x < tau:
        g, g0 = 1 - g, 1 - p_x
    drop = g0 - g
    den = (s @ s) * (drop @ drop)
    return 1.0 if den <= 0 else float(1.0 - (s @ drop) ** 2 / den)


def main():
    thresholds = json.loads((RESULTS / "thresholds.json").read_text())
    ctx, rows = {}, []
    for p in sorted((RESULTS / "e1").glob("*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        ds, tier, mname = meta["dataset"], meta["tier"], meta["model"]
        if (ds, tier) not in ctx:
            df = data.load_processed(ds)
            feats = tier_columns([c for c in df.columns if c not in data.META], tier)
            tr = df[df.split == "train"]
            ctx[(ds, tier)] = (df[df.split == "test"][feats].to_numpy(np.float32),
                               np.median(tr[feats].to_numpy(np.float32), axis=0))
        Xte, median = ctx[(ds, tier)]
        m = models.load(ds, tier, mname)
        if mname in ("RF", "XGB"):
            m.est.set_params(n_jobs=1)          # small batches: avoid thread-pool overhead
        tau = thresholds[f"{ds}|{tier}|{mname}"]
        z = np.load(p, allow_pickle=True)
        rng = np.random.default_rng(SEED)
        vals = [infid_norm(m.predict_proba, Xte[i], phi, median, tau, rng) for i, phi in zip(z["idx"], z["phi"])]
        rows.append({"dataset": ds, "tier": tier, "model": mname, "method": meta["method"], "budget": meta["budget"],
                     "infid_norm": float(np.median(vals)), "infid_norm_mean": float(np.mean(vals))})
        print(p.stem, round(rows[-1]["infid_norm"], 3), flush=True)
    inf = pd.DataFrame(rows)
    inf.to_csv(RESULTS / "e2_infidelity.csv", index=False)
    fid = pd.read_csv(RESULTS / "e2_fidelity.csv")
    fid = fid.drop(columns=[c for c in ("infid_norm", "infid_norm_mean") if c in fid.columns])
    key = ["dataset", "tier", "model", "method"]
    fid["_b"] = fid.budget.fillna(-1)
    inf["_b"] = inf.budget.fillna(-1)
    fid = fid.merge(inf.drop(columns="budget"), on=key + ["_b"], how="left").drop(columns="_b")
    fid.to_csv(RESULTS / "e2_fidelity.csv", index=False)


if __name__ == "__main__":
    main()
