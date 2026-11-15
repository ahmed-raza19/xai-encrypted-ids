"""Known-answer sanity test for the explainers (research validation).

A random feature `planted` ~ N(0, 1), independent of the traffic, is added to the CIC-IDS2017 T3
data, and every benign flow with planted > 1.5 is relabelled as an attack. The retrained detector
must therefore use `planted` for exactly those flows, so the correct explanation is known:

  * group A (benign traffic, attack only because planted > 1.5): `planted` must be the top feature;
  * group B (real attacks, planted <= 1.5): `planted` is irrelevant and must NOT be the top feature.

For each explainer we report the share of group-A flows where `planted` ranks first (hit rate),
the share of attribution mass it receives in group A, and the share of group-B flows where it
wrongly ranks first (false-hit rate). Not a timing experiment, so no core pinning.

Usage: python scripts/12_sanity_planted.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from xaiids import data, explainers, metrics, models   # noqa: E402
from xaiids.config import RESULTS, SEED                # noqa: E402
from xaiids.metrics import threshold_for_fpr           # noqa: E402
from xaiids.tiers import tier_columns                  # noqa: E402

DS = "cic2017"
CUT = 1.5          # benign flows above this planted value become attacks
N = 100            # flows per group


def explainers_for(name, m, Xtr, bg, cols):
    median = np.median(Xtr, 0)
    if name == "XGB":
        return [explainers.TreeSHAP(m, bg, "path"), explainers.TreeSHAP(m, bg, "interventional"),
                explainers.KernelSHAP(m, bg, 100), explainers.KernelSHAP(m, bg, 500),
                explainers.LIME(m, Xtr[:5000], cols, 1000), explainers.PartitionOwen(m, bg, 500)]
    return [explainers.IntegratedGradients(m, median, 16), explainers.DeepSHAP(m, bg),
            explainers.KernelSHAP(m, bg, 100), explainers.KernelSHAP(m, bg, 500),
            explainers.LIME(m, Xtr[:5000], cols, 1000)]


def main():
    df = data.load_processed(DS)
    feats = tier_columns([c for c in df.columns if c not in data.META], "T3")
    rng = np.random.default_rng(SEED)
    df = df.assign(planted=rng.standard_normal(len(df)).astype(np.float32))
    df["y"] = df.label.values.astype(int)
    df.loc[(df.label.values == 0) & (df.planted.values > CUT), "y"] = 1
    cols = feats + ["planted"]
    k = len(cols) - 1                                  # index of the planted feature
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr = tr[cols].to_numpy(np.float32)

    grpA = np.flatnonzero((te.label.values == 0) & (te.planted.values > CUT))
    grpB = np.flatnonzero((te.label.values == 1) & (te.planted.values <= CUT))
    pick = np.random.default_rng(SEED)
    grpA, grpB = pick.permutation(grpA)[:N], pick.permutation(grpB)[:N]
    Xte = te[cols].to_numpy(np.float32)
    bg = explainers.background_kmeans(Xtr, 50, seed=SEED)

    rows = []
    for name in ("XGB", "MLP"):
        t0 = time.time()
        m = models.make(name)
        m.fit(Xtr, tr.y.values, w=models.class_weights(tr.y.values))
        if name == "XGB":
            m.est.set_params(n_jobs=1)
        tau = threshold_for_fpr(va.y.values, m.predict_proba(va[cols].to_numpy(np.float32)), 0.01)
        p = m.predict_proba(Xte)
        caught_A = float((p[grpA] >= tau).mean())      # does the model really use `planted`?
        caught_B = float((p[grpB] >= tau).mean())
        det = metrics.detection_metrics(te.y.values, p, tau)
        print(f"{name}: trained in {time.time() - t0:.0f}s, F1 {det['f1']:.3f}, "
              f"planted attacks caught {caught_A:.2f}, real attacks caught {caught_B:.2f}", flush=True)
        for ex in explainers_for(name, m, Xtr, bg, cols):
            top_a, mass_a, top_b = [], [], []
            for i in grpA:
                phi = np.abs(ex.explain(Xte[i]))
                top_a.append(int(np.argmax(phi) == k))
                mass_a.append(phi[k] / max(phi.sum(), 1e-12))
            for i in grpB:
                phi = np.abs(ex.explain(Xte[i]))
                top_b.append(int(np.argmax(phi) == k))
            rows.append({"model": name, "explainer": ex.label, "f1": det["f1"], "caught_planted": caught_A,
                         "caught_real": caught_B, "hit_top1_A": float(np.mean(top_a)),
                         "mass_share_A": float(np.mean(mass_a)), "false_top1_B": float(np.mean(top_b)),
                         "n_A": len(grpA), "n_B": len(grpB)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "sanity_planted.csv", index=False)
    print("wrote", RESULTS / "sanity_planted.csv")


if __name__ == "__main__":
    main()
