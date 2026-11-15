"""E3 - explanation consistency under distribution shift (RQ3).

Part A  cross-network: CIC-IDS2017 (on-premise testbed) <-> CSE-CIC-IDS2018 (AWS), tiers T1/T3.
        For every shared attack family c:  detection recall in source vs target, and the drift
        Delta_c = 1 - Jaccard@5 between mean |phi| profiles, compared with two noise floors
        (random halves of the same flows; a second model trained with another seed).
Part B  temporal: validation (earlier) vs test (later) flows of the same dataset.
Part C  replication of the cross-dataset protocol of Kilic & Sengul (P9, IEEE Access 2026):
        multi-class RF trained on CIC-IDS2017, evaluated on CSE-CIC-IDS2018 (macro-F1).

Usage:  python scripts/05_e3_drift.py [crossnet] [temporal <ds> ...] [p9]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os as _os
if _os.environ.get("XAI_BG_CORES"):   # keep background analysis off the timing core
    import psutil as _ps
    _lo, _hi = map(int, _os.environ["XAI_BG_CORES"].split("-"))
    _ps.Process().cpu_affinity(list(range(_lo, _hi + 1)))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402
from sklearn.ensemble import RandomForestClassifier   # noqa: E402
from sklearn.metrics import f1_score                  # noqa: E402

from xaiids import data, explainers, metrics, models  # noqa: E402
from xaiids.config import RESULTS, SEED               # noqa: E402
from xaiids.tiers import tier_columns                 # noqa: E402

SHARED = ["DoS", "DDoS", "BruteForce", "WebAttack", "Botnet"]
N_PER_FAMILY = 300


def family_profile(ex, X):
    phis = np.vstack([ex.explain(x) for x in X])
    return np.abs(phis).mean(axis=0), phis


def sample_family(df, fam, n, rng):
    idx = np.flatnonzero(df.family.values == fam)
    if len(idx) == 0:
        return idx
    return rng.choice(idx, size=min(n, len(idx)), replace=False)


def train_xgb(Xtr, ytr, seed):
    m = models.make("XGB")
    m.est.set_params(random_state=seed)
    m.fit(Xtr, ytr, w=models.class_weights(ytr))
    return m


def crossnet(tiers=("T3", "T1")):
    envs = {e: data.load_processed(e) for e in ("cic2017", "cic2018")}
    common = sorted(set(envs["cic2017"].columns) & set(envs["cic2018"].columns) - set(data.META))
    rows, prof_rows = [], []
    for tier in tiers:
        feats = tier_columns(common, tier)
        mods, explns, taus = {}, {}, {}
        for e, df in envs.items():
            tr, va = df[df.split == "train"], df[df.split == "val"]
            Xtr = tr[feats].to_numpy(np.float32)
            m = train_xgb(Xtr, tr.label.values, SEED)
            m2 = train_xgb(Xtr, tr.label.values, SEED + 1)
            from xaiids.metrics import threshold_for_fpr
            taus[e] = threshold_for_fpr(va.label.values, m.predict_proba(va[feats].to_numpy(np.float32)), 0.01)
            bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
            mods[e] = (m, m2)
            explns[e] = (explainers.TreeSHAP(m, bg, "interventional"), explainers.TreeSHAP(m2, bg, "interventional"))
        for src, tgt in (("cic2017", "cic2018"), ("cic2018", "cic2017")):
            m, _ = mods[src]
            ex_src, ex_seed = explns[src]
            ex_tgt_own = explns[tgt][0]
            te_s = envs[src][envs[src].split == "test"]
            te_t = envs[tgt][envs[tgt].split == "test"]
            Xs, Xt = te_s[feats].to_numpy(np.float32), te_t[feats].to_numpy(np.float32)
            ps, pt = m.predict_proba(Xs), m.predict_proba(Xt)
            det_src = metrics.detection_metrics(te_s.label.values, ps, taus[src])
            det_tgt = metrics.detection_metrics(te_t.label.values, pt, taus[src])
            rows.append({"part": "overall", "tier": tier, "source": src, "target": tgt,
                         "f1_source": det_src["f1"], "f1_target": det_tgt["f1"],
                         "auc_source": det_src.get("roc_auc"), "auc_target": det_tgt.get("roc_auc"),
                         "fpr_target": det_tgt["fpr"]})
            rng = np.random.default_rng(SEED)
            for fam in SHARED:
                i_s, i_t = sample_family(te_s, fam, N_PER_FAMILY, rng), sample_family(te_t, fam, N_PER_FAMILY, rng)
                if len(i_s) < 20 or len(i_t) < 20:
                    continue
                rec_s = float((ps[te_s.family.values == fam] >= taus[src]).mean())
                rec_t = float((pt[te_t.family.values == fam] >= taus[src]).mean())
                prof_s, phis_s = family_profile(ex_src, Xs[i_s])
                prof_t, _ = family_profile(ex_src, Xt[i_t])             # same model, new network
                prof_own, _ = family_profile(ex_tgt_own, Xt[i_t])       # each network's own model
                prof_seed, _ = family_profile(ex_seed, Xs[i_s])         # seed noise floor
                halves = []
                for r in range(5):
                    perm = np.random.default_rng(r).permutation(len(i_s))
                    a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
                    halves.append(metrics.jaccard_at_k(np.abs(phis_s[a]).mean(0), np.abs(phis_s[b]).mean(0), 5))
                rows.append({"part": "family", "tier": tier, "source": src, "target": tgt, "family": fam,
                             "n_source": len(i_s), "n_target": len(i_t),
                             "recall_source": rec_s, "recall_target": rec_t, "recall_drop": rec_s - rec_t,
                             "J5_halves": float(np.mean(halves)),
                             "J5_seed": metrics.jaccard_at_k(prof_s, prof_seed, 5),
                             "J5_crossnet_same_model": metrics.jaccard_at_k(prof_s, prof_t, 5),
                             "J5_crossnet_own_models": metrics.jaccard_at_k(prof_s, prof_own, 5),
                             "rho_crossnet_same_model": metrics.rank_spearman(prof_s, prof_t),
                             "drift_same_model": metrics.explanation_drift(prof_s, prof_t, 5),
                             "drift_own_models": metrics.explanation_drift(prof_s, prof_own, 5)})
                top = lambda p: [feats[k] for k in np.argsort(-p)[:5]]
                prof_rows.append({"tier": tier, "source": src, "target": tgt, "family": fam,
                                  "top5_source": top(prof_s), "top5_target_same_model": top(prof_t),
                                  "top5_target_own_model": top(prof_own)})
                print(f"  {tier} {src}->{tgt} {fam:10s} rec {rec_s:.3f}->{rec_t:.3f}  J5 halves={np.mean(halves):.2f} "
                      f"seed={rows[-1]['J5_seed']:.2f} crossnet(same)={rows[-1]['J5_crossnet_same_model']:.2f} "
                      f"(own)={rows[-1]['J5_crossnet_own_models']:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e3_crossnet.csv", index=False)
    pd.DataFrame(prof_rows).to_json(RESULTS / "e3_crossnet_top5.json", orient="records", indent=1)
    fam = out[out.part == "family"]
    if len(fam) >= 5:
        r = spearmanr(fam.drift_same_model, fam.recall_drop)
        (RESULTS / "e3_crossnet_corr.json").write_text(json.dumps(
            {"spearman_drift_vs_recall_drop": float(r.correlation), "p": float(r.pvalue), "n": len(fam)}, indent=1))
        print("Spearman(drift, recall drop) =", round(r.correlation, 3), "p =", round(r.pvalue, 4), "n =", len(fam))


def temporal(datasets, tier="T3"):
    rows = []
    for ds in datasets:
        df = data.load_processed(ds)
        feats = tier_columns([c for c in df.columns if c not in data.META], tier)
        m = models.load(ds, tier, "XGB")
        Xtr = df[df.split == "train"][feats].to_numpy(np.float32)
        ex = explainers.TreeSHAP(m, explainers.background_kmeans(Xtr, 50, seed=SEED), "interventional")
        va, te = df[df.split == "val"], df[df.split == "test"]
        rng = np.random.default_rng(SEED)
        for fam in sorted(set(df.family) - {"Benign"}):
            i_v, i_t = sample_family(va, fam, N_PER_FAMILY, rng), sample_family(te, fam, N_PER_FAMILY, rng)
            if len(i_v) < 20 or len(i_t) < 20:
                continue
            pv, phis_v = family_profile(ex, va[feats].to_numpy(np.float32)[i_v])
            pt, _ = family_profile(ex, te[feats].to_numpy(np.float32)[i_t])
            halves = []
            for r in range(5):
                perm = np.random.default_rng(r).permutation(len(i_v))
                a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
                halves.append(metrics.jaccard_at_k(np.abs(phis_v[a]).mean(0), np.abs(phis_v[b]).mean(0), 5))
            rows.append({"dataset": ds, "tier": tier, "family": fam, "J5_halves": float(np.mean(halves)),
                         "J5_val_vs_test": metrics.jaccard_at_k(pv, pt, 5), "rho_val_vs_test": metrics.rank_spearman(pv, pt)})
            print(f"  temporal {ds} {fam:14s} J5 halves={np.mean(halves):.2f} val->test={rows[-1]['J5_val_vs_test']:.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "e3_temporal.csv", index=False)


def p9_replication():
    """Kilic & Sengul (2026): RF trained on CIC-IDS2017, tested on CSE-CIC-IDS2018, macro-F1."""
    a, b = data.load_processed("cic2017"), data.load_processed("cic2018")
    common = sorted(set(a.columns) & set(b.columns) - set(data.META))
    classes = ["Benign"] + SHARED
    out = []
    for tier in ("T1", "T3"):
        feats = tier_columns(common, tier)
        tr = a[(a.split == "train") & a.family.isin(classes)]
        te_in = a[(a.split == "test") & a.family.isin(classes)]
        te_x = b[(b.split == "test") & b.family.isin(classes)]
        rf = RandomForestClassifier(n_estimators=300, min_samples_split=10, min_samples_leaf=4,
                                    max_features="sqrt", n_jobs=-1, random_state=SEED)
        rf.fit(tr[feats].to_numpy(np.float32), tr.family.values)
        for name, te in (("in-domain CIC-IDS2017", te_in), ("cross-dataset CSE-CIC-IDS2018", te_x)):
            pred = rf.predict(te[feats].to_numpy(np.float32))
            out.append({"tier": tier, "evaluation": name, "n": len(te),
                        "macro_f1": f1_score(te.family, pred, average="macro", labels=classes, zero_division=0),
                        "weighted_f1": f1_score(te.family, pred, average="weighted", zero_division=0),
                        "accuracy": float((pred == te.family.values).mean())})
            print("  P9 replication", out[-1])
    pd.DataFrame(out).to_csv(RESULTS / "e3_p9_replication.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "crossnet" in args:
        crossnet()
    if "temporal" in args:
        i = args.index("temporal")
        ds = [a for a in args[i + 1:] if a not in ("crossnet", "p9")]
        temporal(ds)
    if "p9" in args:
        p9_replication()
