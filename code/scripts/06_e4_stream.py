"""E4 - budget-aware selective XAI trigger under streaming load (RQ4), baselines B1-B6 and
ablations A1-A6.

Usage:  python scripts/06_e4_stream.py <dataset> [--model XGB] [--tier T3] [--flows 20000]
Writes results/e4_<dataset>.csv (one row per policy x load x scenario),
       results/e4_<dataset>_windows.parquet (per-1000-flow window metrics for paired tests),
       results/e4_<dataset>_fidelity.csv (fidelity of delivered explanations).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xaiids.profiling import pin_single_core   # noqa: E402
pin_single_core()

import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

from xaiids import data, explainers, metrics, models, stream, trigger  # noqa: E402
from xaiids.config import RESULTS, SEED, TRIGGER  # noqa: E402
from xaiids.tiers import tier_columns          # noqa: E402

LOADS = [0.25, 0.75, 1.5]
WINDOW = 1000


class Surrogate:
    """FastSHAP-style amortised explainer: scaled features -> reference attributions."""

    def __init__(self, scaler, reg):
        self.scaler, self.reg = scaler, reg

    def predict(self, X):
        return self.reg.predict(self.scaler.transform(X))


def setup(ds, tier, mname, fast_method):
    df = data.load_processed(ds)
    feats = tier_columns([c for c in df.columns if c not in data.META], tier)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = (d[feats].to_numpy(np.float32) for d in (tr, va, te))
    model = models.load(ds, tier, mname)
    if mname in ("RF", "XGB"):
        model.est.set_params(n_jobs=1)
    rng = np.random.default_rng(SEED)
    bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
    median = np.median(Xtr, axis=0)
    fast = explainers.build(model, fast_method, None if fast_method != "IG" else 16, bg, None, feats, median)

    nov = trigger.Novelty(Xtr)
    p_va = model.predict_proba(Xva)
    params = trigger.calibrate(va.label.values, p_va, nov.score(Xva), fpr=TRIGGER["fpr_target"],
                               band_frac=TRIGGER["band_frac"], novelty_q=TRIGGER["novelty_q"])

    # global prior (B3 / provisional) and cache-key features from fast attributions on training alerts
    tr_idx = rng.choice(len(Xtr), size=min(3000, len(Xtr)), replace=False)
    phis_tr = np.vstack([fast.explain(x) for x in Xtr[tr_idx]])
    yhat_tr = model.predict_proba(Xtr[tr_idx]) >= params.tau
    static_phi = phis_tr[yhat_tr].mean(axis=0) if yhat_tr.any() else phis_tr.mean(axis=0)
    cache_feats = np.argsort(-np.abs(phis_tr).mean(axis=0))[:TRIGGER["cache_top_m"]]

    # amortised surrogate (B6) trained to reproduce the fast explainer
    sc = models.Scaler().fit(Xtr)
    reg = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=SEED)
    reg.fit(sc.transform(Xtr[tr_idx]), phis_tr)
    amort = explainers.Amortised(Surrogate(sc, reg))

    # deep-explainer service-time distribution measured in E1 on this machine
    e1dir = RESULTS / "e1_isolated" if (RESULTS / "e1_isolated").exists() else RESULTS / "e1"
    e1 = e1dir / f"{ds}__{tier}__{mname}__KernelSHAP__500.npz"
    deep_times = np.load(e1)["times"] if e1.exists() else np.array([0.5])
    fast_mean = float(np.mean([_t(fast.explain, x) for x in Xte[:50]]))
    eng = stream.Engines(model=model, fast=fast, deep_times=deep_times, fast_mean=fast_mean,
                         static_phi=static_phi, amortised=amort, novelty=nov, cache_features=cache_feats)
    ctx = dict(Xte=Xte, te=te, feats=feats, median=median, model=model, fast=fast, params=params)
    return eng, ctx


def _t(fn, x):
    t0 = time.perf_counter()
    fn(x)
    return time.perf_counter() - t0


def delivered_fidelity(kept, ctx, eng, rng, tau):
    """Faithfulness of delivered explanations and their agreement with a fresh exact one."""
    out = []
    for i, kind, phi in kept:
        x = ctx["Xs"][i]
        fresh = ctx["fast"].explain(x)
        # identical random subsets for both attributions, so the comparison is paired exactly
        ra, rb = np.random.default_rng(SEED + int(i)), np.random.default_rng(SEED + int(i))
        out.append({"i": i, "delivered": kind,
                    "faith": metrics.faithfulness_correlation(ctx["model"].predict_proba, x, phi, ctx["median"], tau, ra),
                    "faith_exact": metrics.faithfulness_correlation(ctx["model"].predict_proba, x, fresh, ctx["median"], tau, rb),
                    "jaccard5_vs_exact": metrics.jaccard_at_k(phi, fresh, 5)})
    return out


def windows(df, adf, name, load, scen):
    rows = []
    for w0 in range(0, len(df), WINDOW):
        d = df.iloc[w0:w0 + WINDOW]
        ta = (d.y == 1) & (d.yhat == 1)
        local = d.delivered.isin(["fast", "cache", "amortised", "deep_sync"])
        lat = d.finish - d.arrival
        rows.append({"policy": name, "load": load, "scenario": scen, "window": w0 // WINDOW,
                     "O_sync": d.t_sync.sum() / d.t_det.sum(),
                     "O_xai_sync": (d.t_sync.sum() + d.t_trig.sum()) / d.t_det.sum(),
                     "coverage": ((ta & local & (lat <= 1.0)).sum() / ta.sum()) if ta.sum() else np.nan,
                     "verdict_p99_ms": float(np.quantile(d.start + d.t_det - d.arrival, 0.99) * 1e3)})
    return rows


def main(ds, mname="XGB", tier="T3", n_flows=20000, fast_method=None):
    fast_method = fast_method or ("IG" if mname == "MLP" else "TreeSHAP-path")
    eng, ctx = setup(ds, tier, mname, fast_method)
    params = ctx["params"]
    te = ctx["te"]
    y_all, fam_all = te.label.to_numpy(), te.family.to_numpy()
    rng = np.random.default_rng(SEED)

    # detection-only saturation throughput (B1 at negligible load)
    base_idx = stream.make_stream(y_all, fam_all, 3000, 0.05, seed=SEED)
    b1 = trigger.Policy.baselines(0.0)[0]
    d0, _, _, _ = stream.replay(b1, params, eng, ctx["Xte"][base_idx], y_all[base_idx], np.arange(3000) * 1.0)
    mu = 1.0 / float(d0.t_det.mean() + d0.t_trig.mean())

    proposed = trigger.Policy.proposed()
    scenarios = {"mixed": stream.make_stream(y_all, fam_all, n_flows, 0.05, seed=SEED),
                 "burst": stream.make_stream(y_all, fam_all, n_flows, 0.05, burst=3000, seed=SEED)}
    rows, wrows, frows = [], [], []
    meta = {"dataset": ds, "model": mname, "tier": tier, "fast": fast_method, "mu_flows_per_s": mu,
            "tau": params.tau, "delta": params.delta, "kappa": params.kappa, "tau_high": params.tau_high,
            "fast_mean_ms": eng.fast_mean * 1e3, "deep_mean_ms": float(np.mean(eng.deep_times) * 1e3)}
    print(json.dumps(meta), flush=True)

    for scen, idx in scenarios.items():
        Xs, ys = ctx["Xte"][idx], y_all[idx]
        ctx["Xs"] = Xs
        # trigger rate of the proposed policy at moderate load -> matched rate for B4
        arr = np.cumsum(rng.exponential(1.0 / (0.75 * mu), len(idx)))
        dprop, aprop, _, tas = stream.replay(proposed, params, eng, Xs, ys, arr)
        pi = float((dprop.delivered != "none").mean())
        policies = trigger.Policy.baselines(pi) + [proposed]
        for load in LOADS:
            arr = np.cumsum(np.random.default_rng(SEED + int(load * 100)).exponential(1.0 / (load * mu), len(idx)))
            todo = list(policies) + (trigger.Policy.ablations() if load == 0.75 else [])
            for pol in todo:
                t0 = time.time()
                keep = 300 if (load == 0.75 and pol.name in ("Proposed", "B2a Explain-all (fast)", "B3 Static global",
                                                             "B6 Amortised (all flows)", "A3 no cache")) else 0
                df, adf, kept, t_async = stream.replay(pol, params, eng, Xs, ys, arr, seed=SEED, keep_phi=keep)
                s = stream.summarise(df, adf, t_async, params)
                rows.append({"dataset": ds, "model": mname, "tier": tier, "scenario": scen, "load": load,
                             "lambda_flows_per_s": load * mu, "policy": pol.name, **{k: v for k, v in s.items() if k != "reasons"},
                             "reasons": json.dumps(s["reasons"])})
                wrows += windows(df, adf, pol.name, load, scen)
                if keep:
                    for r in delivered_fidelity(kept, ctx, eng, np.random.default_rng(SEED), params.tau):
                        frows.append({"scenario": scen, "policy": pol.name, **r})
                print(f"  {scen:5s} load={load:<4} {pol.name:30s} O_xai={s['O_xai_sync']:.3f} O_sync={s['O_sync']:.3f} O_async={s['O_async']:.2f} "
                      f"cov@1s={s['coverage_fast@1s']:.3f} (+def {s['coverage_fast_incl_deferred@1s']:.3f}) "
                      f"deep@30s={s['coverage_deep@30s']:.3f} p99={s['verdict_latency_p99_ms']:.1f}ms "
                      f"backlog={s['backlog_s']:.1f}s hit={s['cache_hit_rate']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    tag = f"{ds}_{mname}_{tier}"
    pd.DataFrame(rows).to_csv(RESULTS / f"e4_{tag}.csv", index=False)
    pd.DataFrame(wrows).to_parquet(RESULTS / f"e4_{tag}_windows.parquet", index=False)
    pd.DataFrame(frows).to_csv(RESULTS / f"e4_{tag}_fidelity.csv", index=False)
    (RESULTS / f"e4_{tag}_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {}
    for flag, key, cast in (("--model", "mname", str), ("--tier", "tier", str), ("--flows", "n_flows", int),
                            ("--fast", "fast_method", str)):
        if flag in args:
            i = args.index(flag)
            kw[key] = cast(args[i + 1])
            args = args[:i] + args[i + 2:]
    for ds in args:
        main(ds, **kw)
