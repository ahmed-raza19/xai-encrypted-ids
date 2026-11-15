"""E4 ablation A6 (second half): sensitivity of the proposed trigger to its rule parameters.

Phase 2 lists A6 as "vary beta, delta, kappa, rho". beta is varied in 06_e4_stream.py; this script
varies the uncertainty band (delta via the validation band fraction), the novelty threshold
(kappa via the validation quantile) and the audit rate rho, one at a time, at 75% offered load on
exactly the streams and arrival times of 06_e4_stream.py.

Usage:  python scripts/06b_e4_sensitivity.py <dataset> [--model XGB] [--tier T3]
Writes results/e4sens_<dataset>_<model>_<tier>.csv
"""
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("e4", ROOT / "scripts" / "06_e4_stream.py")
e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4)                    # pins the process to one core, as in E4

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from xaiids import data, stream, trigger        # noqa: E402
from xaiids.config import RESULTS, SEED, TRIGGER  # noqa: E402

LOAD = 0.75
VARIANTS = [("default", None)] + [("band_frac", v) for v in (0.02, 0.10)] + \
           [("novelty_q", v) for v in (0.95, 0.999)] + [("rho", v) for v in (0.001, 0.02)]


def main(ds, mname="XGB", tier="T3", n_flows=20000):
    fast_method = "IG" if mname == "MLP" else "TreeSHAP-path"
    eng, ctx = e4.setup(ds, tier, mname, fast_method)
    base = ctx["params"]
    df_all = data.load_processed(ds)
    va = df_all[df_all.split == "val"]
    Xva = va[ctx["feats"]].to_numpy(np.float32)
    p_va, nov_va = ctx["model"].predict_proba(Xva), eng.novelty.score(Xva)
    del df_all

    te = ctx["te"]
    y_all, fam_all = te.label.to_numpy(), te.family.to_numpy()
    base_idx = stream.make_stream(y_all, fam_all, 3000, 0.05, seed=SEED)
    b1 = trigger.Policy.baselines(0.0)[0]
    d0, _, _, _ = stream.replay(b1, base, eng, ctx["Xte"][base_idx], y_all[base_idx], np.arange(3000) * 1.0)
    mu = 1.0 / float(d0.t_det.mean() + d0.t_trig.mean())
    scenarios = {"mixed": stream.make_stream(y_all, fam_all, n_flows, 0.05, seed=SEED),
                 "burst": stream.make_stream(y_all, fam_all, n_flows, 0.05, burst=3000, seed=SEED)}
    proposed = trigger.Policy.proposed()
    rows = []
    for scen, idx in scenarios.items():
        Xs, ys = ctx["Xte"][idx], y_all[idx]
        arr = np.cumsum(np.random.default_rng(SEED + int(LOAD * 100)).exponential(1.0 / (LOAD * mu), len(idx)))
        for pname, val in VARIANTS:
            if pname == "band_frac":
                cal = trigger.calibrate(va.label.values, p_va, nov_va, fpr=TRIGGER["fpr_target"], band_frac=val,
                                        novelty_q=TRIGGER["novelty_q"])
                params = dataclasses.replace(base, delta=cal.delta)
            elif pname == "novelty_q":
                params = dataclasses.replace(base, kappa=float(np.quantile(nov_va, val)))
            elif pname == "rho":
                params = dataclasses.replace(base, rho=val)
            else:
                params = base
            t0 = time.time()
            df, adf, _, t_async = stream.replay(proposed, params, eng, Xs, ys, arr, seed=SEED)
            s = stream.summarise(df, adf, t_async, params)
            rows.append({"dataset": ds, "model": mname, "tier": tier, "scenario": scen, "load": LOAD,
                         "param": pname, "value": val if val is not None else np.nan,
                         "delta": params.delta, "kappa": params.kappa, "rho": params.rho,
                         **{k: v for k, v in s.items() if k != "reasons"}, "reasons": json.dumps(s["reasons"])})
            print(f"  {scen:5s} {pname:9s} {str(val):6s} rate={s['trigger_rate']:.3f} O_xai={s['O_xai_sync']:.3f} O_sync={s['O_sync']:.3f} "
                  f"O_async={s['O_async']:.2f} cov@1s={s['coverage_fast@1s']:.3f} "
                  f"(+def {s['coverage_fast_incl_deferred@1s']:.3f}) deep@30s={s['coverage_deep@30s']:.3f} "
                  f"p99={s['verdict_latency_p99_ms']:.1f}ms [{time.time() - t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"e4sens_{ds}_{mname}_{tier}.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {}
    for flag, key in (("--model", "mname"), ("--tier", "tier")):
        if flag in args:
            i = args.index(flag)
            kw[key] = args[i + 1]
            args = args[:i] + args[i + 2:]
    for ds in args:
        main(ds, **kw)
