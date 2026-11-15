"""E1 - explanation cost profiling (RQ1).

Orchestrator:  python scripts/03_e1_cost.py run <dataset> [<dataset> ...] [--tiers T3,T1,T2]
Worker (internal): python scripts/03_e1_cost.py worker <ds> <tier> <model> <method> <budget>

Every configuration runs in a fresh process pinned to one CPU core with single-threaded
libraries; peak resident memory is sampled while the explainer is built and used. Results are
written to results/e1/<config>.json and attributions to results/e1/<config>.npz, so the run is
resumable and the fidelity study (E2) reuses the exact explanations whose cost was measured.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = Path(__import__("os").environ.get("XAI_E1_DIR", ROOT.parent / "results" / "e1"))
OUT.mkdir(parents=True, exist_ok=True)

# Reduced plan for T1/T2 (tier comparison of cost and fidelity); full plan on T3.
TIER_PLAN = {
    "XGB": [("TreeSHAP-path", [None]), ("TreeSHAP-interv", [None]), ("KernelSHAP", [500]), ("LIME", [1000])],
    "MLP": [("IG", [32]), ("DeepSHAP", [None]), ("KernelSHAP", [500]), ("LIME", [1000])],
}


def cfg_name(ds, tier, model, method, budget):
    return f"{ds}__{tier}__{model}__{method}__{budget}"


def worker(ds, tier, model_name, method, budget):
    from xaiids.profiling import PeakRSS, pin_single_core
    pin_single_core()
    import numpy as np
    from xaiids import data, explainers, models
    from xaiids.config import N_WARMUP, SEED
    from xaiids.tiers import tier_columns

    budget = None if budget == "None" else int(budget)
    df = data.load_processed(ds)
    feats = tier_columns([c for c in df.columns if c not in data.META], tier)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    rng = np.random.default_rng(SEED)
    expensive = (method == "KernelSHAP" and (budget or 0) >= 2048) or method == "PartitionSHAP"
    n_each = 50 if expensive else 100
    # same permutation for every configuration -> cheap configs explain 100+100 flows and
    # expensive ones the first 50+50 of the *same* flows, enabling paired comparisons
    idx = np.concatenate([rng.permutation(np.flatnonzero(te.label.values == c))[:n_each] for c in (1, 0)])
    X = te[feats].to_numpy(np.float32)[idx]
    Xtr = tr[feats].to_numpy(np.float32)

    model = models.load(ds, tier, model_name)
    if model.name in ("RF", "XGB"):
        model.est.set_params(n_jobs=1)
    bg = explainers.background_kmeans(Xtr, 50, seed=SEED)
    lime_sample = Xtr[rng.choice(len(Xtr), size=min(len(Xtr), 5000), replace=False)]
    median = np.median(Xtr, axis=0)

    with PeakRSS() as mem:
        t0 = time.perf_counter()
        ex = explainers.build(model, method, budget, bg, lime_sample, feats, median)
        build_s = time.perf_counter() - t0
        for i in range(N_WARMUP):
            ex.explain(X[i % len(X)])
        times, phis = [], []
        for x in X:
            t0 = time.perf_counter()
            phi = ex.explain(x)
            times.append(time.perf_counter() - t0)
            phis.append(phi)
    p = model.predict_proba(X)
    name = cfg_name(ds, tier, model_name, method, budget)
    np.savez_compressed(OUT / f"{name}.npz", idx=idx, phi=np.vstack(phis), times=np.array(times), p=p,
                        y=te.label.values[idx], family=te.family.values[idx].astype(str))
    t = np.array(times)
    res = {"dataset": ds, "tier": tier, "model": model_name, "method": method, "budget": budget,
           "features": len(feats), "n": len(t), "build_s": build_s,
           "mean_ms": float(t.mean() * 1e3), "p50_ms": float(np.median(t) * 1e3),
           "p95_ms": float(np.quantile(t, 0.95) * 1e3), "p99_ms": float(np.quantile(t, 0.99) * 1e3),
           "max_ms": float(t.max() * 1e3), "throughput_per_s": float(1.0 / t.mean()),
           "peak_mem_increment_mb": mem.increment_mb}
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def configs(ds, tiers):
    from xaiids.explainers import PLAN
    for tier in tiers:
        plan = PLAN if tier == "T3" else TIER_PLAN
        for model_name, methods in plan.items():
            if model_name == "EBM" and tier != "T3":
                continue
            for method, budgets in methods:
                for b in budgets:
                    yield ds, tier, model_name, method, str(b)


def run(datasets, tiers):
    py = sys.executable
    todo = [c for ds in datasets for c in configs(ds, tiers)]
    print(f"{len(todo)} configurations", flush=True)
    for k, c in enumerate(todo, 1):
        name = cfg_name(*c)
        if (OUT / f"{name}.json").exists():
            continue
        t0 = time.time()
        r = subprocess.run([py, __file__, "worker", *c], capture_output=True, text=True, timeout=7200)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"[{k}/{len(todo)}] {name} {status} {time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            (OUT / f"{name}.err").write_text(r.stderr[-4000:])


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(*sys.argv[2:7])
    else:
        args = sys.argv[2:]
        tiers = ["T3"]
        if "--tiers" in args:
            i = args.index("--tiers")
            tiers = args[i + 1].split(",")
            args = args[:i] + args[i + 2:]
        run(args, tiers)
