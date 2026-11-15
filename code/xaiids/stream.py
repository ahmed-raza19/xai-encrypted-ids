"""Trace-driven discrete-event replay of the online streaming plane (Phase 2, E4).

Flows arrive by a Poisson process at rate lambda. A single main worker performs detection,
the trigger decision and any synchronous explanation; an asynchronous worker serves the deep
queue and deferred explanations. Detection, novelty scoring, cache look-ups and fast
explanations are executed for real and timed with ``perf_counter``; deep explanations
(KernelSHAP/LIME, hundreds of ms each) use service times drawn from the empirical distribution
measured on the same machine in the cost study (E1), so long replays stay tractable.
A virtual clock turns these service times into queueing delays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trigger import Decision, ExplanationCache, Novelty, Policy, TriggerParams


@dataclass
class Engines:
    model: object                 # detector with predict_proba
    fast: object                  # fast explainer (real compute)
    deep_times: np.ndarray        # empirical deep-explainer service times (s)
    fast_mean: float              # running estimate of fast-explainer cost (s)
    static_phi: np.ndarray        # global prior attribution (B3 / provisional)
    amortised: object | None      # surrogate explainer (B6)
    novelty: Novelty
    cache_features: np.ndarray    # indices of the top-m global features


def make_stream(y, family, n, attack_frac, burst=0, seed=0):
    """Indices of a mixed stream with the given attack prevalence. With ``burst`` > 0 a
    contiguous flood of ``burst`` attack flows of the most frequent family is inserted in the
    middle of the stream."""
    rng = np.random.default_rng(seed)
    ben = np.flatnonzero(y == 0)
    mal = np.flatnonzero(y == 1)
    n_mal = int(round(n * attack_frac))
    n_ben = n - n_mal
    s = np.concatenate([rng.choice(ben, n_ben, replace=len(ben) < n_ben),
                        rng.choice(mal, n_mal, replace=len(mal) < n_mal)])
    rng.shuffle(s)
    if burst:
        fam = pd.Series(family[mal]).value_counts().index[0]
        pool = mal[family[mal] == fam]
        flood = rng.choice(pool, burst, replace=len(pool) < burst)
        mid = len(s) // 2
        s = np.concatenate([s[:mid], flood, s[mid:]])
    return s


def replay(policy: Policy, params: TriggerParams, eng: Engines, X, y, arrivals, seed=0,
           keep_phi=0, deadline_deep=30.0):
    rng = np.random.default_rng(seed)
    beta = policy.beta if policy.beta is not None else params.beta
    cache = ExplanationCache(eng.cache_features, params.cache_ttl)
    bucket = params.bucket
    clock = 0.0
    fast_mean = eng.fast_mean
    rows, kept, async_rows = [], [], []
    X = np.asarray(X, dtype=np.float32)
    # Two asynchronous lanes, each one core, simulated exactly as FIFO queues:
    #  - deferred lane: fast explanations the budget guard pushed off the detection path
    #  - deep lane: KernelSHAP/LIME escalations with deadline-aware admission and de-duplication
    lane = {"fast": 0.0, "deep": 0.0}
    busy = {"fast": 0.0, "deep": 0.0}
    deep_seen, counters = {}, {"deep_shed": 0, "deep_dedup": 0}

    def enqueue(kind, t_enq, i, key):
        if kind == "deep":
            if key is not None and key in deep_seen and t_enq - deep_seen[key] <= params.cache_ttl:
                counters["deep_dedup"] += 1
                return
            s = float(rng.choice(eng.deep_times))
            start_l = max(lane["deep"], t_enq)
            if start_l + s - t_enq > deadline_deep:          # cannot meet the deadline: shed
                counters["deep_shed"] += 1
                return
            if key is not None:
                deep_seen[key] = t_enq
        else:
            s = fast_mean
            start_l = max(lane["fast"], t_enq)
        lane[kind] = start_l + s
        busy[kind] += s
        async_rows.append((i, kind, t_enq, lane[kind]))

    for i in range(len(X)):
        a = arrivals[i]
        start = max(clock, a)
        xi = X[i:i + 1]

        t0 = time.perf_counter()
        p = float(eng.model.predict_proba(xi)[0])
        t_det = time.perf_counter() - t0

        # backpressure: while the detection queue lags by more than params.backpressure, all
        # synchronous XAI work except cache look-ups is deferred and novelty scoring is skipped
        backlogged = policy.use_budget and (start - a) > params.backpressure

        t0 = time.perf_counter()
        nov = float(eng.novelty.score(xi)[0]) if (policy.rule_novel and not backlogged) else 0.0
        dec: Decision = policy.decide(p, nov, params, rng)
        t_trig = time.perf_counter() - t0

        # the budget guard charges every piece of synchronous XAI work: trigger, cache, explanation
        bucket = min(params.bucket, bucket + beta * t_det)
        if policy.use_budget:
            bucket -= t_trig
        yhat = int(p >= params.tau)
        delivered, t_sync, phi = "none", 0.0, None

        if dec.explain:
            k = None
            if policy.use_cache and dec.sync in ("fast", "deep"):
                t0 = time.perf_counter()
                k = cache.key(xi[0], yhat)
                phi = cache.get(k, start)
                dt = time.perf_counter() - t0
                t_sync += dt
                if policy.use_budget:
                    bucket -= dt
                if phi is not None:
                    delivered = "cache"
            if delivered == "none":
                if dec.sync == "static":
                    phi, delivered = eng.static_phi, "static"
                elif dec.sync == "amortised":
                    t0 = time.perf_counter()
                    phi = eng.amortised.explain(xi[0])
                    t_sync += time.perf_counter() - t0
                    delivered = "amortised"
                elif dec.sync == "fast":
                    if (not policy.use_budget) or (bucket >= fast_mean and not backlogged):
                        t0 = time.perf_counter()
                        phi = eng.fast.explain(xi[0])
                        dt = time.perf_counter() - t0
                        t_sync += dt
                        fast_mean = 0.9 * fast_mean + 0.1 * dt
                        if policy.use_budget:
                            bucket -= dt
                        delivered = "fast"
                        if k is not None:
                            cache.put(k, phi, start)
                    else:
                        delivered = "deferred"           # provisional global prior now,
                        enqueue("fast", start + t_det + t_trig + t_sync, i, None)
                elif dec.sync == "deep":
                    s = float(rng.choice(eng.deep_times))
                    if (not policy.use_budget) or (bucket >= s and not backlogged):
                        t_sync += s
                        if policy.use_budget:
                            bucket -= s
                        delivered = "deep_sync"
                        if k is not None:
                            cache.put(k, eng.static_phi, start)
                    else:
                        delivered = "deferred"
                        enqueue("deep", start + t_det + t_trig + t_sync, i, k)
            if dec.escalate and delivered not in ("cache",):
                enqueue("deep", start + t_det + t_trig + t_sync, i, k if policy.use_cache else None)

        finish = start + t_det + t_trig + t_sync
        clock = finish
        rows.append((i, a, start, t_det, t_trig, t_sync, finish, p, int(y[i]), yhat, dec.reason or "",
                     delivered, int(dec.escalate)))
        if keep_phi and phi is not None and delivered in ("fast", "cache", "static", "amortised") \
                and len(kept) < keep_phi and rng.random() < 0.5:
            kept.append((i, delivered, np.asarray(phi, dtype=np.float64)))

    df = pd.DataFrame(rows, columns=["i", "arrival", "start", "t_det", "t_trig", "t_sync", "finish", "p",
                                     "y", "yhat", "reason", "delivered", "escalated"])

    adf = pd.DataFrame(async_rows, columns=["i", "kind", "enqueued", "done"])
    adf.attrs.update(counters)
    adf.attrs["busy_fast"], adf.attrs["busy_deep"] = busy["fast"], busy["deep"]
    return df, adf, kept, busy["fast"] + busy["deep"]


def summarise(df, adf, t_async, params, deadline_fast=1.0, deadline_deep=30.0):
    n = len(df)
    det = df.t_det.sum()
    true_alert = (df.y == 1) & (df.yhat == 1)
    local = df.delivered.isin(["fast", "cache", "amortised", "deep_sync"])
    expl_lat = (df.finish - df.arrival)
    fast_cov = (true_alert & local & (expl_lat <= deadline_fast)).sum() / max(true_alert.sum(), 1)

    # deferred items count as covered if the async worker finishes them in time
    if len(adf):
        arr = df.set_index("i").arrival
        lat_async = adf.done.values - arr.loc[adf.i].values
        adf = adf.assign(lat=lat_async)
        deferred_ok = adf[(adf.kind == "fast") & (adf.lat <= deadline_fast)].i
        deep = adf[adf.kind == "deep"]
        requested = len(deep) + adf.attrs.get("deep_shed", 0)
        deep_cov = float((deep.lat <= deadline_deep).sum() / requested) if requested else np.nan
        deep_p95 = float(np.quantile(deep.lat, 0.95)) if len(deep) else np.nan
    else:
        deferred_ok = pd.Series([], dtype=int)
        deep_cov, deep_p95 = np.nan, np.nan
    covered_late = true_alert & df.i.isin(deferred_ok)
    fast_cov_any = (true_alert & ((local & (expl_lat <= deadline_fast)) | covered_late)).sum() / max(true_alert.sum(), 1)

    verdict_lat = df.start + df.t_det - df.arrival
    explained = df.delivered != "none"
    missed = (df.y == 1) & (df.yhat == 0)                 # false negatives of the detector
    fn_expl = {r: int((missed & explained & (df.reason == r)).sum()) for r in ("U", "N", "R")}
    return {
        "O_xai_sync": float((df.t_sync.sum() + df.t_trig.sum()) / det),
        "false_negatives": int(missed.sum()),
        "fn_explained_U": fn_expl["U"], "fn_explained_N": fn_expl["N"], "fn_explained_R": fn_expl["R"],
        "U_flows": int((df.reason == "U").sum()),
        "U_attack_share": float((df.y[df.reason == "U"] == 1).mean()) if (df.reason == "U").any() else np.nan,
        "flows": n,
        "true_alerts": int(true_alert.sum()),
        "trigger_rate": float(explained.mean()),
        "explained_local_rate": float(local.mean()),
        "O_sync": float(df.t_sync.sum() / det),
        "O_trigger": float(df.t_trig.sum() / det),
        "O_async": float(t_async / det),
        "O_async_fast": float(adf.attrs.get("busy_fast", 0.0) / det),
        "O_async_deep": float(adf.attrs.get("busy_deep", 0.0) / det),
        "deep_served": int((adf.kind == "deep").sum()) if len(adf) else 0,
        "deep_shed": int(adf.attrs.get("deep_shed", 0)),
        "deep_dedup": int(adf.attrs.get("deep_dedup", 0)),
        "coverage_fast@1s": float(fast_cov),
        "coverage_fast_incl_deferred@1s": float(fast_cov_any),
        "coverage_deep@30s": deep_cov,
        "deep_latency_p95_s": deep_p95,
        "verdict_latency_p50_ms": float(np.median(verdict_lat) * 1e3),
        "verdict_latency_p99_ms": float(np.quantile(verdict_lat, 0.99) * 1e3),
        "alert_expl_latency_p95_ms": float(np.quantile(expl_lat[true_alert & local], 0.95) * 1e3)
        if (true_alert & local).any() else np.nan,
        "backlog_s": float(max(0.0, df.finish.iloc[-1] - df.arrival.iloc[-1])),
        "cache_hit_rate": float((df.delivered == "cache").sum() / max(explained.sum(), 1)),
        "deferred": int((df.delivered == "deferred").sum()),
        "audit_explained": int((df.reason == "R").sum()),
        "audit_fn_found": int(((df.reason == "R") & (df.y == 1)).sum()),
        "reasons": df.reason.value_counts().to_dict(),
    }
