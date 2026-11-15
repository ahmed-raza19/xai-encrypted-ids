"""Statistical analysis helpers (Phase 2, Step 9)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, stat=np.mean, n_boot=1000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    return (float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def wilcoxon_rb(a, b):
    """Paired Wilcoxon signed-rank test with matched-pairs rank-biserial effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[d != 0]
    if len(d) < 2:
        return {"p": 1.0, "rank_biserial": 0.0, "n": len(d)}
    res = stats.wilcoxon(d)
    r = np.abs(d)
    ranks = stats.rankdata(r)
    rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    return {"p": float(res.pvalue), "rank_biserial": float(rb), "n": int(len(d))}


def friedman_nemenyi(table):
    """table: (n_blocks, k_methods) array of scores (higher = better).
    Returns Friedman statistic/p, mean ranks (1 = best) and the Nemenyi critical difference."""
    t = np.asarray(table, dtype=float)
    n, k = t.shape
    ranks = np.vstack([stats.rankdata(-row) for row in t])
    mean_ranks = ranks.mean(axis=0)
    if n < 2 or k < 2:
        return {"chi2": np.nan, "p": np.nan, "mean_ranks": mean_ranks, "cd": np.nan}
    chi2, p = stats.friedmanchisquare(*[t[:, j] for j in range(k)])
    # q_alpha for alpha = 0.05 (Demsar 2006, Table 5), k = 2..10
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    cd = q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": float(chi2), "p": float(p), "mean_ranks": mean_ranks, "cd": float(cd)}


def tost_paired(a, b, margin):
    """Two one-sided paired t-tests: equivalence if both p < alpha.
    H0: |mean(a - b)| >= margin."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 3:
        return {"p_lower": 1.0, "p_upper": 1.0, "equivalent": False, "mean_diff": float(np.mean(d)) if n else np.nan}
    se = np.std(d, ddof=1) / np.sqrt(n)
    if se == 0:
        eq = abs(np.mean(d)) < margin
        return {"p_lower": 0.0 if eq else 1.0, "p_upper": 0.0 if eq else 1.0, "equivalent": eq, "mean_diff": float(np.mean(d))}
    t_low = (np.mean(d) + margin) / se
    t_up = (np.mean(d) - margin) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_up = stats.t.cdf(t_up, n - 1)
    return {"p_lower": float(p_low), "p_upper": float(p_up), "equivalent": bool(max(p_low, p_up) < 0.05),
            "mean_diff": float(np.mean(d))}


def mcnemar(y_true, pred_a, pred_b):
    """Exact-binomial McNemar test on paired predictions."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p": p}
