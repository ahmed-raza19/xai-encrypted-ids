"""Detection, explanation-fidelity and explanation-consistency metrics (Phase 2, Step 8).

Conventions
-----------
* ``f`` maps an (n, d) array to P(attack) of shape (n,).
* ``phi`` is the attribution vector for the *attack* output, shape (d,).
* Fidelity metrics evaluate the explanation of the predicted class: for a flow
  predicted benign, attributions are negated (binary case, probability space).
* ``baseline`` is either a (d,) reference vector (e.g. training median) or a
  callable ``rng -> (d,)`` that draws a reference row (e.g. benign marginal).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix, balanced_accuracy_score)


# ----------------------------------------------------------------------------- detection

def detection_metrics(y_true, p, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, p)
        out["pr_auc"] = average_precision_score(y_true, p)
    return out


def threshold_for_fpr(y_val, p_val, fpr_target=0.01):
    """Smallest threshold whose validation false-positive rate is <= fpr_target."""
    neg = np.sort(np.asarray(p_val)[np.asarray(y_val) == 0])
    if len(neg) == 0:
        return 0.5
    k = int(np.floor((1 - fpr_target) * len(neg)))
    k = min(max(k, 0), len(neg) - 1)
    return float(np.nextafter(neg[k], 1.0))


# ----------------------------------------------------------------------------- helpers

def _signed_phi(phi, p_x, threshold):
    """Attribution toward the predicted class (binary)."""
    return phi if p_x >= threshold else -phi


def _target_out(fx, p_x, threshold):
    return fx if p_x >= threshold else 1.0 - fx


def _draw_baseline(baseline, rng):
    return baseline(rng) if callable(baseline) else baseline


# ----------------------------------------------------------------------------- fidelity

def faithfulness_correlation(f, x, phi, baseline, threshold, rng, n_subsets=50, subset_frac=0.2):
    """Bhatt et al. (2020): corr( sum_{i in S} phi_i , g(x) - g(x_[S <- baseline]) )."""
    d = x.shape[0]
    k = max(1, int(round(subset_frac * d)))
    p_x = float(f(x[None])[0])
    sphi = _signed_phi(phi, p_x, threshold)
    X = np.repeat(x[None], n_subsets, axis=0)
    sums = np.empty(n_subsets)
    for j in range(n_subsets):
        S = rng.choice(d, size=k, replace=False)
        b = _draw_baseline(baseline, rng)
        X[j, S] = b[S]
        sums[j] = sphi[S].sum()
    g0 = _target_out(p_x, p_x, threshold)
    gS = _target_out(f(X), p_x, threshold)
    drops = g0 - gS
    if np.std(sums) < 1e-12 or np.std(drops) < 1e-12:
        return 0.0
    return float(pearsonr(sums, drops)[0])


def deletion_aopc(f, x, phi, baseline, threshold, rng, K=10, n_random=5):
    """Samek et al. (2017) AOPC for most-relevant-first deletion, and the same quantity
    for random deletion orders. Returns (aopc, aopc_random)."""
    d = x.shape[0]
    K = min(K, d)
    p_x = float(f(x[None])[0])
    sphi = _signed_phi(phi, p_x, threshold)
    b = _draw_baseline(baseline, rng)

    def curve(order):
        X = np.repeat(x[None], K + 1, axis=0)
        for k in range(1, K + 1):
            X[k:, order[k - 1]] = b[order[k - 1]]
        g = _target_out(f(X), p_x, threshold)
        return float(np.mean(g[0] - g))

    aopc = curve(np.argsort(-sphi))
    rand = np.mean([curve(rng.permutation(d)) for _ in range(n_random)])
    return aopc, float(rand)


def infidelity(f, x, phi, scale, threshold, rng, n=50, sigma=0.1):
    """Yeh et al. (2019): E_I[(I^T phi - (g(x) - g(x - I)))^2], I ~ N(0, (sigma*scale)^2).
    ``scale`` is the per-feature standard deviation so noise is unit-free."""
    p_x = float(f(x[None])[0])
    sphi = _signed_phi(phi, p_x, threshold)
    I = rng.normal(0.0, sigma, size=(n, x.shape[0])) * scale
    g0 = _target_out(p_x, p_x, threshold)
    gI = _target_out(f(x[None] - I), p_x, threshold)
    lin = I @ sphi
    return float(np.mean((lin - (g0 - gI)) ** 2))


def max_sensitivity(explain_fn, x, phi, scale, rng, n=5, radius=0.05):
    """Yeh et al. (2019) max-sensitivity, reported relative to ||phi|| so explainers with
    different attribution scales are comparable."""
    norm = np.linalg.norm(phi) + 1e-12
    worst = 0.0
    for _ in range(n):
        delta = rng.uniform(-radius, radius, size=x.shape[0]) * scale
        worst = max(worst, float(np.linalg.norm(explain_fn(x + delta) - phi) / norm))
    return worst


def gini_sparsity(phi):
    a = np.sort(np.abs(np.asarray(phi, dtype=float)))
    if a.sum() <= 0:
        return 0.0
    n = len(a)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * a) / (n * a.sum()))


# ----------------------------------------------------------------------------- consistency

def topk(phi, k):
    return set(np.argsort(-np.abs(phi))[:k].tolist())


def jaccard_at_k(a, b, k):
    A, B = topk(a, k), topk(b, k)
    return len(A & B) / max(len(A | B), 1)


def rank_spearman(a, b):
    r = spearmanr(np.abs(a), np.abs(b)).correlation
    return 0.0 if np.isnan(r) else float(r)


def explanation_drift(mean_abs_a, mean_abs_b, k=5):
    """Delta_c = 1 - J_k between the mean |phi| profiles of one attack family in two
    environments."""
    return 1.0 - jaccard_at_k(mean_abs_a, mean_abs_b, k)
