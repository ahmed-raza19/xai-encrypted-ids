"""Budget-aware selective XAI trigger (Phase 2, Step 5, Algorithm 1) and the explanation
policies it is compared with (baselines B1-B6) and ablated into (A1-A6).

A policy decides, per flow and after detection, whether an explanation is produced, by which
explainer, and whether it is synchronous (blocks the detection path) or deferred to the
asynchronous deep queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf

from .models import Scaler


# ----------------------------------------------------------------------------- calibration of rules

@dataclass
class TriggerParams:
    tau: float                  # alert threshold (validation FPR <= 1%)
    delta: float                # half-width of the uncertainty band around tau (5% of val flows)
    kappa: float                # Mahalanobis novelty threshold (99th pct of validation)
    tau_high: float             # high-severity threshold (validation FPR <= 0.1%)
    rho: float = 0.005          # benign audit rate
    beta: float = 0.10          # synchronous overhead budget
    bucket: float = 0.05        # token-bucket capacity (s): ~50 fast explanations of burst
    cache_ttl: float = 60.0     # s
    top_m: int = 10             # features in the cache key
    backpressure: float = 0.1   # s of detection-queue delay above which all sync XAI work is deferred


class Novelty:
    """Mahalanobis distance in the scaled feature space with Ledoit-Wolf shrinkage.

    For per-flow speed the scaler, the Ledoit-Wolf location and the Cholesky factor of the
    precision matrix are folded into one offset c and one matrix W, so that
    d(x) = || (slog(x) - c) W ||  with slog(x) = sign(x) log(1 + |x|)."""

    def __init__(self, X_train, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=min(len(X_train), 20000), replace=False)
        self.scaler = Scaler().fit(X_train[idx])
        Z = self.scaler.transform(X_train[idx]).astype(np.float64)
        lw = LedoitWolf().fit(Z)
        self.mu = lw.location_
        self.P = lw.precision_
        L = np.linalg.cholesky(self.P)                       # P = L L^T
        self.c = self.scaler.mu + self.scaler.sd * self.mu   # offset in signed-log space
        self.W = L / self.scaler.sd[:, None]

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        V = (np.copysign(np.log1p(np.abs(X)), X) - self.c) @ self.W
        if V.ndim == 1:
            return np.array([np.sqrt(V @ V)])
        if V.shape[0] == 1:                                  # the per-flow case in streaming
            return np.array([np.sqrt(V[0] @ V[0])])
        return np.sqrt(np.einsum("ij,ij->i", V, V))


def calibrate(y_val, p_val, nov_val, fpr=0.01, band_frac=0.05, novelty_q=0.99, fpr_high=0.001):
    neg = np.sort(p_val[y_val == 0])

    def thr(f):
        k = min(max(int(np.floor((1 - f) * len(neg))), 0), len(neg) - 1)
        return float(np.nextafter(neg[k], 1.0))

    tau = thr(fpr)
    tau_high = max(thr(fpr_high), tau)
    # smallest delta such that |p - tau| <= delta covers band_frac of validation flows
    delta = float(np.quantile(np.abs(p_val - tau), band_frac))
    kappa = float(np.quantile(nov_val, novelty_q))
    return TriggerParams(tau=tau, delta=delta, kappa=kappa, tau_high=tau_high)


# ----------------------------------------------------------------------------- cache

class ExplanationCache:
    def __init__(self, key_features, ttl):
        self.kf = np.asarray(key_features)
        self.ttl = ttl
        self.store = {}

    def key(self, x, yhat):
        v = np.asarray(x)[self.kf]
        q = np.floor(2.0 * np.log2(1.0 + np.abs(v))) * np.sign(v)
        return (int(yhat),) + tuple(q.astype(int).tolist())

    def get(self, k, now):
        hit = self.store.get(k)
        if hit is not None and now - hit[1] <= self.ttl:
            return hit[0]
        return None

    def put(self, k, phi, now):
        self.store[k] = (phi, now)


# ----------------------------------------------------------------------------- decision

@dataclass
class Decision:
    explain: bool = False
    reason: str = ""            # A / U / N / R
    sync: str = "none"          # none | fast | deep | cache | static | amortised
    escalate: bool = False      # add to asynchronous deep queue


@dataclass
class Policy:
    """Configurable explanation policy. The proposed trigger is ``Policy.proposed()``;
    baselines and ablations are produced by switching components off."""
    name: str
    rule_alert: bool = True
    rule_uncertain: bool = True
    rule_novel: bool = True
    rule_audit: bool = True
    use_cache: bool = True
    use_budget: bool = True
    escalate_deep: bool = True
    sync_explainer: str = "fast"      # fast | deep | static | amortised
    explain_all: bool = False         # B2 / B6: every flow
    random_rate: float | None = None  # B4: uniform random at this rate
    beta: float | None = None         # override of params.beta

    # ---- factory methods ------------------------------------------------------------------
    @classmethod
    def proposed(cls, **kw):
        return cls(name=kw.pop("name", "Proposed"), **kw)

    @classmethod
    def baselines(cls, trigger_rate):
        return [
            cls(name="B1 Explain-none", rule_alert=False, rule_uncertain=False, rule_novel=False,
                rule_audit=False, use_cache=False, use_budget=False, escalate_deep=False),
            cls(name="B2a Explain-all (fast)", explain_all=True, rule_novel=False, use_cache=False, use_budget=False,
                escalate_deep=False),
            cls(name="B2b Explain-all (deep)", explain_all=True, rule_novel=False, use_cache=False, use_budget=False,
                escalate_deep=False, sync_explainer="deep"),
            cls(name="B3 Static global", rule_uncertain=False, rule_novel=False, rule_audit=False,
                use_cache=False, use_budget=False, escalate_deep=False, sync_explainer="static"),
            cls(name="B4 Random (matched rate)", rule_alert=False, rule_uncertain=False, rule_novel=False,
                rule_audit=False, use_cache=False, use_budget=False, escalate_deep=False,
                random_rate=trigger_rate),
            cls(name="B5 Alert-only", rule_uncertain=False, rule_novel=False, rule_audit=False,
                use_cache=False, use_budget=False, escalate_deep=False),
            cls(name="B6 Amortised (all flows)", explain_all=True, rule_novel=False, use_cache=False, use_budget=False,
                escalate_deep=False, sync_explainer="amortised"),
        ]

    @classmethod
    def ablations(cls):
        return [
            cls(name="A1 no uncertainty rule", rule_uncertain=False),
            cls(name="A2 no novelty rule", rule_novel=False),
            cls(name="A3 no cache", use_cache=False),
            cls(name="A4 no budget guard", use_budget=False),
            cls(name="A5a fast-only (no deep tier)", escalate_deep=False),
            cls(name="A5b deep-only (sync deep)", sync_explainer="deep", escalate_deep=False),
        ] + [cls(name=f"A6 beta={b}", beta=b) for b in (0.02, 0.05, 0.20)]

    # ---- per-flow decision ------------------------------------------------------------------
    def decide(self, p, nov, params: TriggerParams, rng) -> Decision:
        if self.explain_all:
            return Decision(True, "all", self.sync_explainer, False)
        if self.random_rate is not None:
            if rng.random() < self.random_rate:
                return Decision(True, "rand", self.sync_explainer, False)
            return Decision()
        A = self.rule_alert and p >= params.tau
        U = self.rule_uncertain and abs(p - params.tau) <= params.delta
        N = self.rule_novel and nov >= params.kappa
        R = self.rule_audit and p < params.tau and rng.random() < params.rho
        if not (A or U or N or R):
            return Decision()
        reason = "A" if A else "U" if U else "N" if N else "R"
        escalate = self.escalate_deep and ((A and p >= params.tau_high) or U)
        return Decision(True, reason, self.sync_explainer, escalate)
