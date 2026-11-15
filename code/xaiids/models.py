"""Detectors (Phase 2, Steps 5-6).

All detectors expose ``predict_proba(X_raw) -> P(attack)`` on the *raw* feature space.
Scaling (signed log1p + standardisation) lives inside the LR/MLP wrappers, so every
explainer attributes over the same raw features and fidelity metrics perturb the same space.

Thresholds are chosen on validation-score quantiles (FPR <= 1%), which is invariant to any
monotone calibration; isotonic calibration from the Phase 2 plan is therefore unnecessary.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import MODELS, SEED


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Scaler:
    def fit(self, X):
        Z = signed_log1p(X.astype(np.float64))
        self.mu = Z.mean(axis=0)
        self.sd = Z.std(axis=0) + 1e-6
        return self

    def transform(self, X):
        return ((signed_log1p(np.asarray(X, dtype=np.float64)) - self.mu) / self.sd).astype(np.float32)


# ----------------------------------------------------------------------------- wrappers

class TreeModel:
    kind = "tree"

    def __init__(self, name, est):
        self.name, self.est = name, est

    def fit(self, X, y, w=None):
        self.est.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]


class LinearModel:
    kind = "linear"
    name = "LR"

    def fit(self, X, y, w=None):
        self.scaler = Scaler().fit(X)
        self.est = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        self.est.fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(self.scaler.transform(X))[:, 1]


class TorchMLP(torch.nn.Module):
    """Preprocessing is part of the network so gradient explainers act on raw features."""

    def __init__(self, d, mu, sd, hidden=(64, 32)):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        layers, prev = [], d
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers += [torch.nn.Linear(prev, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward_logit(self, x):
        z = (torch.sign(x) * torch.log1p(torch.abs(x)) - self.mu) / self.sd
        return self.net(z)

    def forward(self, x):                       # probability of attack, shape (n, 1)
        return torch.sigmoid(self.forward_logit(x))


class MLPModel:
    kind = "mlp"
    name = "MLP"

    def fit(self, X, y, w=None, epochs=12, batch=1024, seed=SEED):
        torch.manual_seed(seed)
        sc = Scaler().fit(X)
        self.net = TorchMLP(X.shape[1], sc.mu, sc.sd)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.float32))[:, None]
        pos = float(yt.mean())
        pw = torch.tensor([(1 - pos) / max(pos, 1e-6)])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(Xt)
        g = torch.Generator().manual_seed(seed)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = lossf(self.net.forward_logit(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        return self.net(torch.tensor(np.asarray(X, dtype=np.float32)))[:, 0].numpy()


class EBMModel:
    kind = "ebm"
    name = "EBM"

    def fit(self, X, y, w=None, feature_names=None):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.est = ExplainableBoostingClassifier(interactions=0, outer_bags=4, max_rounds=2000,
                                                 random_state=SEED, n_jobs=-1,
                                                 feature_names=feature_names)
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X))[:, 1]


def make(name):
    if name == "XGB":
        return TreeModel("XGB", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                              subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                              eval_metric="logloss", n_jobs=-1, random_state=SEED))
    if name == "RF":
        return TreeModel("RF", RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                                      class_weight="balanced_subsample", n_jobs=-1,
                                                      random_state=SEED))
    if name == "LR":
        return LinearModel()
    if name == "MLP":
        return MLPModel()
    if name == "EBM":
        return EBMModel()
    raise ValueError(name)


def class_weights(y):
    y = np.asarray(y)
    pos = y.mean()
    return np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))


def save(model, dataset, tier):
    joblib.dump(model, MODELS / f"{dataset}_{tier}_{model.name}.joblib")


def load(dataset, tier, name):
    return joblib.load(MODELS / f"{dataset}_{tier}_{name}.joblib")


def time_batch1(model, X, n=500, warmup=50):
    """Per-flow detection latency with batch size 1 (seconds)."""
    X = np.asarray(X, dtype=np.float32)
    idx = np.arange(min(n + warmup, len(X)))
    times = []
    for j, i in enumerate(idx):
        t0 = time.perf_counter()
        model.predict_proba(X[i:i + 1])
        dt = time.perf_counter() - t0
        if j >= warmup:
            times.append(dt)
    return np.array(times)
