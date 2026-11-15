"""Uniform interface over the post-hoc explainers studied (Phase 2, Step 5).

``explain(x) -> phi`` returns a length-d attribution vector for the *attack* output of a
single flow ``x`` (raw feature space). Budget parameters reproduce the three cost levels of
the Phase 2 design (KernelSHAP coalitions, LIME perturbations, IG steps).

Output spaces (documented because fidelity metrics are computed in probability space):
  TreeSHAP-path        log-odds for XGBoost, probability for Random Forest (exact, fastest)
  TreeSHAP-interv      probability space, interventional w.r.t. a k-means background
  KernelSHAP, LIME,    probability space
  Partition (Owen)
  IG, DeepSHAP         probability space (sigmoid output of the MLP)
  Linear / EBM-native  logit space (closed form, near-zero cost)
"""
from __future__ import annotations

import warnings

import numpy as np
import shap
import torch

warnings.filterwarnings("ignore")


def _cls1(v):
    """Normalise SHAP outputs of various shapes to (d,) for the attack class."""
    if isinstance(v, list):    # older SHAP: one array per class
        v = v[-1]
    v = np.asarray(v)
    if v.ndim == 3:            # (n, d, classes)
        v = v[0, :, -1]
    elif v.ndim == 2:
        v = v[0]
    return v.astype(np.float64)


class Explainer:
    name = "base"
    budget = None

    def explain(self, x):
        raise NotImplementedError

    @property
    def label(self):
        return self.name if self.budget is None else f"{self.name}-{self.budget}"


class TreeSHAP(Explainer):
    def __init__(self, model, background, mode="path"):
        self.name = "TreeSHAP-path" if mode == "path" else "TreeSHAP-interv"
        if mode == "path":
            self.ex = shap.TreeExplainer(model.est, feature_perturbation="tree_path_dependent")
        else:
            self.ex = shap.TreeExplainer(model.est, data=background, feature_perturbation="interventional",
                                         model_output="probability")

    def explain(self, x):
        return _cls1(self.ex.shap_values(np.asarray(x, dtype=np.float32)[None], check_additivity=False))


class KernelSHAP(Explainer):
    name = "KernelSHAP"

    def __init__(self, model, background, nsamples):
        self.budget = nsamples
        self.ex = shap.KernelExplainer(model.predict_proba, background)

    def explain(self, x):
        v = self.ex.shap_values(np.asarray(x)[None], nsamples=self.budget, l1_reg=False, silent=True)
        return _cls1(v)


class LIME(Explainer):
    name = "LIME"

    def __init__(self, model, train_sample, feature_names, num_samples, seed=0):
        from lime.lime_tabular import LimeTabularExplainer
        self.budget = num_samples
        self.model = model
        self.d = train_sample.shape[1]
        self.ex = LimeTabularExplainer(np.asarray(train_sample), feature_names=list(feature_names),
                                       mode="classification", discretize_continuous=True,
                                       random_state=seed)

    def _f(self, X):
        p = self.model.predict_proba(X)
        return np.column_stack([1 - p, p])

    def explain(self, x):
        exp = self.ex.explain_instance(np.asarray(x), self._f, labels=(1,), num_features=self.d,
                                       num_samples=self.budget)
        phi = np.zeros(self.d)
        for i, w in exp.as_map()[1]:
            phi[i] = w
        return phi


class IntegratedGradients(Explainer):
    name = "IG"

    def __init__(self, model, baseline, steps):
        from captum.attr import IntegratedGradients as IG
        self.budget = steps
        self.net = model.net
        self.ig = IG(lambda t: self.net(t)[:, 0])
        self.base = torch.tensor(np.asarray(baseline, dtype=np.float32))[None]

    def explain(self, x):
        xt = torch.tensor(np.asarray(x, dtype=np.float32))[None]
        a = self.ig.attribute(xt, baselines=self.base, n_steps=self.budget, method="gausslegendre")
        return a[0].detach().numpy().astype(np.float64)


class DeepSHAP(Explainer):
    name = "DeepSHAP"

    def __init__(self, model, background):
        self.net = model.net
        self.ex = shap.DeepExplainer(self.net, torch.tensor(np.asarray(background, dtype=np.float32)))

    def explain(self, x):
        v = self.ex.shap_values(torch.tensor(np.asarray(x, dtype=np.float32))[None], check_additivity=False)
        return _cls1(v)


class LinearExact(Explainer):
    """Exact linear SHAP for logistic regression in its (scaled) input space: w_i (z_i - E[z_i])."""
    name = "LinearSHAP"

    def __init__(self, model, background):
        self.model = model
        self.w = model.est.coef_[0]
        self.zbar = model.scaler.transform(background).mean(axis=0)

    def explain(self, x):
        z = self.model.scaler.transform(np.asarray(x)[None])[0]
        return (self.w * (z - self.zbar)).astype(np.float64)


class EBMNative(Explainer):
    """Interpretable-by-design: additive term contributions read directly from the EBM."""
    name = "EBM-native"

    def __init__(self, model):
        self.est = model.est

    def explain(self, x):
        return np.asarray(self.est.eval_terms(np.asarray(x)[None])[0], dtype=np.float64)


class PartitionOwen(Explainer):
    """Owen values over a correlation-based feature hierarchy (grouped attributions, A10)."""
    name = "PartitionSHAP"

    def __init__(self, model, background, max_evals):
        self.budget = max_evals
        masker = shap.maskers.Partition(np.asarray(background), clustering="correlation")
        self.ex = shap.PartitionExplainer(model.predict_proba, masker)

    def explain(self, x):
        out = self.ex(np.asarray(x)[None], max_evals=self.budget, silent=True)
        return _cls1(out.values)


class Amortised(Explainer):
    """FastSHAP-style surrogate (baseline B6): one forward pass predicts the attribution
    vector that a reference explainer would produce."""
    name = "Amortised"

    def __init__(self, surrogate):
        self.s = surrogate

    def explain(self, x):
        return self.s.predict(np.asarray(x)[None])[0]


def background_kmeans(X_train, k=50, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(len(X_train), 20000), replace=False)
    km = shap.kmeans(np.asarray(X_train)[idx], k)
    return np.asarray(km.data, dtype=np.float32)


def build(model, method, budget, bg, train_sample, feature_names, median):
    if method == "TreeSHAP-path":
        return TreeSHAP(model, bg, "path")
    if method == "TreeSHAP-interv":
        return TreeSHAP(model, bg, "interventional")
    if method == "KernelSHAP":
        return KernelSHAP(model, bg, budget)
    if method == "LIME":
        return LIME(model, train_sample, feature_names, budget)
    if method == "IG":
        return IntegratedGradients(model, median, budget)
    if method == "DeepSHAP":
        return DeepSHAP(model, bg)
    if method == "LinearSHAP":
        return LinearExact(model, bg)
    if method == "EBM-native":
        return EBMNative(model)
    if method == "PartitionSHAP":
        return PartitionOwen(model, bg, budget)
    raise ValueError(method)


# Which (explainer, budgets) apply to which detector family in the cost study (E1).
PLAN = {
    "XGB": [("TreeSHAP-path", [None]), ("TreeSHAP-interv", [None]), ("KernelSHAP", [100, 500, 2048]),
            ("LIME", [500, 1000, 5000]), ("PartitionSHAP", [500])],
    "RF": [("TreeSHAP-path", [None]), ("TreeSHAP-interv", [None]), ("KernelSHAP", [100, 500, 2048]),
           ("LIME", [500, 1000, 5000])],
    "MLP": [("IG", [16, 32, 64]), ("DeepSHAP", [None]), ("KernelSHAP", [100, 500, 2048]),
            ("LIME", [500, 1000, 5000])],
    "LR": [("LinearSHAP", [None]), ("KernelSHAP", [500]), ("LIME", [1000])],
    "EBM": [("EBM-native", [None]), ("KernelSHAP", [500])],
}
