# hl_model.py -- the model class that gets PICKLED. Must ship with the package.
"""Unpickling needs this module importable from the working directory, because
the pickle stores the class path `hl_model.Logit`. Do not rename this file."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Logit:
    """L2 logistic regression, standardised features (same maths as training)."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.mu = None
        self.sd = None
        self.w = None

    def fit(self, X, y):
        from scipy.optimize import minimize   # only training needs scipy
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]

        def nll(w):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            eps = 1e-9
            ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
            return ll + self.lam * (w[1:] ** 2).sum() / len(X)

        self.w = minimize(nll, np.zeros(Z.shape[1]), method="L-BFGS-B").x
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (np.asarray(X, dtype=float) - self.mu) / self.sd]
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))


def save(model: Logit, path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f, protocol=4)
    return p


def load(path: str) -> Logit:
    with open(path, "rb") as f:
        return pickle.load(f)
