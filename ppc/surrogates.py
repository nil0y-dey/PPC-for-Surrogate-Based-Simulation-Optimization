"""Surrogate models with a common interface: ``fit(X, y)`` and ``predict(X) -> (mean, sd)``.

Three surrogates are used in the experiments:

* ``GPSurrogate``        - Gaussian process (the standard choice; well specified on smooth problems).
* ``ForestSurrogate``    - extra-trees ensemble; mean and spread across trees (handles jumps).
* ``QuadraticSurrogate`` - quadratic regression; deliberately misspecified on most problems.

The certificate's validity does not depend on which one is used.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class GPSurrogate:
    name = "gp"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        kernel = (ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=0.2, length_scale_bounds=(1e-2, 10.0), nu=2.5)
                  + WhiteKernel(0.1, (1e-6, 1e2)))
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1,
                                              random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, y)
        return self

    def predict(self, X):
        mean, sd = self.model.predict(X, return_std=True)
        return mean, np.maximum(sd, 1e-9)


class ForestSurrogate:
    name = "forest"

    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=2,
                                         random_state=self.seed, n_jobs=1)
        self.model.fit(X, y)
        self.resid_sd = float(np.std(y - self.model.predict(X))) + 1e-9
        return self

    def predict(self, X):
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        mean = per_tree.mean(axis=0)
        sd = np.sqrt(per_tree.var(axis=0) + self.resid_sd**2)
        return mean, sd


class QuadraticSurrogate:
    name = "quadratic"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.poly = PolynomialFeatures(degree=2)
        Z = self.poly.fit_transform(X)
        self.model = Ridge(alpha=1e-3).fit(Z, y)
        self.resid_sd = float(np.std(y - self.model.predict(Z))) + 1e-9
        return self

    def predict(self, X):
        mean = self.model.predict(self.poly.transform(X))
        return mean, np.full(len(mean), self.resid_sd)


SURROGATES = {"gp": GPSurrogate, "forest": ForestSurrogate, "quadratic": QuadraticSurrogate}


def make_surrogate(name: str, seed: int = 0):
    try:
        return SURROGATES[name](seed=seed)
    except KeyError as err:
        raise ValueError(f"unknown surrogate {name!r}; choose from {list(SURROGATES)}") from err
