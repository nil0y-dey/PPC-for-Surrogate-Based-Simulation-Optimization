"""Anytime-valid upper confidence sequences for the mean of [0, 1]-valued data.

Two one-sided (upper) constructions from Waudby-Smith & Ramdas, "Estimating means of
bounded random variables by betting" (JRSS-B, 2024):

* ``UpperCS``: the predictable-plug-in empirical-Bernstein (PrPl-EB) sequence;
* ``BettingUpperCS``: the betting (capital-process) sequence on a grid, used by the
  recommended procedure (Lemma 3 of the paper).

Guarantee: if X_1, X_2, ... lie in [0, 1] and E[X_t | past] = m for all t, then

    P( exists t : m > upper_t ) <= alpha.

Because the bound holds at every t simultaneously, it stays valid when you stop
at a data-dependent time (for example, as soon as the bound drops below a target).
"""

from __future__ import annotations

import math

import numpy as np


class UpperCS:
    """Streaming one-sided PrPl-EB upper confidence sequence."""

    def __init__(self, alpha: float, lam_max: float = 0.5):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < lam_max < 1:
            raise ValueError("lam_max must be in (0, 1)")
        self.log_inv_alpha = math.log(1.0 / alpha)
        self.lam_max = lam_max
        self.t = 0
        self.sum_lam = 0.0
        self.sum_lam_x = 0.0
        self.sum_v_psi = 0.0
        self.sum_x = 0.0
        self.sum_sq = 0.0
        self.mu_hat = 0.5      # predictable mean estimate, starts at 1/2
        self.var_hat = 0.25    # predictable variance estimate, starts at 1/4
        self.upper = 1.0       # running minimum of the bound (intersection)

    def update(self, x: float) -> float:
        if not (-1e-12 <= x <= 1 + 1e-12):
            raise ValueError(f"observation {x} outside [0, 1]")
        x = min(max(x, 0.0), 1.0)
        t = self.t + 1
        lam = min(math.sqrt(2.0 * self.log_inv_alpha / (self.var_hat * t * math.log(1.0 + t))), self.lam_max)
        v = 4.0 * (x - self.mu_hat) ** 2
        psi = (-math.log(1.0 - lam) - lam) / 4.0
        self.sum_lam += lam
        self.sum_lam_x += lam * x
        self.sum_v_psi += v * psi
        # predictable estimates for the next step
        self.sum_x += x
        self.mu_hat = (0.5 + self.sum_x) / (t + 1)
        self.sum_sq += (x - self.mu_hat) ** 2
        self.var_hat = (0.25 + self.sum_sq) / (t + 1)
        self.t = t
        bound = (self.sum_lam_x + self.log_inv_alpha + self.sum_v_psi) / self.sum_lam
        self.upper = min(self.upper, bound, 1.0)
        return self.upper


def upper_path(x: np.ndarray, alpha: float, lam_max: float = 0.5) -> np.ndarray:
    """Upper bounds after each observation of the sequence ``x``."""
    cs = UpperCS(alpha, lam_max)
    return np.array([cs.update(float(v)) for v in x])


class BettingUpperCS:
    """Streaming one-sided betting (capital-process) upper confidence sequence.

    For each candidate mean m on a grid, the capital K_t(m) = prod_i (1 + lam_i (m - X_i))
    is a nonnegative martingale when the true mean is m (lam_i in [0, lam_max], lam_max < 1,
    chosen from past data only). By Ville's inequality, P(sup_t K_t(m_true) >= 1/alpha) <= alpha.
    Every grid value whose capital ever reaches 1/alpha is excluded for good. Because K_t(m) is
    increasing in m, the excluded set is an interval [m*, 1], and the smallest excluded grid
    value is a valid upper bound.

    Much tighter than empirical Bernstein when the data are mostly zeros (rare events).
    """

    def __init__(self, alpha: float, lam_max: float = 0.8, grid_size: int = 600):
        if not 0 < lam_max < 1:
            raise ValueError("lam_max must be in (0, 1)")
        self.log_inv_alpha = math.log(1.0 / alpha)
        self.lam_max = lam_max
        lo = np.logspace(-5, 0, grid_size // 2)
        lin = np.linspace(0, 1, grid_size // 2 + 1)[1:]
        self.grid = np.unique(np.concatenate([lo, lin]))
        self.log_k = np.zeros_like(self.grid)
        self.max_log_k = np.zeros_like(self.grid)
        self.t = 0
        self.sum_x = 0.0
        self.sum_sq = 0.0
        self.mu_hat = 0.5
        self.var_hat = 0.25
        self.upper = 1.0

    def update(self, x: float) -> float:
        if not (-1e-12 <= x <= 1 + 1e-12):
            raise ValueError(f"observation {x} outside [0, 1]")
        x = min(max(x, 0.0), 1.0)
        t = self.t + 1
        lam = min(math.sqrt(2.0 * self.log_inv_alpha / (self.var_hat * t * math.log(1.0 + t))), self.lam_max)
        self.log_k += np.log1p(lam * (self.grid - x))
        np.maximum(self.max_log_k, self.log_k, out=self.max_log_k)
        self.sum_x += x
        self.mu_hat = (0.5 + self.sum_x) / (t + 1)
        self.sum_sq += (x - self.mu_hat) ** 2
        self.var_hat = (0.25 + self.sum_sq) / (t + 1)
        self.t = t
        excluded = np.nonzero(self.max_log_k >= self.log_inv_alpha)[0]
        if len(excluded):
            self.upper = min(self.upper, float(self.grid[excluded[0]]))
        return self.upper


def make_cs(kind: str, alpha: float):
    if kind == "eb":
        return UpperCS(alpha)
    if kind == "betting":
        return BettingUpperCS(alpha)
    raise ValueError(f"unknown confidence sequence {kind!r}")
