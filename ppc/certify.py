"""Phases B and C: verify the incumbent, then audit to certify the improvement mass.

Target: M = nu{ theta : mu(theta) < mu(incumbent) - eps - delta }, the share of the
solution space (under the reference measure nu) that beats the incumbent by
more than eps + delta. Every method below returns an upper bound Mbar with

    P( M > Mbar_k for some k )  <=  alpha + beta,

whatever the surrogate (Theorem 1 of the paper).

Methods
-------
uniform       no predictions; audit nu-random solutions and bound the share labeled 1.
ppi           prediction term + two-sided rectifier w*(L - f), audits drawn from nu.
ppi_1s        prediction term + one-sided rectifier w*(L - f)^+ with a hard prediction
              f = 1{mu_hat < c}; only checks the surrogate's misses, never its false alarms.
ppc           like ppi, but audits drawn from a cost-aware active design.
ppc_1s        like ppi_1s, with the cost-aware active design.
ppi_1s_refit  (in runner.py) spend part of the budget on uniform audits, refit the surrogate
              on everything simulated so far, then run ppi_1s with fresh audits.
model_only    the surrogate's own estimate sum_theta nu*f (no audits, no guarantee).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .confseq import make_cs
from .labeling import LabelConfig, label, max_reps

METHODS = ("uniform", "ppi", "ppi_1s", "ppc", "ppc_1s")


@dataclass
class CertConfig:
    eps: float = 0.5           # improvement margin that matters to the decision maker
    delta: float = 0.1         # indifference zone of the labels
    alpha: float = 0.05        # error of the confidence sequence
    beta: float = 0.025        # error of the incumbent's upper bound
    gamma: float = 0.01        # label error: P(label 0 | true improver)
    n_verify: int = 200        # replications to verify the incumbent (phase B)
    verify: str = "boot_t"     # incumbent upper bound: "boot_t" (robust to skew) or "t"
    budget: int = 6000         # audit budget in replications (phase C)
    target: float = 0.02       # stop-and-certify target p*
    lam: float = 0.5           # defensive share of nu in the active design (weights <= 1/lam)
    cs: str = "eb"             # confidence sequence: "eb" (empirical Bernstein) or "betting"
    sd_bound_factor: float | None = None   # None: t-test labels; else z-test, bound = factor * sd UCB
    refit_share: float = 0.3   # share of the budget spent on uniform audits before refitting
    guided_labels: bool = False  # let the surrogate choose each audit's first look
    label: LabelConfig = field(default_factory=LabelConfig)


def verify_incumbent(problem, idx: int, n: int, beta: float, rng: np.random.Generator,
                     method: str = "boot_t", n_boot: int = 2000):
    """Fresh replications of the incumbent.

    Returns a one-sided (1 - beta) upper confidence bound on mu(idx) and a 99.9% upper
    confidence bound on its noise sd (used only by z-test labels).

    method="t": the one-sided t bound, exact for normal outputs. Simulation costs are often
    right-skewed (stockouts, disruptions), and then the t bound undercovers.
    method="boot_t" (default): the larger of the t bound and the bootstrap-t bound. The
    bootstrap-t bound is second-order accurate for skewed outputs (Hall 1992), and taking
    the maximum keeps the exact normal-theory guarantee.
    """
    y = problem.simulate(idx, n, rng)
    m, s = float(y.mean()), float(y.std(ddof=1))
    u_bar = m + stats.t.ppf(1 - beta, n - 1) * s / math.sqrt(n)
    if method == "boot_t":
        yb = y[rng.integers(0, n, size=(n_boot, n))]
        sb = np.maximum(yb.std(axis=1, ddof=1), 1e-12)
        t_star = (yb.mean(axis=1) - m) / (sb / math.sqrt(n))
        u_bar = max(u_bar, m - float(np.quantile(t_star, beta)) * s / math.sqrt(n))
    elif method != "t":
        raise ValueError(f"unknown verification method {method!r}")
    sd_ucb = float(s * math.sqrt((n - 1) / stats.chi2.ppf(0.001, n - 1)))
    return float(u_bar), sd_ucb


def soft_prediction(mu_hat, sd_hat, c):
    """Surrogate's probability that each solution is below c."""
    return stats.norm.cdf((c - mu_hat) / np.maximum(sd_hat, 1e-9))


def expected_label_cost(mu_hat, sd_hat, c, delta, gamma, sd_hint, cfg: LabelConfig):
    """Plug-in estimate of replications needed to label each solution."""
    n_cap = max_reps(delta, gamma, sd_hint, cfg)
    z = stats.norm.ppf(1 - gamma / 10)
    gap = np.maximum(np.abs(mu_hat - (c - delta / 2)), delta / 2)
    n = (z * sd_hint / gap) ** 2
    return np.clip(n, cfg.n_min, n_cap)


def active_design(nu, var_proxy, cost, lam):
    """pi = lam * nu + (1 - lam) * pi_act with pi_act proportional to nu * sqrt(v / kappa)."""
    score = nu * np.sqrt(np.maximum(var_proxy, 1e-6) / cost)
    pi_act = score / score.sum()
    return lam * nu + (1 - lam) * pi_act


def audit(problem, method: str, nu, f, pi, c: float, cfg: CertConfig, sd_hint: float,
          rng: np.random.Generator, budget: int | None = None, sd_bound: float | None = None,
          n_first_map: np.ndarray | None = None):
    """Run phase C for one method. Returns the path of (replications used, Mbar)."""
    budget = cfg.budget if budget is None else budget
    one_sided = method.endswith("_1s")
    w = nu / pi
    w_max = float(w.max())
    P = 0.0 if method == "uniform" else float(np.sum(nu * f))
    cs = make_cs(cfg.cs, cfg.alpha)
    cdf = np.cumsum(pi)
    cdf /= cdf[-1]
    reps, path_reps, path_bound, audited = 0, [], [], []
    n_ones = 0
    while reps < budget:
        i = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(pi) - 1)
        n_first = None if n_first_map is None else int(n_first_map[i])
        L, used, mean = label(problem, i, c, cfg.delta, cfg.gamma, sd_hint, rng, cfg.label, sd_bound, n_first)
        reps += used
        n_ones += L
        audited.append((i, mean, used))
        if method == "uniform":
            upper_mean = cs.update(float(L))
        elif one_sided:
            upper_mean = w_max * cs.update(w[i] * max(L - f[i], 0.0) / w_max)
        else:
            upper_mean = 2 * w_max * cs.update((w[i] * (L - f[i]) + w_max) / (2 * w_max)) - w_max
        path_reps.append(reps)
        path_bound.append(min(1.0, max(0.0, (P + upper_mean) / (1 - cfg.gamma))))
    return {"reps": np.array(path_reps), "bound": np.minimum.accumulate(np.array(path_bound)),
            "n_audits": len(audited), "n_ones": n_ones, "P": P, "w_max": w_max, "audited": audited}


def design_for(method, nu, f_soft, f_hard, mu_hat, sd_hat, c, cfg: CertConfig, sd_hint):
    """Return (f, pi) used by a method."""
    if method == "uniform":
        return np.zeros_like(nu), nu
    if method in ("ppi", "ppc"):
        f = f_soft
        var_proxy = f_soft * (1 - f_soft)
    else:  # one-sided variants
        f = f_hard
        var_proxy = f_soft * (1 - f_hard)    # chance of a miss the rectifier must catch
    if method.startswith("ppi"):
        return f, nu
    cost = expected_label_cost(mu_hat, sd_hat, c, cfg.delta, cfg.gamma, sd_hint, cfg.label)
    return f, active_design(nu, var_proxy, cost, cfg.lam)
