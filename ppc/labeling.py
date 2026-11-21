"""Sequential labeling of an audited solution: is mu(theta) below the threshold c?

The label only has to satisfy one condition for the certificate to be valid
(assumption A3 of the paper):

    P( label = 0  |  mu(theta) < c - delta )  <=  gamma.

So the rule only says "0" when it is confident that mu(theta) > c - delta, and it
defaults to "1" (conservative) whenever it runs out of replications. Solutions
with mu in [c - delta, c] are in the indifference zone, so either label is fine.

Replications are taken at a fixed schedule of looks, and gamma is split evenly
over the looks (Bonferroni). Three tests are available:

* t-test (default): uses the sample standard deviation; assumes roughly normal
  replication outputs. No other assumption.
* z-test with a variance bound ``sd_bound``: additionally assumes the noise sd
  of every true improver is at most ``sd_bound``. Much cheaper for tiny first
  looks (2-3 replications), because the t-distribution with 1-2 degrees of
  freedom has very heavy tails.
* betting (distribution-free): the label-0 decision is a betting test of
  H0: mu <= c - delta that needs only a known lower bound on the outputs (costs >= 0).
  Its capital is a nonnegative supermartingale under H0 for ANY output distribution, so
  P(label 0 | mu <= c - delta) <= gamma exactly, at every replication, with no Bonferroni
  split. The label-1 stop keeps the t rule: stopping with label 1 never hurts validity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats


@dataclass
class LabelConfig:
    n_min: int = 10            # replications at the first look
    batch: int = 10            # added per look ("linear" schedule)
    growth: float = 2.5        # multiplier per look ("geometric" schedule)
    schedule: str = "linear"   # "linear" or "geometric"
    n_cap: int = 400           # hard cap on replications per audited solution


def max_reps(delta: float, gamma: float, sd_hint: float, cfg: LabelConfig) -> int:
    """Replications after which the label defaults to 1 (standard error about delta / (2 z))."""
    z = stats.norm.ppf(1 - gamma / 20)
    n = math.ceil((2 * z * max(sd_hint, 1e-9) / delta) ** 2)
    return max(cfg.n_min, min(n, cfg.n_cap))


def look_schedule(n_max: int, cfg: LabelConfig) -> list[int]:
    """Cumulative replication counts at which the test is applied."""
    looks = [cfg.n_min]
    while looks[-1] < n_max:
        if cfg.schedule == "geometric":
            nxt = max(looks[-1] + 1, math.ceil(looks[-1] * cfg.growth))
        else:
            nxt = looks[-1] + cfg.batch
        looks.append(min(nxt, n_max))
    return looks


def betting_bets(x_prev: np.ndarray, m0: float, sd_hint: float, lam_cap: float = 0.5) -> float:
    """Predictable Kelly-type bet for the label-0 test (outputs shifted so they are >= 0).

    lambda = (mean - m0) / (var + (mean - m0)^2), with the variance shrunk toward sd_hint^2,
    clipped to [0, lam_cap / m0] so every capital factor stays >= 1 - lam_cap > 0.
    No bet before the first replication.
    """
    k = x_prev.size
    if k == 0:
        return 0.0
    mean = x_prev.mean()
    var = (((x_prev - mean) ** 2).sum() + sd_hint ** 2) / k
    edge = mean - m0
    return float(min(max(edge / (var + edge * edge), 0.0), lam_cap / m0))


def label(problem, idx: int, c: float, delta: float, gamma: float, sd_hint: float,
          rng: np.random.Generator, cfg: LabelConfig = LabelConfig(),
          sd_bound: float | None = None, n_first: int | None = None,
          test: str = "t", lower: float = 0.0) -> tuple[int, int, float]:
    """Return ``(label, replications_used, sample_mean)`` for solution ``idx``.

    ``n_first`` overrides the size of the first look for this solution. It must be
    chosen without using this solution's stream-C replications (for example from
    the surrogate), so the label stays valid.
    """
    if n_first is not None:
        cfg = replace(cfg, n_min=int(n_first))
    n_max = max(max_reps(delta, gamma, sd_bound or sd_hint, cfg), cfg.n_min)
    looks = look_schedule(n_max, cfg)
    level = gamma / len(looks)
    z = stats.norm.ppf(1 - level)
    if test == "betting":
        return _label_betting(problem, idx, c, delta, gamma, sd_hint, rng, looks, level, lower)
    y = np.empty(0)
    for n in looks:
        y = np.concatenate([y, problem.simulate(idx, n - len(y), rng)])
        m = y.mean()
        if sd_bound is None:
            if n < 2:
                continue
            half = stats.t.ppf(1 - level, n - 1) * y.std(ddof=1) / math.sqrt(n)
        else:
            half = z * sd_bound / math.sqrt(n)
        if m - half > c - delta:      # confidently not an improver (by more than delta)
            return 0, n, float(m)
        if m + half < c:              # confidently an improver
            return 1, n, float(m)
    return 1, len(y), float(y.mean())  # undecided: default to the conservative label


def guided_first_look(mu_hat, c: float, delta: float, gamma: float, sd_hint: float,
                      candidates=(3, 4, 5, 7, 10, 15, 20, 30, 40), looks_guess: int = 6) -> np.ndarray:
    """Surrogate-chosen size of the first look for every solution.

    Picks the smallest candidate n at which a t-test would already decide if the true
    mean equaled the surrogate's prediction (with 20% slack): policies the surrogate
    thinks are far from the threshold start with 3 replications, near ones with more.
    Uses only the surrogate, so labels stay valid even when the surrogate is wrong.
    """
    gap = np.maximum(np.abs(np.asarray(mu_hat) - (c - delta / 2)) - delta / 2, 1e-9)
    level = gamma / looks_guess
    half_widths = np.array([stats.t.ppf(1 - level, n - 1) * sd_hint / math.sqrt(n) for n in candidates])
    ok = half_widths[None, :] < 0.8 * gap[:, None]
    first = np.where(ok.any(axis=1), np.argmax(ok, axis=1), len(candidates) - 1)
    return np.asarray(candidates)[first]


def _label_betting(problem, idx, c, delta, gamma, sd_hint, rng, looks, level, lower):
    """Distribution-free label: label 0 only if a betting test rejects mu <= c - delta.

    Outputs must satisfy Y >= lower (lower = 0 for costs). Validity: under H0 the capital
    prod(1 + lambda_i (Y_i - lower - m0)), m0 = c - delta - lower, is a nonnegative
    supermartingale, so by Ville's inequality it ever reaches 1/gamma with probability
    at most gamma, whatever the distribution of Y.
    """
    m0 = c - delta - lower
    if m0 <= 0:          # no policy can have mean below c - delta: label 0 is always valid
        return 0, 0, float("nan")
    log_target = math.log(1 / gamma)
    y = np.empty(0)
    log_k = 0.0
    for n in looks:
        new = problem.simulate(idx, n - len(y), rng)
        if np.any(new < lower):
            raise ValueError(f"output below the declared lower bound {lower}")
        for v in new:
            lam = betting_bets(y - lower, m0, sd_hint)
            log_k += math.log1p(lam * (v - lower - m0))
            y = np.append(y, v)
            if log_k >= log_target:
                return 0, len(y), float(y.mean())
        if n >= 2:
            half = stats.t.ppf(1 - level, n - 1) * y.std(ddof=1) / math.sqrt(n)
            if y.mean() + half < c:          # label-1 stop: never hurts validity
                return 1, n, float(y.mean())
    return 1, len(y), float(y.mean())
