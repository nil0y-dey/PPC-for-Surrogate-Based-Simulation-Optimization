"""Phase A: surrogate-based search (uses its own random-number stream).

A simple batch lower-confidence-bound (LCB) search over the finite solution set:
start from a random design, then repeatedly fit the surrogate and simulate the
solutions with the lowest ``mean - kappa * sd``. The certificate does not care
how the incumbent was found; any optimizer could replace this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surrogates import make_surrogate


@dataclass
class SearchResult:
    eval_idx: np.ndarray       # solutions simulated during search
    ybar: np.ndarray           # their sample means
    sample_sd: np.ndarray      # their sample standard deviations
    mu_hat: np.ndarray         # surrogate mean for every solution
    sd_hat: np.ndarray         # surrogate sd for every solution
    incumbent: int             # index of the recommended solution
    reps_used: int             # replications spent in the search
    noise_sd_hint: float       # pooled estimate of the per-replication sd


def _fit_predict(name, seed, X_train, y_train, X_all, log_y):
    model = make_surrogate(name, seed=seed)
    if log_y:
        model.fit(X_train, np.log(np.maximum(y_train, 1e-9)))
        m, s = model.predict(X_all)
        mean = np.exp(m + 0.5 * s**2)          # lognormal mean
        sd = mean * np.sqrt(np.expm1(s**2))     # lognormal sd
        return mean, np.maximum(sd, 1e-9)
    model.fit(X_train, y_train)
    return model.predict(X_all)


def surrogate_search(problem, surrogate: str, rng: np.random.Generator, n_init: int = 20,
                     batch: int = 10, n_rounds: int = 6, reps_per_eval: int = 10,
                     kappa: float = 2.0, log_y: bool = False, seed: int = 0) -> SearchResult:
    X_all = problem.features()
    N = problem.n
    order = []
    outputs = {}

    def evaluate(indices):
        for i in indices:
            i = int(i)
            outputs[i] = problem.simulate(i, reps_per_eval, rng)
            order.append(i)

    evaluate(rng.choice(N, size=n_init, replace=False))
    for _ in range(n_rounds):
        idx = np.array(order)
        ybar = np.array([outputs[i].mean() for i in idx])
        mean, sd = _fit_predict(surrogate, seed, X_all[idx], ybar, X_all, log_y)
        score = mean - kappa * sd
        score[idx] = np.inf
        evaluate(np.argsort(score)[:batch])

    idx = np.array(order)
    ybar = np.array([outputs[i].mean() for i in idx])
    sample_sd = np.array([outputs[i].std(ddof=1) for i in idx])
    mu_hat, sd_hat = _fit_predict(surrogate, seed, X_all[idx], ybar, X_all, log_y)
    incumbent = int(idx[np.argmin(ybar)])
    return SearchResult(eval_idx=idx, ybar=ybar, sample_sd=sample_sd, mu_hat=mu_hat, sd_hat=sd_hat,
                        incumbent=incumbent, reps_used=len(idx) * reps_per_eval,
                        noise_sd_hint=float(np.median(sample_sd)))
