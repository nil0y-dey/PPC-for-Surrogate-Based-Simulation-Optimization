"""One macro-replication of search -> verify -> audit, and a driver for many of them.

One macro-replication =
  phase A: surrogate search (stream A) -> incumbent and surrogate predictions
  phase B: verify the incumbent with fresh replications (stream B) -> U_bar
  phase C: for each method, audit until the budget is spent (its own stream C)

For every method we record whether the bound ever fell below the true
improvement mass (a violation), the final bound, the bound at several budget
checkpoints, and how many replications it took to certify the target p*.
"model_only" is the surrogate's own estimate, which has no guarantee.

With ``selection_metrics=True`` we also measure the selection effect: how many of
the policies that still beat the incumbent the search surrogate predicts as
better (its recall), compared with a surrogate fitted to a random design of the
same size.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .certify import METHODS, CertConfig, audit, design_for, soft_prediction, verify_incumbent
from .labeling import LabelConfig, guided_first_look
from .problems import inventory_bank, load_table, synthetic_problem
from .search import _fit_predict, surrogate_search

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHECKPOINTS = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Setting:
    factory: Callable                       # builds the problem the search runs on
    cfg: CertConfig                         # certificate configuration
    search_opts: dict                       # options for surrogate_search
    cert_factory: Callable | None = None    # problem certified on (regime shift); None = same


def _table(name, label):
    return lambda: load_table(os.path.join(DATA_DIR, f"{name}.npz"), label)


SETTINGS = {
    "smooth": Setting(lambda: synthetic_problem("smooth", grid=100, noise_sd=0.5),
                      CertConfig(eps=0.2, delta=0.1), {"log_y": False}),
    "cliff": Setting(lambda: synthetic_problem("cliff", grid=100, noise_sd=0.5),
                     CertConfig(eps=0.2, delta=0.1), {"log_y": False}),
    "needle": Setting(lambda: synthetic_problem("needle", grid=100, noise_sd=0.5),
                      CertConfig(eps=0.2, delta=0.1), {"log_y": False}),
    "inventory": Setting(_table("inventory_nominal", "inventory-nominal"),
                         CertConfig(eps=0.5, delta=0.2), {"log_y": True}),
    "inventory_disrupted": Setting(_table("inventory_disrupted", "inventory-disrupted"),
                                   CertConfig(eps=3.0, delta=1.0), {"log_y": True}),
    # regime shift: search and surrogate from the nominal regime, certify in the disrupted one
    "inventory_shift": Setting(_table("inventory_nominal", "inventory-nominal"),
                               CertConfig(eps=3.0, delta=1.0), {"log_y": True},
                               cert_factory=_table("inventory_disrupted", "inventory-shift")),
    # E6: the same three inventory settings, but every replication is resampled from a bank of
    # real simulator outputs (skewed, heavy-tailed) instead of normal noise around the table
    "inventory_real": Setting(lambda: inventory_bank("nominal", "inventory-nominal-real"),
                              CertConfig(eps=0.5, delta=0.2), {"log_y": True}),
    "inventory_disrupted_real": Setting(lambda: inventory_bank("disrupted", "inventory-disrupted-real"),
                                        CertConfig(eps=3.0, delta=1.0), {"log_y": True}),
    "inventory_shift_real": Setting(lambda: inventory_bank("nominal", "inventory-nominal-real"),
                                    CertConfig(eps=3.0, delta=1.0), {"log_y": True},
                                    cert_factory=lambda: inventory_bank("disrupted", "inventory-shift-real")),
}
# E7: the search stops after its initial design (20 policies, no surrogate-guided rounds), so the
# incumbent is mediocre and a sizable improvement mass remains above the resolution
for _name in ("smooth", "cliff", "inventory"):
    SETTINGS[_name + "_early"] = replace(SETTINGS[_name],
                                         search_opts={**SETTINGS[_name].search_opts, "n_rounds": 0})

# Certificate configurations (overrides of CertConfig)
CONFIGS = {
    "baseline": {},
    "betting": {"cs": "betting"},
    "betting+geo5": {"cs": "betting", "label": LabelConfig(n_min=5, schedule="geometric")},
    "betting+geo3": {"cs": "betting", "label": LabelConfig(n_min=3, schedule="geometric")},
    "betting+guided": {"cs": "betting", "guided_labels": True,
                       "label": LabelConfig(n_min=5, schedule="geometric")},
}
CONFIGS["final"] = CONFIGS["betting+geo3"]          # the recommended procedure
CONFIGS["final+guided"] = CONFIGS["betting+guided"]


def _record(base, method, out, m_true, cfg, t1):
    hit = np.nonzero(out["bound"] <= cfg.target)[0]
    row = {**base, "method": method, "bound_final": float(out["bound"][-1]),
           "violated": bool(np.any(out["bound"] < m_true)),
           "reps_to_target": float(out["reps"][hit[0]]) if len(hit) else np.nan,
           "n_audits": out["n_audits"], "P": out["P"], "w_max": out["w_max"],
           "seconds": time.time() - t1}
    for frac in CHECKPOINTS:
        k = np.searchsorted(out["reps"], frac * cfg.budget, side="right") - 1
        row[f"b{int(round(frac * 100)):03d}"] = float(out["bound"][k]) if k >= 0 else 1.0
    if method == "uniform" and out["n_audits"] > 0:
        # Clopper-Pearson upper bound for the final audit count, as if it had been fixed in advance:
        # not anytime-valid; it shows what anytime validity costs
        n, k1 = int(out["n_audits"]), int(out["n_ones"])
        cp = 1.0 if k1 >= n else float(stats.beta.ppf(1 - cfg.alpha, k1 + 1, n - k1))
        row["bound_cp_fixed_n"] = min(1.0, cp / (1 - cfg.gamma))
        row["n_ones"] = k1
    return row


def _recall(mu_hat, c, target_set):
    if not target_set.any():
        return np.nan
    return float(np.mean(mu_hat[target_set] < c))


def run_one(problem, surrogate: str, cfg: CertConfig, search_opts: dict, seed: int,
            methods=METHODS, keep_paths: bool = False, cert_problem=None,
            selection_metrics: bool = False):
    cert = problem if cert_problem is None else cert_problem
    ss = np.random.SeedSequence(seed)
    streams = [np.random.default_rng(s) for s in ss.spawn(4 + 2 * len(methods))]
    rng_a, rng_b, rng_r = streams[0], streams[1], streams[2]
    rng_c = streams[3:]

    t0 = time.time()
    res = surrogate_search(problem, surrogate, rng_a, seed=seed, **search_opts)
    inc = res.incumbent
    u_bar, sd_ucb = verify_incumbent(cert, inc, cfg.n_verify, cfg.beta, rng_b, cfg.verify)
    sd_bound = None if cfg.sd_bound_factor is None else cfg.sd_bound_factor * sd_ucb
    c = u_bar - cfg.eps

    nu = np.full(cert.n, 1.0 / cert.n)
    f_soft = soft_prediction(res.mu_hat, res.sd_hat, c)
    f_hard = (res.mu_hat < c).astype(float)
    n_first_map = (guided_first_look(res.mu_hat, c, cfg.delta, cfg.gamma, res.noise_sd_hint)
                   if cfg.guided_labels else None)

    mu_inc = cert.mu[inc]
    m_true = cert.improvement_mass(mu_inc - cfg.eps - cfg.delta, nu)   # guaranteed target
    m_eps = cert.improvement_mass(mu_inc - cfg.eps, nu)                # for reference
    base = {"problem": cert.name, "surrogate": surrogate, "seed": seed,
            "gap": float(mu_inc - cert.mu.min()), "search_reps": res.reps_used,
            "u_bar_covers": bool(u_bar >= mu_inc), "M_true": m_true, "M_eps": m_eps}

    if selection_metrics:
        target_set = cert.mu < mu_inc - cfg.eps - cfg.delta
        base["recall_search"] = _recall(res.mu_hat, c, target_set)
        # same-size surrogate from a uniformly random design (no search, so no selection)
        X = problem.features()
        n_eval = len(res.eval_idx)
        reps = res.reps_used // n_eval
        idx_r = rng_r.choice(problem.n, size=n_eval, replace=False)
        y_r = np.array([problem.simulate(int(i), reps, rng_r).mean() for i in idx_r])
        mu_r, _ = _fit_predict(surrogate, seed, X[idx_r], y_r, X, search_opts.get("log_y", False))
        base["recall_random"] = _recall(mu_r, c, target_set)
        # quantities in Theorem 3 / Corollary 2 (computed from the truth; no extra randomness)
        evaluated = np.zeros(cert.n, dtype=bool)
        evaluated[res.eval_idx] = True
        base["flag_mass"] = float(np.sum(nu * f_hard))                                 # P with hard flags
        base["false_alarm_mass"] = float(np.sum(nu * f_hard * (cert.mu >= c)))          # flagged, not below c
        base["flags_unevaluated"] = float(np.sum(nu * f_hard * ~evaluated))            # 0 = exhaustion
        base["c_below_best_mean"] = bool(c < res.ybar.min())
        missed = target_set & (f_hard == 0)
        if missed.any():
            _, pi_act = design_for("ppc_1s", nu, f_soft, f_hard, res.mu_hat, res.sd_hat, c, cfg,
                                   res.noise_sd_hint)
            base["lift_missed"] = float(pi_act[missed].sum() / nu[missed].sum())          # Lambda
        else:
            base["lift_missed"] = np.nan

    rows, paths = [], {}
    p_model = float(np.sum(nu * f_soft))
    rows.append({**base, "method": "model_only", "bound_final": p_model,
                 "violated": bool(p_model < m_true), "reps_to_target": np.nan,
                 "n_audits": 0, "P": p_model, "seconds": time.time() - t0})
    for k, m in enumerate(methods):
        t1 = time.time()
        if m == "ppi_1s_refit":
            # stage 1: uniform audits; their labels are not used in the certificate
            b1 = int(cfg.refit_share * cfg.budget)
            out1 = audit(cert, "uniform", nu, np.zeros(cert.n), nu, c, cfg, res.noise_sd_hint,
                         rng_c[2 * k], budget=b1, sd_bound=sd_bound, n_first_map=n_first_map)
            reps1 = int(out1["reps"][-1])
            idx_a = np.array([a[0] for a in out1["audited"]])
            mean_a = np.array([a[1] for a in out1["audited"]])
            X = problem.features()
            idx_all = np.concatenate([res.eval_idx, idx_a])
            y_all = np.concatenate([res.ybar, mean_a])
            mu2, _ = _fit_predict(surrogate, seed, X[idx_all], y_all, X, search_opts.get("log_y", False))
            f2 = (mu2 < c).astype(float)
            # stage 2: ppi_1s with a fresh stream and the remaining budget
            out = audit(cert, "ppi_1s", nu, f2, nu, c, cfg, res.noise_sd_hint, rng_c[2 * k + 1],
                        budget=cfg.budget - reps1, sd_bound=sd_bound, n_first_map=n_first_map)
            out["reps"] = out["reps"] + reps1
        else:
            f, pi = design_for(m, nu, f_soft, f_hard, res.mu_hat, res.sd_hat, c, cfg, res.noise_sd_hint)
            out = audit(cert, m, nu, f, pi, c, cfg, res.noise_sd_hint, rng_c[2 * k], sd_bound=sd_bound,
                        n_first_map=n_first_map)
        rows.append(_record(base, m, out, m_true, cfg, t1))
        if keep_paths:
            paths[m] = out
    return (rows, paths) if keep_paths else rows


def run_settings(problems=("smooth", "cliff", "inventory"), surrogates=("gp", "forest", "quadratic"),
              n_macro: int = 50, seed: int = 0, budget: int | None = None, out_csv: str | None = None,
              verbose: bool = True, methods=METHODS, config: dict | None = None,
              config_name: str = "baseline", selection_metrics: bool = False,
              resume: bool = False, checkpoint_secs: float = 60.0) -> pd.DataFrame:
    """Run ``n_macro`` macro-replications for every (problem, surrogate) pair.

    With ``resume=True`` and an existing ``out_csv``, runs already in the file are kept and
    skipped, so an interrupted session continues where it stopped. The file is rewritten at
    least every ``checkpoint_secs`` seconds and after every (problem, surrogate) pair; it only
    ever holds complete runs.
    """
    rows, done = [], set()
    if resume and out_csv is not None and os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        if not old.empty:
            rows = old.to_dict("records")
            done = set(zip(old["setting"], old["surrogate"], old["seed"].astype(int)))
    last_save = time.time()

    def save():
        if out_csv is not None and rows:
            tmp = out_csv + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out_csv)       # never leaves a half-written file behind

    for pname in problems:
        st = SETTINGS[pname]
        cfg = st.cfg
        if budget is not None:
            cfg = replace(cfg, budget=budget)
        if config:
            cfg = replace(cfg, **config)
        problem = st.factory()
        cert_problem = st.cert_factory() if st.cert_factory else None
        for sname in surrogates:
            t0 = time.time()
            n_new = 0
            for r in range(n_macro):
                run_seed = seed * 100003 + r
                if (pname, sname, run_seed) in done:
                    continue
                new = run_one(problem, sname, cfg, st.search_opts, seed=run_seed, methods=methods,
                              cert_problem=cert_problem, selection_metrics=selection_metrics)
                for row in new:
                    row["config"] = config_name
                    row["setting"] = pname
                rows.extend(new)
                n_new += 1
                if time.time() - last_save > checkpoint_secs:
                    save()
                    last_save = time.time()
            if n_new:
                save()
                last_save = time.time()
            if verbose:
                note = f"{n_new} runs in {time.time() - t0:7.1f}s"
                if n_new < n_macro:
                    note += f" ({n_macro - n_new} already saved)"
                print(f"{config_name:>14s} | {pname:>15s} | {sname:>9s} | {note}", flush=True)
    df = pd.DataFrame(rows)
    save()
    return df


def summarize(df: pd.DataFrame, alpha_plus_beta: float = 0.075) -> pd.DataFrame:
    """One row per (config, problem, surrogate, method)."""
    keys = [k for k in ["config", "problem", "surrogate", "method"] if k in df.columns]
    g = df.groupby(keys, sort=False)
    out = g.agg(runs=("violated", "size"),
                violation_rate=("violated", "mean"),
                M_true_mean=("M_true", "mean"),
                bound_final_mean=("bound_final", "mean"),
                certified_share=("reps_to_target", lambda s: float(np.mean(~np.isnan(s)))),
                reps_to_target_median=("reps_to_target", "median"),
                audits_mean=("n_audits", "mean"))
    out["valid?"] = np.where(out["violation_rate"] <= alpha_plus_beta + 2 * np.sqrt(
        alpha_plus_beta * (1 - alpha_plus_beta) / out["runs"]), "ok", "CHECK")
    return out.reset_index()


def default_config_dict():
    return {k: asdict(v.cfg) for k, v in SETTINGS.items()}
