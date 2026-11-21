"""Core checks. Run with:  python -m pytest -q"""

import numpy as np
import pytest

from ppc.certify import CertConfig, active_design, design_for, expected_label_cost, soft_prediction
from ppc.confseq import BettingUpperCS, UpperCS, upper_path
from ppc.inventory import NOMINAL, simulate_serial
from ppc.labeling import LabelConfig, label
from ppc.runner import SETTINGS, run_one
from ppc.problems import TabulatedProblem, synthetic_problem


def test_confidence_sequence_is_time_uniformly_valid():
    """The bound should fall below the true mean at any time in at most ~alpha of runs."""
    rng = np.random.default_rng(0)
    alpha, runs, steps = 0.1, 1000, 300
    for draw, mean in [(lambda n: rng.random(n) < 0.05, 0.05),     # rare Bernoulli
                       (lambda n: rng.beta(2, 5, n), 2 / 7)]:        # skewed continuous
        violations = 0
        for _ in range(runs):
            path = upper_path(draw(steps).astype(float), alpha)
            violations += np.any(path < mean)
        rate = violations / runs
        assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / runs), rate


def test_betting_sequence_is_time_uniformly_valid():
    rng = np.random.default_rng(10)
    alpha, runs, steps = 0.1, 400, 300
    for draw, mean in [(lambda n: (rng.random(n) < 0.05).astype(float), 0.05),
                       (lambda n: rng.beta(2, 5, n), 2 / 7)]:
        violations = 0
        for _ in range(runs):
            cs = BettingUpperCS(alpha, grid_size=300)
            violations += any(cs.update(float(v)) < mean for v in draw(steps))
        rate = violations / runs
        assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / runs), rate


def test_confidence_sequence_shrinks():
    cs = UpperCS(0.05)
    for _ in range(2000):
        cs.update(0.0)
    assert cs.upper < 0.01


def test_label_rarely_misses_true_improvers():
    """P(label 0 | mu < c - delta) must be at most gamma."""
    c, delta, gamma = 0.0, 0.1, 0.05
    prob = TabulatedProblem(np.array([[0]]), np.array([c - delta - 1e-3]), np.array([1.0]))
    rng = np.random.default_rng(1)
    zeros = sum(label(prob, 0, c, delta, gamma, 1.0, rng, LabelConfig())[0] == 0 for _ in range(3000))
    assert zeros / 3000 <= gamma + 3 * np.sqrt(gamma * (1 - gamma) / 3000)


def test_rectifier_is_unbiased_with_exact_labels():
    """P + E_pi[w (h - f)] must equal the true mass, for any surrogate and design."""
    prob = synthetic_problem("cliff", grid=40, noise_sd=0.5)
    rng = np.random.default_rng(2)
    nu = np.full(prob.n, 1 / prob.n)
    c = 0.5
    h = (prob.mu < c).astype(float)
    mu_hat = prob.mu + rng.normal(0, 0.5, prob.n)          # a noisy, biased-looking surrogate
    sd_hat = np.full(prob.n, 0.4)
    f = soft_prediction(mu_hat, sd_hat, c)
    cost = expected_label_cost(mu_hat, sd_hat, c, 0.1, 0.01, 0.5, LabelConfig())
    pi = active_design(nu, f * (1 - f), cost, lam=0.5)
    exact = np.sum(nu * f) + np.sum(pi * (nu / pi) * (h - f))
    assert abs(exact - h.mean()) < 1e-12
    draws = rng.choice(prob.n, size=200_000, p=pi)
    est = np.sum(nu * f) + np.mean((nu / pi)[draws] * (h - f)[draws])
    assert abs(est - h.mean()) < 0.01


def test_design_for_returns_distributions():
    prob = synthetic_problem("smooth", grid=30)
    nu = np.full(prob.n, 1 / prob.n)
    f = soft_prediction(prob.mu, np.full(prob.n, 0.3), 0.5)
    for m in ("uniform", "ppi", "ppi_1s", "ppc", "ppc_1s"):
        _, pi = design_for(m, nu, f, (f > 0.5).astype(float), prob.mu, np.full(prob.n, 0.3), 0.5,
                           CertConfig(), 0.5)
        assert np.all(pi > 0) and abs(pi.sum() - 1) < 1e-9
        assert np.max(nu / pi) <= 1 / CertConfig().lam + 1e-9


def test_inventory_simulator_shapes_and_interior_optimum():
    rng = np.random.default_rng(3)
    sw = np.array([0, 30, 35, 90])
    sr = np.array([0, 28, 30, 55])
    out = simulate_serial(sw, sr, 50, NOMINAL, rng)
    assert out.shape == (4, 50) and np.all(np.isfinite(out)) and np.all(out >= 0)
    means = out.mean(axis=1)
    assert means[1] < means[0] and means[1] < means[3]    # a sensible policy beats both extremes


def test_end_to_end_runs():
    st = SETTINGS["smooth"]
    rows = run_one(st.factory(), "quadratic", st.cfg, st.search_opts, seed=0)
    methods = {r["method"] for r in rows}
    assert methods == {"model_only", "uniform", "ppi", "ppi_1s", "ppc", "ppc_1s"}
    for r in rows:
        assert 0.0 <= r["bound_final"] <= 1.0


def test_every_experiment_preset_runs(tmp_path, monkeypatch):
    from ppc.experiments import EXPERIMENTS, load_results, run_experiment
    monkeypatch.setenv("CS_BANK_REPS", "20")
    monkeypatch.setenv("CS_CACHE_DIR", str(tmp_path / "cache"))
    for exp in EXPERIMENTS:
        probs = {"E5": ["inventory_shift"], "E6": ["inventory_real"], "E7": ["smooth_early"]}.get(exp, ["smooth"])
        surrs = ["gp"] if exp == "E3" else ["quadratic"]
        run_experiment(exp, n_macro=1, seed=0, out_dir=str(tmp_path), problems=probs,
                       surrogates=surrs, verbose=False)
    df = load_results(str(tmp_path))
    assert set(df.experiment) == set(EXPERIMENTS)
    assert {"b010", "b100"} <= set(df.columns)
    e1 = df[df.experiment == "E1"]
    assert "recall_search" in e1.columns and "recall_random" in e1.columns


def test_interrupted_part_resumes_with_identical_results(tmp_path):
    import pandas as pd
    from ppc.experiments import progress, run_part
    out = str(tmp_path)
    (path,) = run_part("E5", 1, out, n_macro=2, verbose=False)
    full = pd.read_csv(path)
    assert full[["surrogate", "seed"]].drop_duplicates().shape[0] == 6
    # simulate a disconnect: only the first gp run survived
    full[(full.surrogate == "gp") & (full.seed == full.seed.min())].to_csv(path, index=False)
    run_part("E5", 1, out, n_macro=2, verbose=False)
    resumed = pd.read_csv(path)
    assert len(resumed) == len(full)
    both = resumed.merge(full, on=["surrogate", "seed", "method"], suffixes=("", "_full"))
    assert np.allclose(both.bound_final, both.bound_final_full)
    assert progress(out, n_macro=2).loc["E5", "parts done"] == "1/2"


def test_bootstrap_t_verification_is_never_tighter_than_t():
    import math
    from scipy import stats
    from ppc.certify import verify_incumbent
    prob = TabulatedProblem(np.zeros((1, 2)), np.array([5.0]), np.array([2.0]))
    for seed in range(20):
        u_t, _ = verify_incumbent(prob, 0, 50, 0.025, np.random.default_rng(seed), method="t")
        u_b, _ = verify_incumbent(prob, 0, 50, 0.025, np.random.default_rng(seed), method="boot_t")
        assert u_b >= u_t - 1e-12


def test_empirical_problem_resamples_its_bank():
    from ppc.problems import EmpiricalProblem
    bank = np.array([[1.0, 2.0, 3.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
    prob = EmpiricalProblem(np.zeros((2, 2)), bank)
    assert np.allclose(prob.mu, [4.0, 5.0])
    y = prob.simulate(0, 1000, np.random.default_rng(0))
    assert set(np.unique(y)) <= {1.0, 2.0, 3.0, 10.0}
    assert np.all(prob.simulate(1, 10, np.random.default_rng(1)) == 5.0)


def test_betting_label_is_valid_for_skewed_outputs_and_rejects_negative_ones():
    from ppc.labeling import LabelConfig, label
    from ppc.problems import EmpiricalProblem
    rng = np.random.default_rng(3)
    # a strongly right-skewed and a left-skewed nonnegative output distribution, mean exactly c - delta
    right = rng.lognormal(0.0, 1.2, size=4000)
    left = 10.0 - rng.lognormal(0.0, 0.8, size=4000).clip(max=9.9)
    cfg = LabelConfig(n_min=3, growth=2.5, schedule="geometric")
    for row in (right, left):
        prob = EmpiricalProblem(np.zeros((1, 2)), row[None, :])
        mu = float(prob.mu[0])
        zeros = sum(label(prob, 0, c=mu + 0.5 + 1e-9, delta=0.5, gamma=0.05, sd_hint=float(prob.sd[0]),
                          rng=rng, cfg=cfg, test="betting")[0] == 0 for _ in range(400))
        assert zeros / 400 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)
    neg = EmpiricalProblem(np.zeros((1, 2)), np.array([[-1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        for _ in range(50):
            label(neg, 0, c=5.0, delta=0.5, gamma=0.05, sd_hint=1.0, rng=rng, cfg=cfg, test="betting")
