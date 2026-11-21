"""Label validity with the simulator's real outputs (assumption A3, policy by policy).

Theorem 1 needs P(label = 0 | mu(theta) < c - delta) <= gamma for EVERY policy. Lemma 1
proves it for normal outputs; real inventory costs are skewed and heavy-tailed. This script
measures the label error directly against the banks of real simulator outputs used in E6.

For each sampled policy it places the threshold where the label is hardest to get right:
c - delta just above mu(theta), so the policy barely counts as an improver. It then replays
the exact label procedure of ppc/labeling.py (t-test, Bonferroni over looks, geometric
schedule of the recommended configuration) many times, resampling replications from the
policy's bank row, and records how often the label is 0. The same is done with normal
outputs of the same mean and sd, where Lemma 1 guarantees <= gamma.

Usage (from the repository root):
    python scripts/check_label_validity.py                 # both regimes
    python scripts/check_label_validity.py --trials 2000   # quicker

Writes results/label_validity.csv and paper/tables/tab_labelvalidity.tex.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppc.labeling import LabelConfig, look_schedule, max_reps  # noqa: E402
from ppc.runner import CONFIGS  # noqa: E402
from ppc.problems import inventory_bank  # noqa: E402


def label_zero_rate(sample, mu, delta, gamma, sd_hint, cfg, trials, rng):
    """Monte Carlo P(label = 0) with c - delta = mu (worst case for a true improver).

    ``sample(rows, n)`` returns an array (rows, n) of fresh replications.
    Mirrors ppc.labeling.label exactly (t-test branch).
    """
    c = mu + delta + 1e-12                      # mu < c - delta, by the smallest margin
    n_max = max(max_reps(delta, gamma, sd_hint, cfg), cfg.n_min)
    looks = look_schedule(n_max, cfg)
    level = gamma / len(looks)
    s1 = np.zeros(trials)
    s2 = np.zeros(trials)
    have = 0
    alive = np.ones(trials, dtype=bool)
    zeros = 0
    for n in looks:
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        y = sample(idx.size, n - have)
        s1[idx] += y.sum(axis=1)
        s2[idx] += (y * y).sum(axis=1)
        have = n
        if n < 2:
            continue
        m = s1[idx] / n
        var = np.maximum((s2[idx] - n * m * m) / (n - 1), 0.0)
        half = stats.t.ppf(1 - level, n - 1) * np.sqrt(var) / math.sqrt(n)
        dec0 = m - half > c - delta
        dec1 = m + half < c
        zeros += int(dec0.sum())
        alive[idx[dec0 | dec1]] = False
    return zeros / trials, len(looks), looks[-1]


def label_betting(sample, mu, c, delta, gamma, sd_hint, cfg, trials, lower=0.0, lam_cap=0.5):
    """Vectorized replica of ppc.labeling._label_betting: returns (P(label 0), mean reps)."""
    m0 = c - delta - lower
    n_max = max(max_reps(delta, gamma, sd_hint, cfg), cfg.n_min)
    looks = look_schedule(n_max, cfg)
    level = gamma / len(looks)
    target = math.log(1 / gamma)
    s1 = np.zeros(trials); s2 = np.zeros(trials); logk = np.zeros(trials)
    alive = np.ones(trials, dtype=bool); label0 = np.zeros(trials, dtype=bool)
    used = np.zeros(trials); have = 0
    for n in looks:
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        y = sample(idx.size, n - have) - lower
        for j in range(y.shape[1]):
            k = have + j
            sub = alive[idx]
            if not sub.any():
                break
            ii = idx[sub]
            if k == 0:
                lam = np.zeros(ii.size)
            else:
                mean = s1[ii] / k
                var = (np.maximum(s2[ii] - k * mean * mean, 0) + sd_hint ** 2) / k
                edge = mean - m0
                lam = np.clip(edge / (var + edge * edge), 0, lam_cap / m0)
            x = y[sub, j]
            logk[ii] += np.log1p(lam * (x - m0))
            s1[ii] += x; s2[ii] += x * x; used[ii] = k + 1
            hit = logk[ii] >= target
            label0[ii[hit]] = True
            alive[ii[hit]] = False
        have = n
        idx = np.flatnonzero(alive)
        if n >= 2 and idx.size:
            m = s1[idx] / n + lower
            var = np.maximum((s2[idx] - n * (s1[idx] / n) ** 2) / (n - 1), 0)
            half = stats.t.ppf(1 - level, n - 1) * np.sqrt(var) / math.sqrt(n)
            alive[idx[m + half < c]] = False
    return label0.mean(), used.mean()


def t_label_cost(sample, mu, c, delta, gamma, sd_hint, cfg, trials):
    """Vectorized replica of the t-test label at threshold c: mean replications used."""
    n_max = max(max_reps(delta, gamma, sd_hint, cfg), cfg.n_min)
    looks = look_schedule(n_max, cfg)
    level = gamma / len(looks)
    s1 = np.zeros(trials); s2 = np.zeros(trials); used = np.zeros(trials)
    alive = np.ones(trials, dtype=bool); have = 0
    for n in looks:
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        y = sample(idx.size, n - have)
        s1[idx] += y.sum(axis=1); s2[idx] += (y * y).sum(axis=1); used[idx] = n; have = n
        if n < 2:
            continue
        m = s1[idx] / n
        var = np.maximum((s2[idx] - n * m * m) / (n - 1), 0.0)
        half = stats.t.ppf(1 - level, n - 1) * np.sqrt(var) / math.sqrt(n)
        alive[idx[(m - half > c - delta) | (m + half < c)]] = False
    return used.mean()


def run(regime, delta, eps, n_policies, trials, seed, gamma=0.01, cost_trials=400):
    cfg = CONFIGS["final"]["label"] if "label" in CONFIGS["final"] else LabelConfig(
        n_min=3, growth=2.5, schedule="geometric")
    prob = inventory_bank(regime)
    rng = np.random.default_rng(seed)
    bank = prob.bank
    skew = stats.skew(bank, axis=1)
    # policies: the best ones (the only ones that can be improvers over a good incumbent),
    # the most and least skewed ones, and a random sample of the rest
    # the least skewed policies are closest to normal (or left-skewed): the hardest cases
    best = np.argsort(prob.mu)[:n_policies // 4]
    most_skewed = np.argsort(-skew)[:n_policies // 4]
    least_skewed = np.argsort(skew)[:n_policies // 4]
    rest = rng.choice(prob.n, size=n_policies - 3 * (n_policies // 4), replace=False)
    chosen = np.unique(np.concatenate([best, most_skewed, least_skewed, rest]))
    rows = []
    for k, i in enumerate(chosen):
        row = bank[i].astype(float)
        mu, sd = float(prob.mu[i]), float(prob.sd[i])
        real = lambda r, n, row=row: row[rng.integers(0, row.size, size=(r, n))]
        normal = lambda r, n, mu=mu, sd=sd: mu + sd * rng.standard_normal((r, n))
        p_real, J, nJ = label_zero_rate(real, mu, delta, gamma, sd, cfg, trials, rng)
        p_norm, _, _ = label_zero_rate(normal, mu, delta, gamma, sd, cfg, trials, rng)
        p_bet, _ = label_betting(real, mu, mu + delta + 1e-12, delta, gamma, sd, cfg, trials)
        rows.append({"regime": regime, "policy": int(i), "mu": mu, "sd": sd, "skewness": float(skew[i]),
                     "looks": J, "n_max": nJ, "p0_real": p_real, "p0_normal": p_norm,
                     "p0_betting": p_bet, "trials": trials})
        if (k + 1) % 50 == 0:
            print(f"  {regime}: {k + 1}/{len(chosen)} policies", flush=True)
    # cost of one audit at a typical threshold: incumbent = best policy, c = mu_best - eps,
    # audited policies drawn uniformly (nu), real outputs
    c_typ = float(prob.mu.min()) - eps
    audit = rng.choice(prob.n, size=min(400, prob.n), replace=False)
    cost_t, cost_b = [], []
    for i in audit:
        row = bank[i].astype(float)
        real = lambda r, n, row=row: row[rng.integers(0, row.size, size=(r, n))]
        sd = float(prob.sd[i])
        cost_t.append(t_label_cost(real, prob.mu[i], c_typ, delta, gamma, sd, cfg, cost_trials))
        cost_b.append(label_betting(real, prob.mu[i], c_typ, delta, gamma, sd, cfg, cost_trials)[1])
    df = pd.DataFrame(rows)
    df.attrs["cost_t"], df.attrs["cost_b"] = float(np.mean(cost_t)), float(np.mean(cost_b))
    return df


def recheck_worst(df, delta_of, trials, seed, gamma=0.01, top=10):
    """Re-estimate the worst policies, and any above gamma, with many more trials."""
    cfg = CONFIGS["final"]["label"] if "label" in CONFIGS["final"] else LabelConfig(
        n_min=3, growth=2.5, schedule="geometric")
    rng = np.random.default_rng(seed + 1)
    out = []
    for regime, g in df.groupby("regime"):
        prob = inventory_bank(regime)
        cand = set(g.nlargest(top, "p0_real").policy) | set(g.nlargest(top, "p0_normal").policy) \
            | set(g.policy[(g.p0_real > 0.8 * gamma) | (g.p0_normal > 0.8 * gamma)])
        for _, r in g[g.policy.isin(cand)].iterrows():
            row = prob.bank[int(r.policy)].astype(float)
            real = lambda m, n, row=row: row[rng.integers(0, row.size, size=(m, n))]
            normal = lambda m, n, mu=r.mu, sd=r.sd: mu + sd * rng.standard_normal((m, n))
            p, _, _ = label_zero_rate(real, r.mu, delta_of[regime], gamma, r.sd, cfg, trials, rng)
            q, _, _ = label_zero_rate(normal, r.mu, delta_of[regime], gamma, r.sd, cfg, trials, rng)
            out.append({"regime": regime, "policy": int(r.policy), "p0_real_precise": p,
                        "p0_normal_precise": q, "trials_precise": trials})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=300)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--recheck-trials", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/label_validity.csv")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    deltas = {"nominal": 0.2, "disrupted": 1.0}       # the delta used with each regime
    epss = {"nominal": 0.5, "disrupted": 3.0}
    parts = [run(r, d, epss[r], args.policies, args.trials, args.seed + k)
             for k, (r, d) in enumerate(deltas.items())]
    costs = {p.regime.iloc[0]: (p.attrs["cost_t"], p.attrs["cost_b"]) for p in parts}
    df = pd.concat(parts, ignore_index=True)
    worst = recheck_worst(df, deltas, args.recheck_trials, args.seed)
    df = df.merge(worst, on=["regime", "policy"], how="left")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    gamma = 0.01
    lines = []
    for regime, g in df.groupby("regime", sort=False):
        # worst cases from the 50,000-replay re-estimates (the re-estimated set holds the top
        # policies of the first pass and every policy first estimated above gamma)
        real = g.p0_real_precise.dropna()
        normal = g.p0_normal_precise.dropna()
        ct, cb = costs[regime]
        lines.append([regime.capitalize(), str(len(g)),
                      f"{g.skewness.min():.2f} / {g.skewness.median():.2f} / {g.skewness.max():.2f}",
                      f"{100 * normal.max():.2f}", f"{100 * g.p0_real.mean():.2f}", f"{100 * real.max():.2f}",
                      str(int((real > gamma).sum())), f"{100 * g.p0_betting.max():.2f}",
                      f"{ct:.1f}", f"{cb:.1f}"])
        print(f"{regime}: policies {len(g)}, skewness min {g.skewness.min():.2f} median {g.skewness.median():.2f} "
              f"max {g.skewness.max():.2f}; t-label P(L=0): normal worst {normal.max():.5f}, real mean "
              f"{g.p0_real.mean():.5f} worst {real.max():.5f}, above gamma {int((real > gamma).sum())}; "
              f"betting label worst {g.p0_betting.max():.5f}; reps per audit t {ct:.2f} betting {cb:.2f}")
    tab_dir = os.path.join(args.paper, "tables")
    os.makedirs(tab_dir, exist_ok=True)
    with open(os.path.join(tab_dir, "tab_labelvalidity.tex"), "w") as f:
        f.write("\n".join([
            r"\begin{table}[t]", r"\centering\small",
            r"\caption{Label validity policy by policy (assumption~\ref{A3}) with the inventory simulator's real "
            r"outputs. For each sampled policy the threshold is set where the label is hardest to get right "
            r"($c-\delta$ just above $\mu(\theta)$), and the label procedure is replayed with replications "
            r"resampled from 2{,}000 real outputs (normal outputs with the same mean and standard deviation as a "
            r"reference). Entries are $\Prob(L=0)$ in \%; the requirement is at most $100\gamma=1\%$. Worst "
            r"values of the $t$-test labels are re-estimated with 50{,}000 replays. The last two columns give the "
            r"mean replications per audit at a typical threshold ($c=\mu(\theta^\ast)-\varepsilon$, policies "
            r"drawn uniformly).}",
            r"\label{tab:labelvalidity}", r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrcrrrrrrr}", r"\toprule",
            r" & & & \multicolumn{4}{c}{$t$-test labels} & Betting & \multicolumn{2}{c}{Reps per audit} \\",
            r"\cmidrule(lr){4-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}",
            r"Regime & Policies & Skewness min / med / max & Normal: worst & Real: mean & Real: worst & "
            r"Real $>\gamma$ & Real: worst & $t$-test & Betting \\", r"\midrule"]
            + [" & ".join(l) + r" \\" for l in lines]
            + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]) + "\n")
    print("wrote", args.out, "and", os.path.join(tab_dir, "tab_labelvalidity.tex"))
