"""Turn experiment CSVs into the paper's figures and LaTeX tables.

    python scripts/make_figures.py --results results --paper paper

Writes paper/figures/*.pdf (plus .png previews) and paper/tables/*.tex. Experiments that have not been
run yet are skipped with a message, so this can be re-run at any time.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ppc.experiments import load_results  # noqa: E402

PROB_ORDER = ["synthetic-smooth", "synthetic-cliff", "synthetic-needle", "inventory-nominal", "inventory-shift"]
PROB_LABEL = {"synthetic-smooth": "Smooth", "synthetic-cliff": "Cliff", "synthetic-needle": "Needle",
              "inventory-nominal": "Inventory", "inventory-shift": "Inventory (shift)"}
SURR_LABEL = {"gp": "GP", "forest": "Forest", "quadratic": "Quadratic"}
METH_LABEL = {"uniform": "Uniform", "ppi": "PPI", "ppi_1s": "PPI-1s", "ppc": "PPC", "ppc_1s": "PPC-1s",
              "ppi_1s_refit": "PPI-1s-refit", "model_only": "Model only"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


METHOD_LABEL = {"uniform": "Uniform", "ppi": "PPI", "ppi_1s": "PPI-1s", "ppc": "PPC",
                "ppc_1s": "PPC-1s", "ppi_1s_refit": "PPI-1s-refit"}


def save_fig(fig, fig_dir, name):
    """PDF for the manuscript, PNG for viewing in the notebook."""
    fig.savefig(os.path.join(fig_dir, name + ".pdf"))
    fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=110)


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def ordered(values):
    return [p for p in PROB_ORDER if p in set(values)]


def write_table(path, header, rows, caption, label, align):
    wide = len(header) >= 6
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if wide:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else ""), r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def num(x, digits=2):
    return "--" if pd.isna(x) else f"{x:.{digits}f}"


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def e1(df, fig_dir, tab_dir):
    d = df[(df.experiment == "E1") & (df.method == "model_only")]
    if d.empty:
        print("E1: no results yet"); return
    surrs = [s for s in SURR_LABEL if s in set(d.surrogate)]
    rows = []
    for p in ordered(d.problem):
        dp = d[d.problem == p]
        cells = [pct(dp[dp.surrogate == s].violated.mean()) for s in surrs]
        rows.append([PROB_LABEL[p], pct((dp.M_true > 0).mean())] + cells + [str(len(dp) // max(len(surrs), 1))])
    write_table(os.path.join(tab_dir, "tab_diagnosis.tex"),
                ["Problem", r"Runs with $M>0$ (\%)"] + [f"{SURR_LABEL[s]} (\\%)" for s in surrs] + ["Runs"],
                rows, r"Diagnosis (E1): share of runs in which the surrogate's own estimate of the "
                      r"improvement mass falls below the true mass (no audits).",
                "tab:diagnosis", "l" + "r" * (len(surrs) + 2))
    s = d.dropna(subset=["recall_search", "recall_random"])
    if s.empty:
        return
    g = s.groupby("problem")[["recall_search", "recall_random"]].mean().reindex(ordered(s.problem))
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(g))
    ax.bar(x - 0.18, g.recall_search, 0.36, label="surrogate after search", color="#d62728")
    ax.bar(x + 0.18, g.recall_random, 0.36, label="surrogate on random design", color="#7f7f7f")
    ax.set_xticks(x, [PROB_LABEL[p] for p in g.index])
    ax.set_ylabel("recall of policies that\nstill beat the incumbent")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "fig_selection")
    plt.close(fig)
    print("wrote fig_selection.pdf")


def e2(df, fig_dir, tab_dir):
    d = df[(df.experiment == "E2") & (df.method.isin(["uniform", "ppi_1s"]))]
    if d.empty:
        print("E2: no results yet"); return
    rows = []
    for p in ordered(d.problem):
        for s in [s for s in SURR_LABEL if s in set(d.surrogate)]:
            cells = []
            for m in ["uniform", "ppi_1s"]:
                x = d[(d.problem == p) & (d.surrogate == s) & (d.method == m)]
                k, n = int(x.violated.sum()), len(x)
                cells.append(f"{pct(k / n if n else np.nan)} [{pct(wilson_upper(k, n))}]")
            rows.append([PROB_LABEL[p], SURR_LABEL[s]] + cells + [str(n)])
    write_table(os.path.join(tab_dir, "tab_validity.tex"),
                ["Problem", "Surrogate", r"Uniform (\%)", r"PPI-1s (\%)", "Runs"], rows,
                r"Validity (E2): share of runs in which the certified bound ever fell below the true "
                r"improvement mass, with the 95\% Wilson upper limit in brackets. Nominal level "
                r"$\alpha+\beta=7.5\%$.", "tab:validity", "llrrr")
    cols = [c for c in d.columns if c.startswith("b0") or c == "b100"]
    if not cols:
        return
    fracs = [int(c[1:]) / 100 for c in cols]
    probs = ordered(d.problem)
    fig, axes = plt.subplots(1, len(probs), figsize=(2.6 * len(probs), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, probs):
        for m, st in [("uniform", dict(color="#333333")), ("ppi_1s", dict(color="#1f77b4", ls="--"))]:
            x = d[(d.problem == p) & (d.method == m)]
            ax.plot(fracs, [x[c].mean() for c in cols], marker="o", ms=3, label=METH_LABEL[m], **st)
        ax.plot(fracs, [d[d.problem == p].M_true.mean()] * len(fracs), ls=":", color="black", label="true mass")
        ax.set_yscale("log")
        ax.set_title(PROB_LABEL[p])
        ax.set_xlabel("share of audit budget")
    axes[0].set_ylabel("mean certified bound")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, fig_dir, "fig_budget")
    plt.close(fig)
    print("wrote fig_budget.pdf")


def e3(df, tab_dir):
    d = df[(df.experiment == "E3") & (df.method == "uniform")]
    if d.empty:
        print("E3: no results yet"); return
    names = {"baseline": "EB, labels from 10", "betting": "Betting, labels from 10",
             "betting+geo5": "Betting, labels from 5", "final": "Betting, labels from 3 (final)"}
    rows = []
    for p in ordered(d.problem):
        for c in names:
            x = d[(d.problem == p) & (d.config == c)]
            if x.empty:
                continue
            x0 = x[x.M_true == 0]
            cp = x0.bound_cp_fixed_n.mean() if "bound_cp_fixed_n" in x0 and len(x0) else np.nan
            rows.append([PROB_LABEL[p], names[c], f"{x.n_audits.mean():.0f}",
                         f"{x0.bound_final.mean():.4f}" if len(x0) else "--", num(cp, 4),
                         pct((~x.reps_to_target.isna()).mean())])
    write_table(os.path.join(tab_dir, "tab_ablation.tex"),
                ["Problem", "Procedure", "Audits", r"Bound when $M=0$", r"Fixed-$n$ CP when $M=0$",
                 r"Certified at $p^\ast$ (\%)"], rows,
                r"Ablation (E3), uniform auditing with the GP search, all on the same replication budget: "
                r"audits per run, mean final bound in runs where nothing beats the incumbent, the "
                r"Clopper--Pearson bound for the same audits as if their number had been fixed in advance "
                r"(not anytime-valid; the gap is the price of stopping at will), and share of runs certified "
                r"at $p^\ast=0.02$.", "tab:ablation", "llrrrr")


def e4(df, fig_dir):
    d = df[df.experiment == "E4"]
    d = d[d.method != "model_only"]
    if d.empty:
        print("E4: no results yet"); return
    d = d.assign(key=d.config + ":" + d.method)
    piv = d.pivot_table(index=["problem", "surrogate", "seed"], columns="key", values="bound_final")
    ref = "final:uniform"
    order = ["final:ppi", "final:ppi_1s", "final:ppc", "final:ppc_1s", "final:ppi_1s_refit",
             "final+guided:uniform", "final+guided:ppi_1s"]
    keys = [k for k in order if k in piv.columns] + [k for k in piv.columns if k not in order and k != ref]
    probs = ordered(d.problem)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / max(len(keys), 1)
    for j, k in enumerate(keys):
        meds = [(piv.loc[p][k] / piv.loc[p][ref]).median() for p in probs]
        cfg, m = k.split(":")
        lab = METH_LABEL.get(m, m) + (", surrogate first looks" if cfg == "final+guided" else "")
        ax.bar(np.arange(len(probs)) + (j - (len(keys) - 1) / 2) * width, meds, width, label=lab)
    ax.axhline(1.0, color="black", lw=1)
    ax.set_xticks(np.arange(len(probs)), [PROB_LABEL[p] for p in probs])
    ax.set_ylabel("median bound ratio\nto uniform auditing")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_fig(fig, fig_dir, "fig_surrogate")
    plt.close(fig)
    print("wrote fig_surrogate.pdf")


def e5(df, tab_dir):
    d = df[df.experiment == "E5"]
    if d.empty:
        print("E5: no results yet"); return
    rows = []
    for s in [s for s in SURR_LABEL if s in set(d.surrogate)]:
        x = d[d.surrogate == s]
        mo = x[x.method == "model_only"]
        un = x[x.method == "uniform"]
        pp = x[x.method == "ppi_1s"]
        rows.append([SURR_LABEL[s], pct(mo.violated.mean()), pct(un.violated.mean()), pct(pp.violated.mean()),
                     f"{un.M_true.mean():.3f}", f"{un.bound_final.mean():.3f}",
                     f"{mo.recall_search.mean():.2f}", f"{mo.recall_random.mean():.2f}"])
    write_table(os.path.join(tab_dir, "tab_shift.tex"),
                ["Surrogate", r"Model only wrong (\%)", r"Uniform wrong (\%)", r"PPI-1s wrong (\%)",
                 "Mean $M$", "Mean uniform bound", "Recall (search)", "Recall (random)"], rows,
                r"Regime shift (E5): surrogate and search from the nominal regime, certificate in the "
                r"disrupted regime: share of runs in which each estimate fell below the true mass, mean true "
                r"mass and mean final uniform bound, and recall of the remaining improvers.",
                "tab:shift", "lrrrrrrr")


def e1_mechanism(df, tab_dir):
    """Quantities of Theorem 3 / Corollary 2 after search (E1)."""
    d = df[(df.experiment == "E1") & (df.method == "model_only")]
    need = ["recall_search", "recall_random", "flag_mass", "false_alarm_mass", "flags_unevaluated", "lift_missed"]
    if d.empty or not all(c in d.columns for c in need):
        print("E1 mechanism: no results with theory diagnostics yet"); return
    rows = []
    for p in ordered(d.problem):
        for s in [s for s in SURR_LABEL if s in set(d[d.problem == p].surrogate)]:
            x = d[(d.problem == p) & (d.surrogate == s)]
            rows.append([PROB_LABEL[p], SURR_LABEL[s],
                         num(x.recall_search.mean()), num(x.recall_random.mean()),
                         num(x.flag_mass.mean(), 4), num(x.false_alarm_mass.mean(), 4),
                         "--" if x.lift_missed.isna().all() else f"{x.lift_missed.median():.2f}",
                         pct((x.flags_unevaluated == 0).mean())])
    write_table(os.path.join(tab_dir, "tab_mechanism.tex"),
                ["Problem", "Surrogate", "Recall $r$ (search)", "Recall (random)", "Flag mass $P$",
                 r"False alarms $\phi$", r"Lift $\Lambda_{\mathrm{act}}$ (median)", r"Exhausted (\%)"], rows,
                r"Mechanism (E1): recall of the remaining improvers by the surrogate after search and by a "
                r"surrogate fitted to a random design of the same size, flagged and false-alarm masses, lift of "
                r"the active part $\pi_{\mathrm{act}}$ of the PPC-1s design on missed improvers, and share of runs in which no flagged policy was left "
                r"unsimulated (Theorem~\ref{thm:selection}, Corollary~\ref{cor:exhaust}).",
                "tab:mechanism", "llrrrrrr")


E6_SETTINGS = [("inventory", "Nominal", "normal"), ("inventory_real", "Nominal", "real"),
               ("inventory_disrupted", "Disrupted", "normal"), ("inventory_disrupted_real", "Disrupted", "real"),
               ("inventory_shift", "Shift", "normal"), ("inventory_shift_real", "Shift", "real")]


def e6(df, tab_dir):
    """Normal noise versus resampled real simulator outputs (E6), pooled over surrogates."""
    d = df[df.experiment == "E6"]
    if d.empty:
        print("E6: no results yet"); return
    rows = []
    for setting, regime, noise in E6_SETTINGS:
        x = d[d.setting == setting]
        if x.empty:
            continue
        un, pp = x[x.method == "uniform"], x[x.method == "ppi_1s"]
        runs = len(un)
        k_un, k_pp = int(un.violated.sum()), int(pp.violated.sum())
        miss = 1 - un.u_bar_covers.mean()
        rows.append([regime, noise, str(runs),
                     f"{pct(un.violated.mean())} ({pct(wilson_upper(k_un, runs))})",
                     f"{pct(pp.violated.mean())} ({pct(wilson_upper(k_pp, len(pp)))})",
                     pct(miss), pct((~un.reps_to_target.isna()).mean()), f"{un.bound_final.median():.3f}"])
    write_table(os.path.join(tab_dir, "tab_realnoise.tex"),
                ["Regime", "Outputs", "Runs", r"Uniform wrong \% (UCL)", r"PPI-1s wrong \% (UCL)",
                 r"$\bar U<\mu(\hat\theta)$ (\%)", r"Certified at $p^\ast$ (\%)", "Median bound"], rows,
                r"Real simulator outputs (E6): replications drawn from normal noise around the ground truth "
                r"versus resampled from 2{,}000 actual simulator outputs per policy, on the same seeds. "
                r"Violation rates with 95\% Wilson upper limits; target $\alpha+\beta=7.5\%$. "
                r"The verification column should stay near $\beta=2.5\%$.", "tab:realnoise", "llrrrrrr")


def e7(df, tab_dir):
    """Above the resolution (E7): early-stopped search; paired bound ratios to uniform auditing."""
    d = df[df.experiment == "E7"]
    if d.empty:
        print("E7: no results yet"); return
    labels = {"smooth_early": "Smooth", "cliff_early": "Cliff", "inventory_early": "Inventory"}
    meth = [m for m in ("ppi_1s", "ppi", "ppc_1s") if m in set(d.method)]
    key = ["setting", "surrogate", "seed"]
    base = d[d.method == "model_only"].set_index(key)
    uni = d[d.method == "uniform"].set_index(key)["bound_final"]
    rows = []
    for st in [s for s in labels if s in set(d.setting)]:
        for sur in [s for s in SURR_LABEL if s in set(d[d.setting == st].surrogate)]:
            b = base.xs((st, sur), level=(0, 1), drop_level=False)
            cells = []
            for m in meth:
                x = d[(d.method == m) & (d.setting == st) & (d.surrogate == sur)].set_index(key)["bound_final"]
                r = (x / uni.reindex(x.index)).replace([np.inf], np.nan).dropna()
                cells.append(num(r.median()))
            u = uni.xs((st, sur), level=(0, 1))
            rows.append([labels[st], SURR_LABEL[sur], num(b.M_true.mean(), 3), num(u.median(), 3),
                         num(b.recall_search.mean()), num(b.false_alarm_mass.mean(), 4)] + cells)
    write_table(os.path.join(tab_dir, "tab_aboveres.tex"),
                ["Problem", "Surrogate", "Mean $M$", "Uniform bound", "Recall $r$", r"False alarms $\phi$"]
                + [f"{METHOD_LABEL.get(m, m)} / uniform" for m in meth], rows,
                r"Above the resolution (E7): the search stops after its initial design, so a sizable "
                r"improvement mass remains. Median ratio of each variant's final bound to the uniform bound on "
                r"the same seeds (below 1: the surrogate helps), with the recall and false-alarm mass of "
                r"Theorem~\ref{thm:selection}.", "tab:aboveres", "llrrrr" + "r" * len(meth))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    fig_dir = os.path.join(args.paper, "figures")
    tab_dir = os.path.join(args.paper, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    df = load_results(args.results)
    if df.empty:
        sys.exit(f"No results found under {args.results}/E*/")
    e1(df, fig_dir, tab_dir)
    e2(df, fig_dir, tab_dir)
    e3(df, tab_dir)
    e4(df, fig_dir)
    e5(df, tab_dir)
    e1_mechanism(df, tab_dir)
    e6(df, tab_dir)
    e7(df, tab_dir)
