"""Experiment presets for the paper (E1-E7).

Each preset is a list of "runs": (config name, methods, problems, surrogates, extra options).
Results land in ``<out_dir>/<exp>/<exp>_<config>_<tag>_seed<seed>.csv``.

The paper's run plan (``PLAN``) splits every experiment into numbered *parts* of 100
macro-replications, each with its own fixed seed (part k uses seed k). ``run_part(exp, k)``
runs one part; it skips a part that is already finished and continues a half-finished one,
so after a disconnect you simply run the same parts again. ``progress()`` shows what is left.

Run plan: 500 runs (5 parts) for E1 and E2, 200 runs (2 parts) for E3-E7.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from .runner import CONFIGS, run_settings

ALL = ("smooth", "cliff", "needle", "inventory")
CORE = ("smooth", "cliff", "inventory")
SURR = ("gp", "forest", "quadratic")
EARLY = ("smooth_early", "cliff_early", "inventory_early")
INV_PAIRED = ("inventory", "inventory_real", "inventory_disrupted", "inventory_disrupted_real",
              "inventory_shift", "inventory_shift_real")

EXPERIMENTS = {
    # E1 - diagnosis and selection effect: does the surrogate's own estimate of what is left hold up,
    #      and does the search surrogate miss the policies that still beat the incumbent?
    "E1": {"title": "Diagnosis of surrogate-based stopping and the selection effect",
           "runs": [("final", (), ALL, SURR, {"selection_metrics": True})]},
    # E2 - validity of the recommended procedure at scale, and the bound as the budget grows
    "E2": {"title": "Validity and budget curves of the certified stopping rule",
           "runs": [("final", ("uniform", "ppi_1s"), ALL, SURR, {})]},
    # E3 - ablation: which ingredient makes the certificate practical?
    "E3": {"title": "Ablation of confidence sequence and label schedule",
           "runs": [(c, ("uniform",), CORE, ("gp",), {})
                    for c in ("baseline", "betting", "betting+geo5", "final")]},
    # E4 - can a surrogate make the certificate cheaper? (every variant we know of)
    "E4": {"title": "Surrogate-powered certificate variants versus uniform auditing",
           "runs": [("final", ("uniform", "ppi", "ppi_1s", "ppc", "ppc_1s", "ppi_1s_refit"), CORE, SURR, {}),
                    ("final+guided", ("uniform", "ppi_1s"), CORE, SURR, {})]},
    # E5 - regime shift: surrogate trained before a disruption, certificate needed after it
    "E5": {"title": "Certification after a disruption-regime shift",
           "runs": [("final", ("uniform", "ppi_1s"), ("inventory_shift",), SURR, {"selection_metrics": True})]},
    # E6 - real simulator outputs: replications resampled from the simulator's actual (skewed)
    #      output distribution, paired with the normal-noise version on the same seeds
    "E6": {"title": "Validity with real simulator outputs instead of normal noise",
           "runs": [("final", ("uniform", "ppi_1s"), INV_PAIRED, SURR, {"selection_metrics": True})]},
    # E7 - above the resolution: the search stops after its initial design, leaving a sizable
    #      improvement mass, which is where Theorem 3(b) allows a surrogate with good recall to help
    "E7": {"title": "Above the resolution: surrogate-powered variants after an early-stopped search",
           "runs": [("final", ("uniform", "ppi_1s", "ppi", "ppc_1s"), EARLY, SURR, {"selection_metrics": True})]},
}


# Paper run plan: part k of an experiment uses seed k and 100 macro-replications.
PLAN = {"E1": 5, "E2": 5, "E3": 2, "E4": 2, "E5": 2, "E6": 2, "E7": 2}          # number of parts
RUNS_PER_PART = 100
ORDER = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")


def _expected_runs(exp: str, n_macro: int) -> list[tuple[str, int]]:
    """(config, number of runs) for every CSV file one part of ``exp`` writes."""
    return [(config, len(probs) * len(surrs) * n_macro)
            for config, _, probs, surrs, _ in EXPERIMENTS[exp]["runs"]]


def _runs_in(path: str) -> int:
    """Number of complete runs saved in one result file."""
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, usecols=["setting", "surrogate", "seed"])
    except (pd.errors.EmptyDataError, ValueError):
        return 0
    return len(df.drop_duplicates())


def _part_path(out_dir: str, exp: str, config: str, part: int) -> str:
    return os.path.join(out_dir, exp, f"{exp}_{config}_all_seed{part}.csv")


def run_part(exp: str, part: int, out_dir: str = "results", n_macro: int | None = None,
             verbose: bool = False) -> list[str]:
    """Run part ``part`` (1, 2, ...) of the run plan for experiment ``exp``.

    Finished parts are skipped; interrupted parts continue where they stopped, with
    results identical to an uninterrupted run. ``n_macro`` overrides the 100 runs per
    part (for tests; the environment variable ``CS_N_MACRO`` does the same).
    """
    if exp not in PLAN:
        raise ValueError(f"unknown experiment {exp!r}; choose from {list(PLAN)}")
    if not 1 <= part <= PLAN[exp]:
        raise ValueError(f"{exp} has parts 1-{PLAN[exp]}, not {part}")
    n_macro = n_macro or int(os.environ.get("CS_N_MACRO", RUNS_PER_PART))
    expected = _expected_runs(exp, n_macro)
    files = [_part_path(out_dir, exp, c, part) for c, _ in expected]
    if not all(_runs_in(f) >= n for f, (_, n) in zip(files, expected)):
        run_experiment(exp, n_macro, seed=part, out_dir=out_dir, verbose=False, resume=True)
    if verbose:
        print(f"{exp} part {part}/{PLAN[exp]}: done", flush=True)
    return files


def progress(out_dir: str = "results", n_macro: int | None = None) -> pd.DataFrame:
    """Parts finished and runs saved, per experiment."""
    n_macro = n_macro or int(os.environ.get("CS_N_MACRO", RUNS_PER_PART))
    rows = []
    for exp in ORDER:
        expected = _expected_runs(exp, n_macro)
        total = sum(n for _, n in expected)
        done_parts, saved = 0, 0
        for part in range(1, PLAN[exp] + 1):
            have = sum(min(_runs_in(_part_path(out_dir, exp, c, part)), n) for c, n in expected)
            saved += have
            done_parts += have >= total
        rows.append({"experiment": exp, "parts done": f"{done_parts}/{PLAN[exp]}",
                     "runs saved": f"{saved}/{total * PLAN[exp]}"})
    return pd.DataFrame(rows).set_index("experiment")


def run_experiment(exp: str, n_macro: int, seed: int = 0, out_dir: str = "results",
                   problems=None, surrogates=None, verbose: bool = True,
                   resume: bool = False) -> list[str]:
    """Run one preset. Returns the list of CSV files written.

    With ``resume=True`` runs already saved in the target files are skipped.
    """
    spec = EXPERIMENTS[exp]
    folder = os.path.join(out_dir, exp)
    os.makedirs(folder, exist_ok=True)
    written = []
    for config, methods, probs, surrs, extra in spec["runs"]:
        probs = tuple(p for p in probs if problems is None or p in problems)
        surrs = tuple(s for s in surrs if surrogates is None or s in surrogates)
        if not probs or not surrs:
            continue
        tag = "-".join(probs) if problems is not None else "all"
        path = os.path.join(folder, f"{exp}_{config}_{tag}_seed{seed}.csv")
        t0 = time.time()
        run_settings(probs, surrs, n_macro=n_macro, seed=seed, out_csv=path, verbose=verbose,
                  methods=tuple(methods), config=CONFIGS[config], config_name=config,
                  resume=resume, **extra)
        if verbose:
            print(f"[{exp}] {config}: saved {path} ({time.time() - t0:.0f}s)", flush=True)
        written.append(path)
    return written


def load_results(out_dir: str = "results", exp: str | None = None) -> pd.DataFrame:
    """Combine every CSV under ``out_dir`` (or ``out_dir/exp``) into one table."""
    frames = []
    for root, _, files in os.walk(out_dir if exp is None else os.path.join(out_dir, exp)):
        for f in files:
            if f.endswith(".csv") and f.startswith("E"):
                df = pd.read_csv(os.path.join(root, f))
                df["experiment"] = f.split("_")[0]
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["experiment", "config", "setting", "surrogate", "seed", "method"])
