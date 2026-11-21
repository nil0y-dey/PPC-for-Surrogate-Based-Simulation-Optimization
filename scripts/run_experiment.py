"""Run the paper's experiments (E1-E7) from the command line.

The run plan splits each experiment into numbered parts of 100 runs (E1 and E2: parts
1-5; E3-E7: parts 1-2). Part k uses seed k. Finished parts are skipped and interrupted
parts continue where they stopped, so re-running a command is always safe.

From the repository root:

    python scripts/run_experiment.py --all                 # the whole plan
    python scripts/run_experiment.py --exp E1              # every part of E1
    python scripts/run_experiment.py --exp E1 --part 3     # one part
    python scripts/run_experiment.py --progress            # what is done

Custom runs outside the plan write their own files:

    python scripts/run_experiment.py --exp E2 --n-macro 50 --seed 99 --problems inventory
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppc.experiments import EXPERIMENTS, ORDER, PLAN, progress, run_experiment, run_part  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(EXPERIMENTS))
    ap.add_argument("--part", type=int, default=None, help="part of the run plan (default: all parts)")
    ap.add_argument("--all", action="store_true", help="run the whole plan, quick experiments first")
    ap.add_argument("--progress", action="store_true", help="show what is done and what is left")
    ap.add_argument("--out-dir", default="results")
    custom = ap.add_argument_group("custom runs outside the plan")
    custom.add_argument("--n-macro", type=int, default=None)
    custom.add_argument("--seed", type=int, default=None)
    custom.add_argument("--problems", nargs="+", default=None)
    custom.add_argument("--surrogates", nargs="+", default=None)
    args = ap.parse_args()

    if args.progress:
        print(progress(args.out_dir).to_string())
    elif args.all:
        for exp in ORDER:
            for part in range(1, PLAN[exp] + 1):
                run_part(exp, part, args.out_dir, verbose=True)
        print(progress(args.out_dir).to_string())
    elif args.exp is None:
        ap.error("give --exp, --all or --progress")
    elif args.seed is not None or args.n_macro is not None or args.problems or args.surrogates:
        print(f"{args.exp}: {EXPERIMENTS[args.exp]['title']} (custom run)", flush=True)
        run_experiment(args.exp, args.n_macro or 100, args.seed or 0, args.out_dir,
                       args.problems, args.surrogates, resume=True)
    else:
        parts = [args.part] if args.part else range(1, PLAN[args.exp] + 1)
        for part in parts:
            run_part(args.exp, part, args.out_dir, verbose=True)
