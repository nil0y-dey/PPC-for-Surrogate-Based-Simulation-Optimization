"""Precompute ground-truth cost tables for the inventory problem.

Run from the repository root:

    python scripts/build_inventory_tables.py

Writes data/inventory_nominal.npz and data/inventory_disrupted.npz from 2000
replications per policy. The same
seeds and replication counts define the replication banks of experiment E6
(ppc.problems.inventory_bank), so the tables equal the bank means.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppc.inventory import DISRUPTED, NOMINAL, build_table  # noqa: E402

SW = np.arange(0, 100)   # warehouse base-stock levels
SR = np.arange(0, 60)    # retailer base-stock levels
N_REPS = 2000

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for name, params, seed in [("nominal", NOMINAL, 11), ("disrupted", DISRUPTED, 12)]:
        path = os.path.join("data", f"inventory_{name}.npz")
        tab = build_table(params, SW, SR, n_reps=N_REPS, seed=seed, path=path)
        i = int(np.argmin(tab["mu"]))
        print(f"{name}: {len(tab['mu'])} policies, best (S_w, S_r) = {tuple(int(v) for v in tab['thetas'][i])}, "
              f"cost = {tab['mu'][i]:.3f}, max s.e. = {tab['se'].max():.3f} -> {path}")
