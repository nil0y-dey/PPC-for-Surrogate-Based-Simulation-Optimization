"""Two-echelon serial inventory simulator (supplier -> warehouse -> retailer).

Each node follows an installation base-stock policy. The decision is the pair
of integer base-stock levels ``(S_w, S_r)``. Demand at the retailer is Poisson,
unmet demand is backordered, and the external supplier can go down for a few
periods at a time (a simple Markov disruption process).

The simulator is vectorized: it runs many policies and many replications at
once as NumPy arrays. Within one replication, all policies see the same demand
and the same disruption path (common random numbers across policies).

The output of one replication is the average cost per period after a warm-up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SerialParams:
    demand_mean: float = 10.0      # Poisson demand per period at the retailer
    lead_retailer: int = 2         # warehouse -> retailer transit time (periods, >= 1)
    lead_warehouse: int = 3        # supplier -> warehouse transit time (periods, >= 1)
    h_retailer: float = 1.0        # holding cost per unit per period at the retailer
    h_warehouse: float = 0.5       # holding cost per unit per period at the warehouse
    backorder_cost: float = 9.0    # cost per backordered unit per period
    p_fail: float = 0.0            # P(supplier goes down next period | up)
    p_recover: float = 0.3         # P(supplier comes back next period | down)
    horizon: int = 300             # periods counted in the cost
    warmup: int = 50               # periods discarded at the start


NOMINAL = SerialParams()
DISRUPTED = SerialParams(p_fail=0.05, p_recover=0.3)


def simulate_serial(S_w, S_r, n_reps: int, params: SerialParams, rng: np.random.Generator) -> np.ndarray:
    """Simulate policies ``(S_w[i], S_r[i])``.

    Returns an array of shape ``(len(S_w), n_reps)`` holding the average cost
    per period of each replication.
    """
    S_w = np.asarray(S_w, dtype=float)[:, None]
    S_r = np.asarray(S_r, dtype=float)[:, None]
    if S_w.shape != S_r.shape:
        raise ValueError("S_w and S_r must have the same length")
    Lr, Lw = int(params.lead_retailer), int(params.lead_warehouse)
    if Lr < 1 or Lw < 1:
        raise ValueError("lead times must be at least 1 period")

    P = S_w.shape[0]
    shape = (P, n_reps)
    inv_r = np.repeat(S_r, n_reps, axis=1)        # retailer net inventory (negative = backorders)
    onhand_w = np.repeat(S_w, n_reps, axis=1)     # warehouse on-hand stock
    owed_w = np.zeros(shape)                      # units the warehouse owes the retailer
    supplier_owed = np.zeros(shape)               # units the supplier owes the warehouse
    pipe_r = np.zeros((Lr,) + shape)              # in transit to the retailer (circular buffer)
    pipe_w = np.zeros((Lw,) + shape)              # in transit to the warehouse
    pipe_r_tot = np.zeros(shape)
    pipe_w_tot = np.zeros(shape)
    up = np.ones(n_reps, dtype=bool)              # supplier state, shared by all policies
    total = np.zeros(shape)

    for t in range(params.warmup + params.horizon):
        sr, sw = t % Lr, t % Lw
        # 1. arrivals (the slot emptied now is refilled below, arriving L periods later)
        inv_r += pipe_r[sr]
        pipe_r_tot -= pipe_r[sr]
        pipe_r[sr] = 0.0
        onhand_w += pipe_w[sw]
        pipe_w_tot -= pipe_w[sw]
        pipe_w[sw] = 0.0

        # 2. demand at the retailer
        demand = rng.poisson(params.demand_mean, size=n_reps).astype(float)
        inv_r -= demand[None, :]

        # 3. retailer orders up to S_r (position includes units owed by the warehouse)
        ip_r = inv_r + pipe_r_tot + owed_w
        owed_w += np.maximum(S_r - ip_r, 0.0)

        # 4. warehouse ships what it can
        ship = np.minimum(owed_w, onhand_w)
        onhand_w -= ship
        owed_w -= ship
        pipe_r[sr] += ship
        pipe_r_tot += ship

        # 5. warehouse orders up to S_w from the supplier
        ip_w = onhand_w - owed_w + pipe_w_tot + supplier_owed
        supplier_owed += np.maximum(S_w - ip_w, 0.0)

        # 6. supplier ships everything it owes if it is up
        if params.p_fail > 0:
            u = rng.random(n_reps)
            up = np.where(up, u >= params.p_fail, u < params.p_recover)
        ship_s = supplier_owed * up[None, :]
        supplier_owed -= ship_s
        pipe_w[sw] += ship_s
        pipe_w_tot += ship_s

        # 7. costs
        if t >= params.warmup:
            total += (params.h_retailer * np.maximum(inv_r, 0.0)
                      + params.backorder_cost * np.maximum(-inv_r, 0.0)
                      + params.h_warehouse * onhand_w)

    return total / params.horizon


def build_bank(params: SerialParams, sw_values, sr_values, n_reps: int = 2000,
               seed: int = 0, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_reps`` replications of every policy on a grid.

    Returns ``(thetas, Y)`` with ``Y[i, r]`` the average cost per period of policy ``i`` in
    replication ``r`` (float32). Policies in one chunk share random numbers (CRN), which
    does not matter for later use: experiments resample each policy's row independently.
    """
    sw_values = np.asarray(sw_values)
    sr_values = np.asarray(sr_values)
    a, b = np.meshgrid(sw_values, sr_values, indexing="ij")
    thetas = np.column_stack([a.ravel(), b.ravel()]).astype(int)
    rng = np.random.default_rng(seed)
    Y = np.empty((len(thetas), n_reps), dtype=np.float32)
    for start in range(0, len(thetas), chunk):
        sl = slice(start, start + chunk)
        Y[sl] = simulate_serial(thetas[sl, 0], thetas[sl, 1], n_reps, params, rng)
    return thetas, Y


def build_table(params: SerialParams, sw_values, sr_values, n_reps: int = 2000,
                seed: int = 0, chunk: int = 1000, path: str | None = None) -> dict:
    """Mean cost and per-replication sd for every policy on a grid (from ``build_bank``).

    The result serves as ground truth for experiments E1-E5. With 2000 replications the
    standard error of each mean is about sd / 45.
    """
    thetas, Y = build_bank(params, sw_values, sr_values, n_reps, seed, chunk)
    mu = Y.mean(axis=1, dtype=np.float64)
    sd = Y.std(axis=1, ddof=1, dtype=np.float64)
    table = {"thetas": thetas, "mu": mu, "sd": sd, "se": sd / np.sqrt(n_reps),
             "n_reps": n_reps, **{k: v for k, v in asdict(params).items()}}
    if path is not None:
        np.savez_compressed(path, **table)
    return table
