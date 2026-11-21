"""Test problems with a known ground truth.

Every problem is a finite set of integer solutions ``thetas`` (shape ``(N, d)``)
with a known mean cost ``mu`` (shape ``(N,)``) and a per-replication noise
standard deviation ``sd`` (shape ``(N,)``). Knowing ``mu`` exactly is what lets
us check whether a certificate is valid.

Costs are minimized throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TabulatedProblem:
    """A finite simulation-optimization problem with known means.

    Simulation output for solution ``i`` is ``mu[i] + sd[i] * N(0, 1)``. For the
    inventory problem, ``mu`` and ``sd`` come from a long offline simulation, and
    the normal approximation is justified because each replication is an
    average over many periods.
    """

    thetas: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    name: str = "problem"
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.thetas = np.asarray(self.thetas)
        self.mu = np.asarray(self.mu, dtype=float)
        self.sd = np.asarray(self.sd, dtype=float)
        if self.thetas.ndim != 2:
            raise ValueError("thetas must have shape (N, d)")
        if not (len(self.thetas) == len(self.mu) == len(self.sd)):
            raise ValueError("thetas, mu and sd must have the same length")
        lo = self.thetas.min(axis=0)
        hi = self.thetas.max(axis=0)
        self._lo = lo
        self._span = np.where(hi > lo, hi - lo, 1)

    @property
    def n(self) -> int:
        return len(self.mu)

    @property
    def dim(self) -> int:
        return self.thetas.shape[1]

    def features(self, idx=None) -> np.ndarray:
        """Solutions rescaled to [0, 1]^d (inputs for the surrogate)."""
        th = self.thetas if idx is None else self.thetas[idx]
        return (th - self._lo) / self._span

    def simulate(self, idx: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``n_reps`` independent replications of solution ``idx``."""
        return self.mu[idx] + self.sd[idx] * rng.standard_normal(n_reps)

    def improvement_mass(self, threshold: float, nu: np.ndarray | None = None) -> float:
        """True nu-mass of solutions with mean strictly below ``threshold``."""
        below = self.mu < threshold
        if nu is None:
            return float(below.mean())
        return float(nu[below].sum())


# ----------------------------------------------------------------------------
# Synthetic problems on a 2-D integer grid
# ----------------------------------------------------------------------------

def _base_bowl(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Smooth, mildly wiggly bowl with its minimum near (0.35, 0.6)."""
    return 8.0 * ((x1 - 0.35) ** 2 + 1.5 * (x2 - 0.6) ** 2) + 0.3 * np.sin(8 * x1) * np.cos(6 * x2)


def synthetic_problem(kind: str = "smooth", grid: int = 100, noise_sd: float = 1.0) -> TabulatedProblem:
    """Build a synthetic problem on a ``grid x grid`` integer lattice.

    kind:
      * ``"smooth"`` - a smooth bowl; a Gaussian-process surrogate fits it well.
      * ``"cliff"``  - the bowl plus a corner plateau that is *better* than the
        bowl's minimum. The jump is the kind of threshold effect that a smooth
        surrogate tends to miss.
      * ``"needle"`` - the bowl plus one narrow, deep well (a tiny region of
        much better solutions).
    """
    g = np.arange(grid)
    t1, t2 = np.meshgrid(g, g, indexing="ij")
    thetas = np.column_stack([t1.ravel(), t2.ravel()])
    x1 = thetas[:, 0] / (grid - 1)
    x2 = thetas[:, 1] / (grid - 1)
    mu = _base_bowl(x1, x2)

    if kind == "smooth":
        pass
    elif kind == "cliff":
        corner = (x1 > 0.78) & (x2 < 0.22)
        mu = np.where(corner, -0.6 + 0.5 * ((x1 - 0.9) ** 2 + (x2 - 0.1) ** 2), mu)
    elif kind == "needle":
        r2 = (x1 - 0.85) ** 2 + (x2 - 0.15) ** 2
        mu = mu - 6.0 * np.exp(-r2 / (2 * 0.025**2))
    else:
        raise ValueError(f"unknown kind {kind!r}")

    sd = np.full_like(mu, float(noise_sd))
    return TabulatedProblem(thetas, mu, sd, name=f"synthetic-{kind}",
                            info={"grid": grid, "noise_sd": noise_sd})


def load_table(path: str, name: str | None = None) -> TabulatedProblem:
    """Load a precomputed ground-truth table saved by ``ppc.inventory.build_table``."""
    data = np.load(path, allow_pickle=False)
    info = {k: data[k].item() for k in data.files if k not in ("thetas", "mu", "sd", "se")}
    return TabulatedProblem(data["thetas"], data["mu"], data["sd"],
                            name=name or str(path), info=info)


class EmpiricalProblem(TabulatedProblem):
    """A problem whose replications are resampled from a bank of real simulator outputs.

    ``bank[i, r]`` is replication ``r`` of solution ``i``. Each call to ``simulate`` draws
    with replacement from row ``i``, so replications keep the simulator's real distribution
    (skewness, heavy tails) and the true mean is exactly the row mean.
    """

    def __init__(self, thetas, bank, name="empirical", info=None):
        bank = np.asarray(bank, dtype=np.float32)
        mu = bank.mean(axis=1, dtype=np.float64)
        sd = bank.std(axis=1, ddof=1, dtype=np.float64)
        super().__init__(thetas, mu, sd, name=name, info=info)
        self.bank = bank

    def simulate(self, idx: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
        cols = rng.integers(0, self.bank.shape[1], size=n_reps)
        return self.bank[idx, cols].astype(float)


_BANKS: dict = {}


def inventory_bank(regime: str, name: str | None = None, n_reps: int | None = None) -> EmpiricalProblem:
    """Bank of real inventory-simulator replications for experiment E6.

    Uses the same grid, seeds and replication count as ``scripts/build_inventory_tables.py``,
    so the bank means equal the stored tables (on the same NumPy version). Built once
    once, then cached in memory and on disk under
    ``$CS_CACHE_DIR`` (default ``data/cache``). ``CS_BANK_REPS`` overrides the size (tests).
    """
    import os

    from .inventory import DISRUPTED, NOMINAL, build_bank

    params, seed = {"nominal": (NOMINAL, 11), "disrupted": (DISRUPTED, 12)}[regime]
    n_reps = n_reps or int(os.environ.get("CS_BANK_REPS", 2000))
    key = (regime, n_reps)
    if key not in _BANKS:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache = os.environ.get("CS_CACHE_DIR", os.path.join(root, "data", "cache"))
        path = os.path.join(cache, f"bank_{regime}_{n_reps}_seed{seed}.npz")
        if os.path.exists(path):
            data = np.load(path)
            thetas, bank = data["thetas"], data["bank"]
        else:
            thetas, bank = build_bank(params, np.arange(100), np.arange(60), n_reps, seed)
            os.makedirs(cache, exist_ok=True)
            tmp = path + ".tmp.npz"
            np.savez(tmp, thetas=thetas, bank=bank)
            os.replace(tmp, path)
        _BANKS[key] = (thetas, bank)
    thetas, bank = _BANKS[key]
    return EmpiricalProblem(thetas, bank, name=name or f"inventory-{regime}-real",
                            info={"regime": regime, "bank_reps": n_reps})
