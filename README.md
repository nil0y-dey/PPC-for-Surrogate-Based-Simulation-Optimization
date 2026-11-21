# Certified Stopping for Surrogate-Based Simulation Optimization

Code, data and results for the paper *When Can You Stop? Certified Stopping for Surrogate-Based
Simulation Optimization*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nil0y-dey/PPC-for-Surrogate-Based-Simulation-Optimization/blob/main/notebooks/run_experiments_colab.ipynb)

## The method in brief

After a surrogate-based search returns an incumbent, the certified stopping rule bounds the share of
the policy space that beats the incumbent by more than a margin, with no assumption on the response
surface and no trust in the surrogate:

1. **Verify** the incumbent with fresh replications: an upper confidence bound `U_bar` on its mean cost.
2. **Audit** policies drawn at random, labelling each by a sequential test against `c = U_bar - eps`,
   with an indifference zone of width `delta`.
3. **Bound** the mean label with an anytime-valid betting confidence sequence.

The bound holds at every time simultaneously, so the analyst may stop whenever it falls below a target.
The search, the verification and the audits use independent random-number streams, which makes the
guarantee hold whatever the surrogate did.

## Reproducing the results

**Google Colab (no installation).** Open the notebook with the badge above, choose
**Runtime → Run all**, and allow access to Google Drive. Results are written to
`MyDrive/certified_stopping/`. Completed parts are skipped on later runs, and interrupted parts resume
with identical results. 

**Locally.**

```bash
pip install -r requirements.txt
python -m pytest -q                                  
python scripts/run_experiment.py --all                
python scripts/check_label_validity.py                
python scripts/make_figures.py --results results --paper paper
```

The raw results used in the paper are in `results/`,  `make_figures.py` regenerates tables
and figures.

## Experiments

Each experiment is split into parts of 100 macro-replications; part *k* uses seed *k*.

| Experiment | Question | Parts | Outputs | 
|---|---|---|---|
| E1 | Is the surrogate's own estimate of the remaining improvement reliable? Does the search surrogate recall the policies that still beat the incumbent? | 5 | `tab_diagnosis`, `tab_mechanism`, `fig_selection` | 
| E2 | Is the certified stopping rule valid? How does the bound shrink with the budget? | 5 | `tab_validity`, `fig_budget` |
| E3 | Which ingredients make certification affordable (betting bound, label schedule)? | 2 | `tab_ablation` | 
| E4 | Can surrogate-powered variants beat uniform auditing after a full search? | 2 | `fig_surrogate` | 
| E5 | Does the certificate hold after a supply-disruption regime shift? | 2 | `tab_shift` | 
| E6 | Is it valid with the simulator's real (skewed) outputs instead of normal noise? | 2 | `tab_realnoise` | 
| E7 | Can surrogate-powered variants help after an early-stopped search? | 2 | `tab_aboveres` | 
| Label check | Does each label meet its error bound with real outputs? | | `tab_labelvalidity` |

## Problems and data

* `smooth`, `cliff`, `needle`: synthetic 100 x 100 grids with known costs (`ppc/problems.py`).
* `inventory`: a two-echelon serial supply chain (supplier, warehouse, retailer) with installation
  base-stock levels `(S_w, S_r)` on a 100 x 60 grid, Poisson demand and backorders, in a nominal and
  a disrupted regime (`ppc/inventory.py`).
* `inventory_shift`: search in the nominal regime, certification in the disrupted regime.
* `*_real` (E6): the inventory problems with replications resampled from banks of 2,000 actual
  simulator outputs per policy.
* `*_early` (E7): the search stops after its initial design.

`data/inventory_nominal.npz` and `data/inventory_disrupted.npz` hold the ground-truth mean cost and
standard deviation of every policy, from 2,000 replications each
(`python scripts/build_inventory_tables.py`). 

## Results files

`results/E<k>/E<k>_<config>_all_seed<part>.csv` holds one row per method and macro-replication.
`results/label_validity.csv` holds the label-validity check.

| Column | Meaning |
|---|---|
| `experiment`, `config`, `setting`, `problem`, `surrogate`, `seed` | identifiers of the run |
| `method` | `uniform`, `ppi`, `ppi_1s`, `ppc`, `ppc_1s`, `ppi_1s_refit`, or `model_only` (the surrogate's own estimate) |
| `M_true`, `M_eps` | true improvement mass at margin `eps + delta` (the guaranteed target) and at `eps` |
| `bound_final`, `violated` | final certified bound; whether the bound was ever below `M_true` |
| `b010` ... `b100` | bound after 10%, 25%, 50%, 75% and 100% of the replication budget |
| `reps_to_target` | replications until the bound first fell below `p* = 0.02` (empty if never) |
| `n_audits`, `n_ones`, `P`, `w_max` | audits, audits labelled 1, predicted mass, largest importance weight |
| `bound_cp_fixed_n` | Clopper-Pearson bound for the same audits as if their number had been fixed (not anytime-valid) |
| `gap`, `search_reps`, `u_bar_covers` | optimality gap of the incumbent, replications used by the search, whether `U_bar >= mu(incumbent)` |
| `recall_search`, `recall_random` | recall of the remaining improvers by the search surrogate and by a surrogate fitted to a random design |
| `flag_mass`, `false_alarm_mass`, `lift_missed`, `flags_unevaluated`, `c_below_best_mean` | quantities of Theorem 3 and Corollary 1 |
| `seconds` | wall-clock time of the method |

## Repository layout

```
ppc/                  the package (named after prediction-powered certification)
  certify.py          verification, audit designs and the certificates
  confseq.py          anytime-valid confidence sequences (betting, empirical Bernstein)
  labeling.py         sequential labels (t-test, z-test, distribution-free betting)
  search.py           batch lower-confidence-bound search
  surrogates.py       Gaussian process, extra trees, quadratic regression
  problems.py         test problems and the banks of real simulator outputs
  inventory.py        vectorized two-echelon inventory simulator
  runner.py           one macro-replication: search, verify, audit
  experiments.py      the experiments E1-E7 and the run plan
scripts/              command-line entry points
notebooks/            the Colab notebook
results/              raw results of every experiment
data/                 ground-truth tables
paper/                manuscript, bibliography, figures and tables
tests/                unit and end-to-end tests
```
