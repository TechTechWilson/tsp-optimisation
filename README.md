# TSP Optimisation: Local Search vs Evolutionary Algorithm

Solving the **Travelling Salesman Problem (TSP)** — finding the shortest
closed route that visits a set of cities exactly once and returns to the
start — and comparing two families of metaheuristic:

| Family | Algorithm | Role in this project |
|---|---|---|
| Single-solution local search | **Simulated Annealing** | Chosen single-member algorithm |
| Single-solution local search | Stochastic Hill Climbing | Baseline (gets stuck in local optima) |
| Population-based / evolutionary | **Genetic Algorithm** | Evolutionary algorithm (permutation operators) |

Module: **UFCEL1-15-M — AI for Search and Optimisation**.

---

## What this project does

1. Loads city coordinates from `data/cities.csv`.
2. Finds a near-optimal route with each algorithm.
3. Prints the shortest total distance to the console and writes the best
   route to `results/best_route.txt`.
4. Compares the algorithms over repeated runs using a **statistical
   hypothesis test** (Welch's t-test / Mann–Whitney U).
5. Runs a **scalability study** from 10 to 50 cities, measuring both
   solution quality and run time.

---

## Project structure

```
tsp-optimisation/
├── README.md
├── requirements.txt
├── demo.py                     # Reproduces all results and figures
├── tune_ga.py                 # GA parameter sweep -> results/ga_tuning.csv
├── data/
│   └── cities.csv              # 50 cities (the supplied Blackboard dataset)
├── src/
│   ├── tsp.py                  # Data loading, distance matrix, tour length, plots
│   ├── simulated_annealing.py  # Chosen single-member algorithm
│   ├── tabu_search.py         # Tabu Search (single-member, memory-based)
│   ├── hill_climbing.py        # Baseline local search
│   ├── genetic_algorithm.py    # Evolutionary algorithm + permutation operators
│   └── experiments.py          # Repeated runs, statistics, scalability
├── tests/
│   └── test_tsp.py             # Sanity tests (valid tours, correct distances)
├── notebooks/
│   └── tsp_demonstration.ipynb # Narrative walk-through of every experiment
└── results/                    # Generated figures and result tables
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Reproduce every result and figure:

```bash
python demo.py
```

Run the interactive walk-through:

```bash
jupyter notebook notebooks/tsp_demonstration.ipynb
```

Run the tests:

```bash
pytest
```

---

## The dataset

`data/cities.csv` is the 50-city dataset supplied in the Assessments folder
on Blackboard. It has a `City` label column plus `X` and `Y` coordinate
columns.

The coordinates are **synthetic, not real locations**. They are
pseudo-random values from a uniform distribution on `[0, 100)`, and they
reproduce exactly from NumPy's legacy generator seeded with 42:

```bash
python -c "import numpy as np, pandas as pd; np.random.seed(42); \
print(abs(np.random.rand(50,2)*100 - pd.read_csv('data/cities.csv')[['X','Y']].to_numpy()).max())"
# ~4.8e-09  (floating-point rounding, i.e. an exact match)
```

This matters for the legal/ethical section: the data contains no personal
information, so no data-protection obligations arise, and there is no
third-party copyright to clear.

The loader is deliberately tolerant and also accepts:

* a file with `x`/`y` columns in any case (an `id`/`city` column is ignored), or
* a header-less file whose last two numeric columns are the coordinates.

---

## Key design choices (see the report for full justification)

* **Simulated Annealing** is the single-member algorithm because its
  cooling schedule gives an explicit, tunable balance between exploration
  and exploitation, and it can escape the local optima that trap a plain
  hill climber.
* The **2-opt segment-reversal** neighbourhood is used for the local
  searches; it un-crosses routes and is much stronger on TSP than a plain
  swap.
* The Genetic Algorithm uses **Order Crossover (OX1)** and **inversion
  mutation** because a TSP tour is a *permutation* — the single-point
  crossover from the introductory GA example would create invalid tours
  with duplicated and missing cities.
* Every stochastic algorithm is run multiple times with fixed seeds, and
  differences are checked with a statistical hypothesis test, so results
  are reproducible and conclusions are evidence-based.

---

## Licensing and AI use

All third-party libraries (NumPy, SciPy, pandas, Matplotlib) are released
under permissive BSD-style licences that allow academic and commercial
use. Generative AI was used in line with the module policy to clarify
concepts and improve code quality and writing; all design decisions,
analysis and conclusions are the author's own. See the report for the
full legal/ethical discussion.
