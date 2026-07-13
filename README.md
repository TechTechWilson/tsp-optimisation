# Travelling Salesman Problem: Hill Climbing vs Genetic Algorithm

A comparative study of a single-solution search algorithm and a population-based
evolutionary algorithm on a 50-city Euclidean instance of the Travelling
Salesman Problem (TSP).

Submitted for **UFCEL1-15-M — AI for Search and Optimisation**, UWE Bristol.

---

## Overview

The TSP asks for the shortest closed tour that visits each of *n* cities exactly
once and returns to the origin. It is NP-hard, so exact methods become
intractable quickly and heuristic search is the practical route to good tours.

This repository implements and compares:

| Algorithm | Family | Key operators |
|---|---|---|
| **Hill Climbing with Random Restarts** | Single-solution local search | Steepest ascent, 2-opt neighbourhood, 20 random restarts |
| **Genetic Algorithm** | Population-based evolutionary search | Tournament selection, order crossover (OX1), inversion mutation, elitism |

Both algorithms are written from first principles. No third-party TSP solver,
optimisation library or metaheuristic framework is used anywhere in this code.
NumPy is used only for array storage and vectorised arithmetic, SciPy only for
the statistical tests, pandas and matplotlib only for tabulating and plotting
results.

Both algorithms search the **same 2-opt move landscape** (the GA's inversion
mutation is a 2-opt move applied at random), so any difference in performance is
attributable to the search strategy rather than to one algorithm being handed a
better move operator.

---

## Repository structure

```
.
├── data/
│   └── cities.csv                  # 50 city coordinates (provided instance)
├── src/
│   ├── tsp_core.py                 # Data loading, distance matrix, tour evaluation, 2-opt
│   ├── hill_climbing.py            # Steepest-ascent hill climber with random restarts
│   ├── genetic.py                  # Genetic algorithm and all its operators
│   └── experiments.py              # Full experimental pipeline
├── notebooks/
│   └── tsp_analysis.ipynb          # Narrated walkthrough of the same experiments
├── results/                        # Generated: CSVs of every run, tests, best route
├── figures/                        # Generated: all figures used in the report
├── requirements.txt
└── README.md
```

---

## Setup

Python 3.10 or newer.

```bash
git clone <your-repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Reproduce every result and figure

```bash
cd src
python experiments.py
```

This runs both algorithms 30 times at each of 10, 20, 30, 40 and 50 cities,
performs the GA hyperparameter sweep, runs the hypothesis tests, writes the
figures and records the best route found. It takes roughly six minutes on a
standard laptop. All randomness is seeded from a single master seed
(`MASTER_SEED = 42` in `experiments.py`), so results are reproducible exactly.

### Outputs

| File | Contents |
|---|---|
| `results/raw_results.csv` | Tour length and runtime for every individual run |
| `results/summary.csv` | Mean, standard deviation, best and mean runtime per size |
| `results/ga_tuning.csv` | Hyperparameter sweep results |
| `results/hypothesis_tests.csv` | Welch t-test and Mann-Whitney U results |
| `results/best_route.txt` | The shortest tour found, as an ordered city sequence |
| `figures/*.png` | Box plots, scalability curves, convergence traces, route plots |

### Solve the instance once and print the tour length

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from tsp_core import load_cities, distance_matrix, save_route
from hill_climbing import hill_climb_random_restart

names, coords = load_cities("data/cities.csv")
dist = distance_matrix(coords)

tour, length, _ = hill_climb_random_restart(
    dist, restarts=20, rng=np.random.default_rng(42)
)
print(f"Shortest route found: {length:.2f}")
save_route("results/best_route.txt", tour, names, dist)
```

### Notebook

`notebooks/tsp_analysis.ipynb` walks through the same experiments with the
reasoning made explicit at each step. Launch with:

```bash
jupyter notebook notebooks/tsp_analysis.ipynb
```

---

## Data

`data/cities.csv` is the instance supplied with the assessment. It contains 50
rows with columns `City, X, Y`. The coordinates are synthetic: they were drawn
from a uniform distribution on `[0, 100]²` with a fixed seed, which is why the
first values reproduce exactly the standard NumPy seed-42 uniform sequence. The
file therefore contains no personal data and no third-party copyrighted content.
The licensing and ethical position is discussed in full in the technical report.

---

## Code standards

* PEP 8 throughout (verified with `flake8`).
* Google-style docstrings on every module, function and class.
* Input validation with explicit exceptions in `tsp_core.load_cities` and
  `genetic.genetic_algorithm`.

To check:

```bash
pip install flake8
flake8 src --max-line-length 88
```

---

## References

1. Alanzi, E. and Menai, M.E.B. (2025) 'Solving the travelling salesman problem
   with machine learning: a review of recent advances and challenges',
   *Artificial Intelligence Review*, 58(9), p. 267.
2. Mamatova, Z. and Abdumajidova, M. (2025) 'The travelling salesman problem:
   mathematical modeling and optimal solutions', *International Journal of
   Artificial Intelligence*, 1(3), pp. 1204–1212.
3. Croes, G.A. (1958) 'A method for solving traveling-salesman problems',
   *Operations Research*, 6(6), pp. 791–812.
4. Davis, L. (1985) 'Applying adaptive algorithms to epistatic domains', in
   *Proceedings of the 9th International Joint Conference on Artificial
   Intelligence*, pp. 162–164.
5. Johnson, D.S. and McGeoch, L.A. (1997) 'The traveling salesman problem: a
   case study in local optimization', in Aarts, E. and Lenstra, J.K. (eds)
   *Local Search in Combinatorial Optimization*. Chichester: Wiley, pp. 215–310.
