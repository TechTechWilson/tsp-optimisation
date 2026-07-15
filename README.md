# Travelling Salesman Problem: Four Search Strategies, One Neighbourhood

A comparative study of four metaheuristic search algorithms on a 50-city
Euclidean instance of the Travelling Salesman Problem (TSP).

| Family | Algorithm | Role |
|---|---|---|
| Single-solution | **Hill Climbing** with random restarts | Baseline — no escape mechanism beyond restarting |
| Single-solution | **Simulated Annealing** | Probabilistic escape from local optima |
| Single-solution | **Tabu Search** | Memory-based escape from local optima |
| Population-based | **Genetic Algorithm** | Evolutionary search with permutation operators |

Submitted for **UFCEL1-15-M — AI for Search and Optimisation**, UWE Bristol.

---

## The central methodological claim

All four algorithms search the **same 2-opt neighbourhood**. A 2-opt move
reverses a contiguous segment of the tour, and its effect on tour length is
evaluated in constant time by the formula

$$\Delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]$$

— four table look-ups, no tour re-scan. The GA's inversion mutation *is* a
2-opt move applied blindly, which keeps the GA inside the same landscape as
the other three. Any difference in performance is therefore attributable to
**search strategy**, not to one algorithm being handed a better move
operator.

Every algorithm is written from first principles. No third-party TSP
solver, optimisation library, or metaheuristic framework is used. NumPy
stores arrays and draws random numbers; SciPy runs the statistical tests;
pandas and matplotlib tabulate and plot results.

## Video demonstration via google drive 

A walkthrough of the implementation and results:
**[Watch the demonstration](https://drive.google.com/drive/folders/18KYLXNaomrJ-c0rUKuJhoeo1Spv9BpFS?usp=share_link)**

---

## Results

30 independent runs per algorithm on the 50-city instance. Run *r* uses
seed *r* for all four algorithms, so the comparison is paired.

| Algorithm | Mean | SD | Best | Worst | Mean time |
|---|---|---|---|---|---|
| Hill Climbing | 566.34 | 5.83 | 559.87 | 578.68 | 0.41 s |
| Simulated Annealing | 571.41 | 5.64 | 559.87 | 585.82 | 1.37 s |
| Tabu Search | 568.71 | 6.21 | 559.87 | 580.23 | 0.16 s |
| Genetic Algorithm | 594.07 | 18.63 | 559.87 | 634.22 | 0.50 s |

**Key findings:**

- All four algorithms reach the same best tour (559.87).
- The three local-search methods are statistically indistinguishable from
  each other (HC vs SA: *p* = 1.12 × 10⁻³; HC vs Tabu: *p* = 0.13).
- Every local-search method beats the GA with a large effect size
  (*p* < 10⁻⁶, Cohen's *d* > 1.5 in all cases).
- Tabu Search is the best value for money: it matches Hill Climbing quality
  at roughly 40% of its runtime.

---

## Repository structure

```
.
├── data/
│   └── cities.csv                       # 50 city coordinates
├── src/
│   ├── tsp.py                           # Data loading, distance matrix, 2-opt delta
│   ├── hill_climbing.py                 # Steepest-ascent with random restarts
│   ├── simulated_annealing.py           # Metropolis acceptance, geometric cooling
│   ├── tabu_search.py                   # Best-of-neighbourhood with short-term memory
│   ├── genetic_algorithm.py             # Tournament selection, OX1, inversion mutation
│   └── experiments.py                   # Helper classes for repeated runs
├── tests/
│   └── test_tsp.py                      # 16 tests: validity, determinism, delta correctness
├── notebooks/
│   └── TSP_Coursework.ipynb             # Narrated walkthrough — single source of truth
├── results/                             # Generated CSV tables and best_route.txt
├── figures/                             # Generated plots
├── demo.py                              # One-shot reproduction of all results & figures
├── tune_ga.py                           # GA hyperparameter sweep
├── requirements.txt
└── README.md
```

---

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/TechTechWilson/tsp-optimisation.git
cd tsp-optimisation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Reproduce every result and figure

```bash
python demo.py
```

This runs all four algorithms 30 times each on the 50-city instance,
performs the six pairwise hypothesis tests, runs the GA hyperparameter
sweep, and writes all CSVs and figures. It takes about two minutes on a
standard laptop. All randomness is seeded from `RANDOM_SEED = 42`, so
results reproduce exactly.

### Run the tests

```bash
pytest tests/ -v
```

16 tests covering: distance matrix properties, tour-length arithmetic,
2-opt delta correctness against full recomputation, tour validity for every
algorithm, improvement over random starts, and determinism (same seed →
same tour).

### Code quality

```bash
flake8 src tests --max-line-length 88
```

### Notebook

The notebook `notebooks/TSP_Coursework.ipynb` walks through the same
experiments with the reasoning made explicit at each step. It is the single
source of truth for this project; `src/` and `demo.py` are faithful
extractions of it.

---

## Data

`data/cities.csv` is the instance supplied with the assessment. It contains
50 rows with columns `City, X, Y`. The coordinates are synthetic, drawn
from a uniform distribution on [0, 100]² with a fixed seed. The file
contains no personal data and no third-party copyrighted content. The
licensing and ethical position is discussed in full in the technical report.

---

## Dependencies

| Package | Version | Licence | Used for |
|---|---|---|---|
| numpy | ≥1.26 | BSD-3-Clause | Array storage, random number generation |
| scipy | ≥1.11 | BSD-3-Clause | Welch's t-test, Mann-Whitney U |
| pandas | ≥2.1 | BSD-3-Clause | Tabular data, CSV I/O |
| matplotlib | ≥3.8 | PSF-based | Plotting |
| seaborn | ≥0.12 | BSD-3-Clause | Box plots |
| pytest | ≥7.0 | MIT | Test runner (dev only) |
| flake8 | ≥7.0 | MIT | Linting (dev only) |

All algorithms are original work. No code has been adapted from third-party
TSP solvers, optimisation frameworks, or AI-generated sources.

---

## References

1. Croes, G.A. (1958) 'A method for solving traveling-salesman problems',
   *Operations Research*, 6(6), pp. 791–812.
2. Davis, L. (1985) 'Applying adaptive algorithms to epistatic domains', in
   *Proceedings of the 9th International Joint Conference on Artificial
   Intelligence*, pp. 162–164.
3. Glover, F. (1989) 'Tabu search — Part I', *ORSA Journal on Computing*,
   1(3), pp. 190–206.
4. Johnson, D.S. and McGeoch, L.A. (1997) 'The traveling salesman problem:
   a case study in local optimization', in Aarts, E. and Lenstra, J.K. (eds)
   *Local Search in Combinatorial Optimization*. Chichester: Wiley,
   pp. 215–310.
5. Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P. (1983) 'Optimization by
   simulated annealing', *Science*, 220(4598), pp. 671–680.
