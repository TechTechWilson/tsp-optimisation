"""End-to-end demonstration: run, compare and visualise all algorithms.

This script reproduces every result and figure referenced in the report.
Run it from the project root with ``python demo.py``. All outputs are
written to the ``results/`` directory.

Four algorithms are compared:

* Hill Climbing       - single-member baseline (improving moves only)
* Simulated Annealing - single-member, escapes local optima probabilistically
* Tabu Search         - single-member, escapes local optima using memory
* Genetic Algorithm   - population-based evolutionary algorithm

All three single-member algorithms use the same 2-opt neighbourhood, so a
difference between them is attributable to the search strategy rather than
to the move operator.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.experiments import (  # noqa: E402
    run_repeated,
    scalability_study,
    statistical_comparison,
)
from src.genetic_algorithm import genetic_algorithm  # noqa: E402
from src.hill_climbing import hill_climbing  # noqa: E402
from src.simulated_annealing import simulated_annealing  # noqa: E402
from src.tabu_search import tabu_search  # noqa: E402
from src.tsp import (  # noqa: E402
    build_distance_matrix,
    load_cities,
    plot_convergence,
    plot_tour,
    write_route,
)

# Repeats for the head-to-head comparison. 30 is standard practice for
# benchmarking stochastic algorithms.
N_RUNS = 30
SCALE_RUNS = 3        # Repeats per size in the scalability study.

# --------------------------------------------------------------------------- #
# 1. Load the problem
# --------------------------------------------------------------------------- #
coords = load_cities("data/cities.csv")
dist = build_distance_matrix(coords)
print(f"Loaded {len(coords)} cities.\n")

# --------------------------------------------------------------------------- #
# 2. Single demonstration run of each algorithm on all 50 cities
# --------------------------------------------------------------------------- #
print("Single-run demonstration (50 cities)")
print("-" * 44)

t0 = time.perf_counter()
hc_tour, hc_len, hc_hist = hill_climbing(dist, seed=0)
print(f"Hill Climbing       : {hc_len:8.2f}  ({time.perf_counter()-t0:.2f}s)")

t0 = time.perf_counter()
sa_tour, sa_len, sa_hist = simulated_annealing(dist, seed=0)
print(f"Simulated Annealing : {sa_len:8.2f}  ({time.perf_counter()-t0:.2f}s)")

t0 = time.perf_counter()
ts_tour, ts_len, ts_hist = tabu_search(dist, seed=0)
print(f"Tabu Search         : {ts_len:8.2f}  ({time.perf_counter()-t0:.2f}s)")

t0 = time.perf_counter()
ga_tour, ga_len, ga_hist = genetic_algorithm(dist, seed=0)
print(f"Genetic Algorithm   : {ga_len:8.2f}  ({time.perf_counter()-t0:.2f}s)")

# Record the best overall route to an external file (brief requirement).
best_len, best_tour = min(
    [(hc_len, hc_tour), (sa_len, sa_tour), (ts_len, ts_tour), (ga_len, ga_tour)],
    key=lambda pair: pair[0],
)
print(f"\nShortest distance found: {best_len:.4f}")
write_route(best_tour, dist, "results/best_route.txt")

# Route plots.
plot_tour(sa_tour, coords, dist, "Simulated Annealing best tour",
          "results/route_sa.png")
plot_tour(ga_tour, coords, dist, "Genetic Algorithm best tour",
          "results/route_ga.png")
plot_tour(ts_tour, coords, dist, "Tabu Search best tour",
          "results/route_tabu.png")

# Convergence comparison.
plot_convergence(
    {
        "Hill Climbing": hc_hist,
        "Simulated Annealing": sa_hist,
        "Tabu Search": ts_hist,
        "Genetic Algorithm": ga_hist,
    },
    title="Convergence on 50-city TSP",
    save_path="results/convergence.png",
)

# --------------------------------------------------------------------------- #
# 3. Repeated runs + statistical hypothesis tests (50 cities)
# --------------------------------------------------------------------------- #
print("\nRepeated runs for statistical comparison")
print("-" * 44)
hc_res = run_repeated("Hill Climbing", hill_climbing, dist, n_runs=N_RUNS)
sa_res = run_repeated("Simulated Annealing", simulated_annealing, dist,
                      n_runs=N_RUNS)
ts_res = run_repeated("Tabu Search", tabu_search, dist, n_runs=N_RUNS)
ga_res = run_repeated("Genetic Algorithm", genetic_algorithm, dist,
                      n_runs=N_RUNS)

for res in (hc_res, sa_res, ts_res, ga_res):
    print(f"{res.name:20s}: mean={res.mean_length:8.2f}  "
          f"std={res.std_length:6.2f}  mean_time={res.mean_time:.3f}s")

# Primary test required by the brief: single-member vs evolutionary.
print("\nHypothesis test 1 - Simulated Annealing vs Genetic Algorithm")
test_sa_ga = statistical_comparison(sa_res, ga_res)
print(f"  {test_sa_ga['test']}: p = {test_sa_ga['p_value']:.4g}")
print(f"  {test_sa_ga['conclusion']}")

# Secondary test: the two memory-based single-member algorithms.
print("\nHypothesis test 2 - Simulated Annealing vs Tabu Search")
test_sa_ts = statistical_comparison(sa_res, ts_res)
print(f"  {test_sa_ts['test']}: p = {test_sa_ts['p_value']:.4g}")
print(f"  {test_sa_ts['conclusion']}")

# Box plot of solution quality across repeats.
plt.figure(figsize=(8, 5))
plt.boxplot(
    [hc_res.lengths, sa_res.lengths, ts_res.lengths, ga_res.lengths],
    tick_labels=["Hill\nClimbing", "Simulated\nAnnealing", "Tabu\nSearch",
                 "Genetic\nAlgorithm"],
)
plt.ylabel("Best tour length")
plt.title(f"Distribution of best tour over {N_RUNS} runs (50 cities)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/boxplot_quality.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------- #
# 4. Scalability study (10..50 cities)
# --------------------------------------------------------------------------- #
print("\nScalability study (10-50 cities)")
print("-" * 44)
algorithms = {
    "Hill Climbing": (hill_climbing, {}),
    "Simulated Annealing": (simulated_annealing, {"cooling_rate": 0.99}),
    "Tabu Search": (tabu_search, {"max_iterations": 1000}),
    "Genetic Algorithm": (genetic_algorithm, {"generations": 250}),
}
table = scalability_study(coords, algorithms, n_runs=SCALE_RUNS)
table.to_csv("results/scalability.csv", index=False)
print(table.to_string(index=False))

# Quality vs size.
plt.figure(figsize=(8, 5))
for name, group in table.groupby("algorithm"):
    plt.plot(group["cities"], group["mean_length"], "-o", label=name)
plt.xlabel("Number of cities")
plt.ylabel("Mean best tour length")
plt.title("Solution quality vs problem size")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/scalability_quality.png", dpi=130)
plt.close()

# Run time vs size.
plt.figure(figsize=(8, 5))
for name, group in table.groupby("algorithm"):
    plt.plot(group["cities"], group["mean_time_s"], "-o", label=name)
plt.xlabel("Number of cities")
plt.ylabel("Mean run time (s)")
plt.title("Computational cost vs problem size")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/scalability_time.png", dpi=130)
plt.close()

print("\nAll figures and result files written to results/.")
