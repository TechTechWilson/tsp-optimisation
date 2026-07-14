"""Reproduce every result and figure from the TSP coursework notebook.

Four algorithms are compared on a 50-city Euclidean TSP using 30 paired
runs per algorithm (run ``r`` uses seed ``42 + r`` for all four, so the
comparison is paired rather than a race between different random draws).

Usage::

    python demo.py          # writes results/ and figures/

The notebook ``TSP_Coursework.ipynb`` is the single source of truth; this
script is a faithful extraction of its experiment loop.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402
from scipy import stats          # noqa: E402

from src.genetic_algorithm import genetic_algorithm   # noqa: E402
from src.hill_climbing import hill_climbing            # noqa: E402
from src.simulated_annealing import simulated_annealing  # noqa: E402
from src.tabu_search import tabu_search                # noqa: E402
from src.tsp import (                                  # noqa: E402
    build_distance_matrix,
    load_cities,
    tour_length,
    write_route,
)

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

RANDOM_SEED = 42
N_RUNS = 30

ORDER = ["Hill Climbing", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
PALETTE = dict(zip(ORDER, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))

ALGORITHMS = {
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Tabu Search": tabu_search,
    "Genetic Algorithm": genetic_algorithm,
}

# --------------------------------------------------------------------------- #
# 1. Load the problem
# --------------------------------------------------------------------------- #
coords = load_cities("data/cities.csv")
D = build_distance_matrix(coords)
print(f"Loaded {len(coords)} cities.\n")

# --------------------------------------------------------------------------- #
# 2. Paired experiment: 4 algorithms x 30 runs
# --------------------------------------------------------------------------- #
rows, curves, tours, best_overall = [], {}, {}, (None, float("inf"))

for name, fn in ALGORITHMS.items():
    print(f"{name:<22}", end=" ", flush=True)
    for run in range(N_RUNS):
        route, length, secs, hist = fn(D, seed=RANDOM_SEED + run)
        assert sorted(route) == list(range(len(coords))), "invalid tour!"
        rows.append(
            {"algorithm": name, "run": run, "distance": length, "time": secs}
        )
        if run == 0:
            curves[name], tours[name] = hist, route
        if length < best_overall[1]:
            best_overall = (route, length)
    print("done")

df = pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 3. Summary table
# --------------------------------------------------------------------------- #
summary = (
    df.groupby("algorithm")
    .agg(
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        best_distance=("distance", "min"),
        worst_distance=("distance", "max"),
        mean_time=("time", "mean"),
    )
    .round(3)
    .loc[ORDER]
)
print("\nSummary (30 runs, 50 cities)")
print(summary.to_string())

summary.to_csv("results/summary.csv")
df.to_csv("results/raw_results.csv", index=False)
print("\nWrote results/summary.csv  results/raw_results.csv")


# --------------------------------------------------------------------------- #
# 4. Hypothesis tests  (all six pairwise)
# --------------------------------------------------------------------------- #
def _get(name):
    return df[df.algorithm == name].distance.values


pairs = [
    ("Hill Climbing", "Simulated Annealing"),
    ("Hill Climbing", "Tabu Search"),
    ("Hill Climbing", "Genetic Algorithm"),
    ("Simulated Annealing", "Tabu Search"),
    ("Simulated Annealing", "Genetic Algorithm"),
    ("Tabu Search", "Genetic Algorithm"),
]

tests = []
for a, b in pairs:
    x, y = _get(a), _get(b)
    t_stat, p = stats.ttest_ind(x, y, equal_var=False)
    d = (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    tests.append(
        {
            "comparison": f"{a} vs {b}",
            "t": round(t_stat, 3),
            "p_value": f"{p:.2e}",
            "cohens_d": round(d, 2),
            "reject_H0_at_0.05": "Yes" if p < 0.05 else "No",
        }
    )

hyp_df = pd.DataFrame(tests)
hyp_df.to_csv("results/hypothesis_tests.csv", index=False)
print("Wrote results/hypothesis_tests.csv")

# --------------------------------------------------------------------------- #
# 5. Best route  (external file — assessment requirement)
# --------------------------------------------------------------------------- #
best_route, best_len = best_overall
write_route(best_route, D, "results/best_route.txt")
print(f"Wrote results/best_route.txt  (length = {best_len:.4f})")

# --------------------------------------------------------------------------- #
# 6. Figures
# --------------------------------------------------------------------------- #
# 6a. Boxplot: tour length distribution
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df, x="algorithm", y="distance", order=ORDER,
    hue="algorithm", palette=PALETTE, legend=False,
)
sns.stripplot(
    data=df, x="algorithm", y="distance", order=ORDER,
    color="0.25", size=3, alpha=0.6,
)
plt.title("Tour length across 30 runs (lower is better)")
plt.ylabel("Tour length")
plt.xlabel("")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig("figures/boxplot_quality.png", dpi=130)
plt.close()

# 6b. Boxplot: runtime
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df, x="algorithm", y="time", order=ORDER,
    hue="algorithm", palette=PALETTE, legend=False,
)
plt.title("Runtime across 30 runs")
plt.ylabel("Seconds")
plt.xlabel("")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig("figures/boxplot_runtime.png", dpi=130)
plt.close()

# 6c. Convergence (run 0)
plt.figure(figsize=(9, 5))
for name in ORDER:
    h = curves[name]
    plt.plot(
        np.linspace(0, 100, len(h)), h,
        label=name, color=PALETTE[name], lw=1.8,
    )
plt.xlabel("Search progress (% of budget)")
plt.ylabel("Best tour found so far")
plt.title("Convergence behaviour (run 0)")
plt.ylim(540, 900)
plt.legend()
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=130)
plt.close()

# 6d. Best-route plot for each algorithm (run 0)
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, name in zip(axes.ravel(), ORDER):
    r = tours[name] + [tours[name][0]]
    pts = coords[r]
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=4, lw=1.2, color=PALETTE[name])
    ax.set_title(f"{name}\n{tour_length(tours[name], D):.2f}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("Best tour from each algorithm (run 0)")
plt.tight_layout()
plt.savefig("figures/best_routes.png", dpi=130)
plt.close()

# 6e. Quality against cost
agg = df.groupby("algorithm").agg(
    m=("distance", "mean"), s=("distance", "std"), t=("time", "mean")
).loc[ORDER]
plt.figure(figsize=(8, 5.5))
for name in ORDER:
    plt.errorbar(
        agg.loc[name, "t"], agg.loc[name, "m"], yerr=agg.loc[name, "s"],
        fmt="o", ms=11, capsize=5, color=PALETTE[name],
    )
    plt.annotate(
        name, (agg.loc[name, "t"], agg.loc[name, "m"]),
        textcoords="offset points", xytext=(10, 8), fontsize=9,
    )
plt.xlabel("Mean runtime (s)")
plt.ylabel("Mean tour length (± 1 SD)")
plt.title("Quality against cost")
plt.tight_layout()
plt.savefig("figures/quality_vs_cost.png", dpi=130)
plt.close()

print("Wrote figures/  (5 files)")

# --------------------------------------------------------------------------- #
# 7. GA hyperparameter sweep  (re-run with notebook's GA config)
# --------------------------------------------------------------------------- #
print("\nGA parameter sweep ...")
tune_rows = []
for ts in [2, 5, 7, 9]:
    for mr in [0.05, 0.2, 0.5]:
        lengths = []
        for run in range(5):
            _, length, _, _ = genetic_algorithm(
                D, seed=RANDOM_SEED + run,
                pop_size=100, generations=260,
                mutation_rate=mr, tournament_k=ts, elitism=5,
            )
            lengths.append(length)
        tune_rows.append(
            {
                "tournament_size": ts,
                "mutation_rate": mr,
                "mean_distance": round(np.mean(lengths), 3),
                "std_distance": round(np.std(lengths), 3),
            }
        )
        print(
            f"  k={ts}  mut={mr:.2f}  "
            f"mean={np.mean(lengths):.2f}  std={np.std(lengths):.2f}"
        )

pd.DataFrame(tune_rows).to_csv("results/ga_tuning.csv", index=False)
print("Wrote results/ga_tuning.csv")

print("\nDone. All results in results/  All figures in figures/")
