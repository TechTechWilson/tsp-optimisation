"""Experimental harness for the TSP algorithm comparison.

Running this module end to end reproduces every number and every figure quoted
in the technical report.

Methodology in brief:

* Both algorithms are stochastic, so a single run tells us nothing. Each
  algorithm is run ``RUNS`` independent times at each problem size, with a
  distinct seed per run, and we report the distribution rather than a single
  best case.
* Scalability is measured by repeating the whole comparison at 10, 20, 30, 40
  and 50 cities, taking the first ``k`` rows of the dataset as the instance.
* The full 50-city results are then subjected to two hypothesis tests: Welch's
  t-test (does not assume equal variances) and the Mann-Whitney U test (does
  not assume normality). Agreement between a parametric and a non-parametric
  test guards against a conclusion that is an artefact of one test's
  assumptions.

Usage:
    python experiments.py
"""

from __future__ import annotations

import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from genetic import genetic_algorithm
from hill_climbing import hill_climb_random_restart
from tsp_core import distance_matrix, load_cities, save_route, tour_length

# Render to file rather than to a screen: the harness runs head-less.
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "cities.csv")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

SIZES = [10, 20, 30, 40, 50]
RUNS = 30
MASTER_SEED = 42

HC_RESTARTS = 20
GA_PARAMS = dict(
    population_size=150,
    generations=400,
    crossover_rate=0.9,
    mutation_rate=0.2,
    tournament_size=5,
    elitism=4,
)


def _timed(function, *args, **kwargs):
    """Call ``function`` and return its result alongside the wall-clock time.

    Args:
        function (Callable): The function to time.
        *args: Positional arguments forwarded to ``function``.
        **kwargs: Keyword arguments forwarded to ``function``.

    Returns:
        tuple: ``(result, elapsed_seconds)``.
    """
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def run_scalability(coords, sizes=None):
    """Run both algorithms ``RUNS`` times at each problem size.

    Args:
        coords (numpy.ndarray): Full ``(50, 2)`` coordinate array.
        sizes (Sequence[int] | None): Problem sizes to evaluate. Defaults to
            the module-level ``SIZES``.

    Returns:
        pandas.DataFrame: One row per (size, algorithm, run) with the tour
        length and the runtime in seconds.
    """
    rows = []
    for size in (SIZES if sizes is None else sizes):
        dist = distance_matrix(coords[:size])
        print(f"--- {size} cities ---", flush=True)

        for run in range(RUNS):
            rng = np.random.default_rng(MASTER_SEED + 1000 * size + run)
            (_, length, _), elapsed = _timed(
                hill_climb_random_restart, dist, HC_RESTARTS, rng
            )
            rows.append(
                {"size": size, "algorithm": "Hill Climbing", "run": run,
                 "distance": length, "time": elapsed}
            )

        for run in range(RUNS):
            rng = np.random.default_rng(MASTER_SEED + 1000 * size + run)
            (_, length, _), elapsed = _timed(
                genetic_algorithm, dist, rng=rng, **GA_PARAMS
            )
            rows.append(
                {"size": size, "algorithm": "Genetic Algorithm", "run": run,
                 "distance": length, "time": elapsed}
            )

    return pd.DataFrame(rows)


def tune_genetic_algorithm(dist, runs=5):
    """Grid-search a handful of GA configurations on the full instance.

    The sweep isolates the two parameters with the largest effect on the
    exploration/exploitation balance: selection pressure (tournament size) and
    mutation rate. Each configuration is run several times because a single
    run of a stochastic algorithm cannot rank configurations reliably.

    Args:
        dist (numpy.ndarray): Distance matrix of the full instance.
        runs (int): Independent runs per configuration.

    Returns:
        pandas.DataFrame: Mean and standard deviation of the tour length for
        each configuration, sorted best first.
    """
    grid = [
        {"tournament_size": 2, "mutation_rate": 0.2},
        {"tournament_size": 5, "mutation_rate": 0.2},
        {"tournament_size": 9, "mutation_rate": 0.2},
        {"tournament_size": 5, "mutation_rate": 0.05},
        {"tournament_size": 5, "mutation_rate": 0.5},
        {"tournament_size": 9, "mutation_rate": 0.5},
    ]

    rows = []
    for config in grid:
        params = dict(GA_PARAMS)
        params.update(config)
        lengths = []
        for run in range(runs):
            rng = np.random.default_rng(MASTER_SEED + run)
            _, length, _ = genetic_algorithm(dist, rng=rng, **params)
            lengths.append(length)
        rows.append(
            {"tournament_size": config["tournament_size"],
             "mutation_rate": config["mutation_rate"],
             "mean_distance": float(np.mean(lengths)),
             "std_distance": float(np.std(lengths, ddof=1))}
        )
        print(f"  k={config['tournament_size']} "
              f"mut={config['mutation_rate']} -> "
              f"{rows[-1]['mean_distance']:.2f}", flush=True)

    return pd.DataFrame(rows).sort_values("mean_distance").reset_index(drop=True)


def hypothesis_tests(frame, size=50):
    """Test whether the two algorithms differ in solution quality and runtime.

    Args:
        frame (pandas.DataFrame): Output of :func:`run_scalability`.
        size (int): Problem size to test on.

    Returns:
        pandas.DataFrame: One row per test with the statistic, the p-value and
        the effect size (Cohen's d for the t-tests).
    """
    subset = frame[frame["size"] == size]
    hc = subset[subset["algorithm"] == "Hill Climbing"]
    ga = subset[subset["algorithm"] == "Genetic Algorithm"]

    def cohens_d(a, b):
        """Standardised mean difference using the pooled standard deviation."""
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                          + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
        return float((a.mean() - b.mean()) / pooled)

    t_stat, t_p = stats.ttest_ind(hc["distance"], ga["distance"],
                                  equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(hc["distance"], ga["distance"],
                                     alternative="two-sided")
    tt_stat, tt_p = stats.ttest_ind(hc["time"], ga["time"], equal_var=False)

    return pd.DataFrame([
        {"test": "Welch t-test (tour length)", "statistic": t_stat,
         "p_value": t_p, "effect_size_d": cohens_d(hc["distance"],
                                                   ga["distance"])},
        {"test": "Mann-Whitney U (tour length)", "statistic": u_stat,
         "p_value": u_p, "effect_size_d": np.nan},
        {"test": "Welch t-test (runtime)", "statistic": tt_stat,
         "p_value": tt_p, "effect_size_d": cohens_d(hc["time"], ga["time"])},
    ])


def make_figures(frame, coords, dist, names):
    """Produce every figure used in the report.

    Args:
        frame (pandas.DataFrame): Output of :func:`run_scalability`.
        coords (numpy.ndarray): Full coordinate array.
        dist (numpy.ndarray): Full distance matrix.
        names (list[str]): City names.

    Returns:
        dict: Mapping of figure name to the tour it depicts, for the two route
        plots. Used by the caller to record the best routes.
    """
    os.makedirs(FIGURES, exist_ok=True)
    palette = {"Hill Climbing": "#1f77b4", "Genetic Algorithm": "#d62728"}

    # Figure 1: solution quality distribution at 50 cities.
    full = frame[frame["size"] == 50]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    data = [full[full["algorithm"] == a]["distance"] for a in palette]
    axes[0].boxplot(data, tick_labels=list(palette))
    axes[0].set_ylabel("Tour length")
    axes[0].set_title(f"Solution quality, 50 cities ({RUNS} runs)")
    axes[0].grid(alpha=0.3)

    times = [full[full["algorithm"] == a]["time"] for a in palette]
    axes[1].boxplot(times, tick_labels=list(palette))
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Runtime, 50 cities ({RUNS} runs)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_quality_runtime_boxplot.png"),
                dpi=150)
    plt.close(fig)

    # Figure 2: scalability of quality and runtime.
    summary = frame.groupby(["size", "algorithm"]).agg(
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        mean_time=("time", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for algorithm, colour in palette.items():
        part = summary[summary["algorithm"] == algorithm]
        axes[0].errorbar(part["size"], part["mean_distance"],
                         yerr=part["std_distance"], marker="o", capsize=4,
                         label=algorithm, color=colour)
        axes[1].plot(part["size"], part["mean_time"], marker="o",
                     label=algorithm, color=colour)
    axes[0].set_xlabel("Number of cities")
    axes[0].set_ylabel("Mean tour length (± 1 SD)")
    axes[0].set_title("Solution quality against problem size")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Number of cities")
    axes[1].set_ylabel("Mean runtime (s, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Computational cost against problem size")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_scalability.png"), dpi=150)
    plt.close(fig)

    # Figure 3: convergence traces on the full instance.
    rng = np.random.default_rng(MASTER_SEED)
    hc_tour, hc_length, hc_history = hill_climb_random_restart(
        dist, HC_RESTARTS, rng
    )
    rng = np.random.default_rng(MASTER_SEED)
    ga_tour, ga_length, ga_history = genetic_algorithm(dist, rng=rng,
                                                       **GA_PARAMS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(hc_history, marker="o", markersize=3,
                 color=palette["Hill Climbing"])
    axes[0].set_xlabel("Accepted 2-opt move")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title(f"Hill Climbing, best restart (final {hc_length:.1f})")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ga_history, color=palette["Genetic Algorithm"])
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best-so-far tour length")
    axes[1].set_title(f"Genetic Algorithm (final {ga_length:.1f})")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig3_convergence.png"), dpi=150)
    plt.close(fig)

    # Figure 4: the best tours themselves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for axis, tour, label, colour in (
        (axes[0], hc_tour, "Hill Climbing", palette["Hill Climbing"]),
        (axes[1], ga_tour, "Genetic Algorithm", palette["Genetic Algorithm"]),
    ):
        closed = list(tour) + [tour[0]]
        axis.plot(coords[closed, 0], coords[closed, 1], "-", color=colour,
                  linewidth=1.2)
        axis.scatter(coords[:, 0], coords[:, 1], s=22, color="black", zorder=3)
        axis.set_title(f"{label}: {tour_length(tour, dist):.1f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_best_routes.png"), dpi=150)
    plt.close(fig)

    return {"hill_climbing": hc_tour, "genetic_algorithm": ga_tour}


def main():
    """Run the full experimental pipeline and write all artefacts to disk."""
    os.makedirs(RESULTS, exist_ok=True)
    names, coords = load_cities(DATA)
    dist = distance_matrix(coords)
    print(f"Loaded {len(names)} cities from {DATA}\n")

    print("Running scalability experiment...")
    raw = run_scalability(coords)
    raw.to_csv(os.path.join(RESULTS, "raw_results.csv"), index=False)

    summary = raw.groupby(["size", "algorithm"]).agg(
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        best_distance=("distance", "min"),
        mean_time=("time", "mean"),
    ).round(3).reset_index()
    summary.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    print("\n" + summary.to_string(index=False) + "\n")

    print("Tuning the genetic algorithm...")
    tuning = tune_genetic_algorithm(dist)
    tuning.round(3).to_csv(os.path.join(RESULTS, "ga_tuning.csv"), index=False)
    print("\n" + tuning.round(2).to_string(index=False) + "\n")

    print("Hypothesis tests on the 50-city instance...")
    tests = hypothesis_tests(raw)
    tests.to_csv(os.path.join(RESULTS, "hypothesis_tests.csv"), index=False)
    print(tests.to_string(index=False) + "\n")

    print("Generating figures...")
    best = make_figures(raw, coords, dist, names)

    for label, tour in best.items():
        path = os.path.join(RESULTS, f"best_route_{label}.txt")
        save_route(path, tour, names, dist)
        print(f"  {label}: {tour_length(tour, dist):.2f} -> {path}")

    overall = min(best.values(), key=lambda t: tour_length(t, dist))
    save_route(os.path.join(RESULTS, "best_route.txt"), overall, names, dist)
    print(f"\nShortest route found overall: {tour_length(overall, dist):.2f}")


if __name__ == "__main__":
    main()
