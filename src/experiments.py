"""Experiment harness for comparing TSP search algorithms.

This module turns the individual algorithms into a fair, repeatable
experiment. It provides:

* ``run_repeated`` - run a stochastic algorithm many times and collect
  the best length and wall-clock time of each run,
* ``statistical_comparison`` - test whether the difference in solution
  quality between two algorithms is statistically significant (the
  brief explicitly asks for a hypothesis test such as a t-test), and
* ``scalability_study`` - measure how solution quality and run time grow
  as the number of cities increases from 10 up to 50.

Running a randomised algorithm only once is not enough to draw
conclusions, because a single run can be lucky or unlucky. Every helper
here therefore repeats each algorithm with different seeds and reports
the mean and spread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RunResult:
    """Container for the outcome of repeated runs of one algorithm.

    Attributes:
        name: Human-readable algorithm name.
        lengths: Best tour length from each repeat.
        times: Wall-clock seconds taken by each repeat.
        best_tour: The single best tour found across all repeats.
        best_history: Convergence history of that best run.
    """

    name: str
    lengths: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    best_tour: list[int] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)

    @property
    def mean_length(self) -> float:
        """Mean best length across repeats."""
        return float(np.mean(self.lengths))

    @property
    def std_length(self) -> float:
        """Standard deviation of best length (a robustness measure)."""
        return float(np.std(self.lengths))

    @property
    def mean_time(self) -> float:
        """Mean wall-clock seconds across repeats."""
        return float(np.mean(self.times))


def run_repeated(
    name: str,
    algorithm,
    dist: np.ndarray,
    n_runs: int = 10,
    base_seed: int = 0,
    **kwargs,
) -> RunResult:
    """Run a stochastic algorithm ``n_runs`` times and gather statistics.

    Args:
        name: Label for the algorithm (used in tables and plots).
        algorithm: A callable returning ``(tour, length, history)`` and
            accepting a ``seed`` keyword argument.
        dist: Distance matrix for the instance.
        n_runs: Number of independent repeats.
        base_seed: Seeds used are ``base_seed + run_index`` so the whole
            study is reproducible.
        **kwargs: Extra hyperparameters forwarded to ``algorithm``.

    Returns:
        A populated :class:`RunResult`.
    """
    result = RunResult(name=name)
    best_overall = float("inf")

    for run in range(n_runs):
        start = time.perf_counter()
        tour, length, history = algorithm(
            dist, seed=base_seed + run, **kwargs
        )
        elapsed = time.perf_counter() - start

        result.lengths.append(length)
        result.times.append(elapsed)
        if length < best_overall:
            best_overall = length
            result.best_tour = tour
            result.best_history = history

    return result


def statistical_comparison(
    result_a: RunResult,
    result_b: RunResult,
    alpha: float = 0.05,
) -> dict[str, float | str | bool]:
    """Test whether two algorithms differ significantly in tour length.

    The procedure is:

    1. Check each sample for normality with the Shapiro-Wilk test.
    2. If both look normal, run Welch's independent two-sample t-test
       (Welch's variant does not assume equal variances).
    3. Otherwise fall back to the non-parametric Mann-Whitney U test.

    Args:
        result_a: Results for the first algorithm.
        result_b: Results for the second algorithm.
        alpha: Significance level (default 0.05).

    Returns:
        A dictionary summarising the test used, the p-value, and a
        plain-English conclusion.
    """
    a = np.asarray(result_a.lengths)
    b = np.asarray(result_b.lengths)

    # Shapiro-Wilk needs at least three samples to be meaningful.
    normal_a = stats.shapiro(a).pvalue > alpha if len(a) >= 3 else False
    normal_b = stats.shapiro(b).pvalue > alpha if len(b) >= 3 else False

    if normal_a and normal_b:
        test_name = "Welch's t-test"
        _, p_value = stats.ttest_ind(a, b, equal_var=False)
    else:
        test_name = "Mann-Whitney U test"
        _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")

    significant = bool(p_value < alpha)
    better = result_a.name if a.mean() < b.mean() else result_b.name
    conclusion = (
        f"The difference is statistically significant (p < {alpha}); "
        f"{better} produces shorter tours on average."
        if significant
        else f"No statistically significant difference (p >= {alpha})."
    )

    return {
        "test": test_name,
        "p_value": float(p_value),
        "significant": significant,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "conclusion": conclusion,
    }


def scalability_study(
    coords: np.ndarray,
    algorithms: dict[str, tuple],
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
    n_runs: int = 5,
) -> pd.DataFrame:
    """Measure quality and run time as the problem size grows.

    For each requested size the study uses the first ``size`` cities of
    the dataset, then runs every algorithm ``n_runs`` times.

    Args:
        coords: Full ``(n, 2)`` array of city coordinates.
        algorithms: Mapping ``name -> (callable, kwargs_dict)``.
        sizes: City counts to test.
        n_runs: Repeats per algorithm per size.

    Returns:
        A tidy :class:`pandas.DataFrame` with one row per
        (size, algorithm) combination and columns for mean/std length
        and mean time.
    """
    from .tsp import build_distance_matrix

    records = []
    for size in sizes:
        subset = coords[:size]
        dist = build_distance_matrix(subset)
        for name, (algorithm, kwargs) in algorithms.items():
            result = run_repeated(
                name, algorithm, dist, n_runs=n_runs, **kwargs
            )
            records.append(
                {
                    "cities": size,
                    "algorithm": name,
                    "mean_length": result.mean_length,
                    "std_length": result.std_length,
                    "mean_time_s": result.mean_time,
                }
            )

    return pd.DataFrame.from_records(records)
