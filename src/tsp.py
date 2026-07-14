"""Core Travelling Salesman Problem (TSP) utilities.

This module provides the problem-independent building blocks that every
search algorithm in this project relies on:

* loading city data from a CSV file,
* pre-computing an (n x n) Euclidean distance matrix for fast lookups,
* evaluating the total length of a closed tour,
* generating random tours and the 2-opt neighbourhood operator, and
* plotting tours and convergence curves.

The 2-opt delta formula ``D[a,c] + D[b,d] - D[a,b] - D[c,d]`` evaluates
a candidate move in constant time (four table look-ups). All four search
algorithms use this same operator, which is the central methodological
claim of the project.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# A "tour" is represented throughout the project as a list of integer city
# indices, e.g. [0, 4, 2, 1, 3]. The tour is *closed*: after the last city
# the salesman returns to the first city.
Tour = list


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_cities(csv_path: str | Path) -> np.ndarray:
    """Load city coordinates from a CSV file.

    The loader is deliberately tolerant so that it works with the
    ``cities.csv`` provided on Blackboard *and* with common variants:

    * A file with ``x`` and ``y`` columns (any extra ``id``/``city``
      column is ignored).
    * A header-less file whose last two numeric columns are treated as
      the x and y coordinates.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A ``(n, 2)`` NumPy array of float coordinates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If fewer than two coordinate columns can be found.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"City file not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    # Prefer explicitly named columns if present (case-insensitive).
    lowered = {c.lower(): c for c in frame.columns}
    if "x" in lowered and "y" in lowered:
        coords = frame[[lowered["x"], lowered["y"]]].to_numpy(dtype=float)
    else:
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError(
                "Could not find two numeric coordinate columns in the CSV."
            )
        coords = numeric.iloc[:, -2:].to_numpy(dtype=float)

    return coords


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pre-compute the symmetric Euclidean distance matrix.

    Computing all pairwise distances once and then using cheap array
    look-ups inside the search loops is far faster than recomputing
    distances on every iteration.  The double-loop with ``math.sqrt``
    reproduces the notebook's computation exactly.

    Args:
        coords: ``(n, 2)`` array of city coordinates.

    Returns:
        An ``(n, n)`` array where entry ``[i, j]`` is the distance
        between city ``i`` and city ``j``.
    """
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            D[i, j] = d
            D[j, i] = d
    return D


# --------------------------------------------------------------------------- #
# Objective function (fitness)
# --------------------------------------------------------------------------- #
def tour_length(tour: Tour, dist: np.ndarray) -> float:
    """Return the total length of a *closed* tour.

    This is the objective function the whole project minimises: the sum
    of the distances between consecutive cities, plus the distance from
    the final city back to the start.  The explicit Python loop matches
    the notebook's summation order exactly, which matters for
    deterministic reproducibility of the GA.

    Args:
        tour: A permutation of city indices.
        dist: The pre-computed distance matrix.

    Returns:
        The total Euclidean distance of the round trip.
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist[tour[i], tour[(i + 1) % n]]
    return total


def fitness(tour: Tour, dist: np.ndarray) -> float:
    """Convert tour length into a 'higher-is-better' fitness value.

    The lectures frame selection in terms of *maximising* utility, but
    TSP is a *minimisation* problem. Selection schemes such as roulette
    wheel therefore need a value that grows as the tour gets shorter.
    Using the reciprocal of length achieves exactly that.

    Args:
        tour: A permutation of city indices.
        dist: The pre-computed distance matrix.

    Returns:
        ``1 / length`` (a small positive number; larger = better).
    """
    return 1.0 / tour_length(tour, dist)


# --------------------------------------------------------------------------- #
# Tour construction and neighbourhood moves
# --------------------------------------------------------------------------- #
def random_tour(n_cities: int, rng: random.Random | None = None) -> Tour:
    """Generate a random starting tour (a random permutation).

    Args:
        n_cities: Number of cities in the problem.
        rng: Optional seeded ``random.Random`` for reproducibility.

    Returns:
        A shuffled list of city indices ``0 .. n_cities - 1``.
    """
    rng = rng or random
    tour = list(range(n_cities))
    rng.shuffle(tour)
    return tour


def swap_move(tour: Tour, rng: random.Random | None = None) -> Tour:
    """Neighbourhood move: swap two randomly chosen cities.

    This is the simple operator used in the Week 7/8 lecture worked
    examples. It makes a small, local change to the current tour.

    Args:
        tour: The current tour.
        rng: Optional seeded RNG.

    Returns:
        A *new* tour with two positions exchanged (the input is left
        untouched).
    """
    rng = rng or random
    new_tour = tour[:]
    i, j = rng.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def two_opt_move(tour: Tour, rng: random.Random | None = None) -> Tour:
    """Neighbourhood move: reverse a random segment (a 2-opt move).

    A 2-opt move removes two edges from the tour and reconnects the
    path the other way round, which is equivalent to reversing the
    cities between two cut points. For TSP this is a much stronger
    operator than a plain swap because it un-crosses routes, so it is
    used as the default neighbourhood for the local-search algorithms.

    Args:
        tour: The current tour.
        rng: Optional seeded RNG.

    Returns:
        A *new* tour with one segment reversed.
    """
    rng = rng or random
    new_tour = tour[:]
    i, j = sorted(rng.sample(range(len(tour)), 2))
    new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
    return new_tour


# --------------------------------------------------------------------------- #
# Constant-time 2-opt evaluation  (the shared neighbourhood operator)
# --------------------------------------------------------------------------- #
def two_opt_delta(
    route: Tour, i: int, j: int, D: np.ndarray,
) -> float:
    """Cost change from reversing ``route[i:j+1]``.  O(1) — four look-ups.

    The central methodological claim of the project is that all four
    algorithms use this identical operator, so any difference in solution
    quality is attributable to search strategy rather than to one
    algorithm being handed a better move.

    Args:
        route: Current tour (a permutation of city indices).
        i: Start index of the segment to reverse.
        j: End index (inclusive) of the segment to reverse.
        D: ``(n, n)`` distance matrix.

    Returns:
        ``D[a,c] + D[b,d] - D[a,b] - D[c,d]`` — negative when the move
        shortens the tour.
    """
    n = len(route)
    a, b = route[i - 1], route[i]
    c, d = route[j], route[(j + 1) % n]
    if a == c or b == d:
        return 0.0
    return (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])


def two_opt_apply(route: Tour, i: int, j: int) -> Tour:
    """Return a new tour with the segment ``route[i:j+1]`` reversed.

    Args:
        route: Current tour.
        i: Start index of the segment.
        j: End index (inclusive).

    Returns:
        A new list; the input is not modified.
    """
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def two_opt_moves(n: int) -> list[tuple[int, int]]:
    """Every valid ``(i, j)`` 2-opt move for a tour of ``n`` cities.

    Args:
        n: Number of cities.

    Returns:
        A list of ``(i, j)`` pairs with ``1 <= i < j < n``.
    """
    return [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]


def random_route(n_cities: int, rng: np.random.Generator) -> Tour:
    """Generate a random permutation using a NumPy generator.

    This is the route-initialisation helper used by the notebook; it
    accepts a ``numpy.random.Generator`` so that the same seed sequence
    reproduces the notebook's results exactly.

    Args:
        n_cities: Number of cities.
        rng: A seeded ``numpy.random.Generator``.

    Returns:
        A shuffled list of city indices ``0 .. n_cities - 1``.
    """
    route = list(range(n_cities))
    rng.shuffle(route)
    return route


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_tour(
    tour: Tour,
    coords: np.ndarray,
    dist: np.ndarray,
    title: str = "Best tour",
    save_path: str | Path | None = None,
) -> None:
    """Plot a tour as a closed loop over the city coordinates.

    Args:
        tour: The tour to draw.
        coords: ``(n, 2)`` city coordinates.
        dist: Distance matrix (used to annotate the length).
        title: Plot title.
        save_path: If given, the figure is written to this path.
    """
    ordered = coords[np.asarray(tour + [tour[0]])]  # Close the loop.
    length = tour_length(tour, dist)

    plt.figure(figsize=(7, 6))
    plt.plot(ordered[:, 0], ordered[:, 1], "-o", linewidth=1.2, markersize=5)
    plt.scatter(
        coords[tour[0], 0], coords[tour[0], 1],
        c="red", s=90, zorder=5, label="Start / end",
    )
    plt.title(f"{title}\nTotal distance = {length:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    plt.close()


def plot_convergence(
    histories: dict[str, list[float]],
    title: str = "Convergence",
    save_path: str | Path | None = None,
) -> None:
    """Plot best-distance-so-far against iteration for one or more runs.

    Args:
        histories: Mapping of label -> list of best lengths per
            iteration/generation.
        title: Plot title.
        save_path: If given, the figure is written to this path.
    """
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        plt.plot(history, label=label, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Iteration / generation")
    plt.ylabel("Best tour length so far")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    plt.close()


def write_route(tour: Tour, dist: np.ndarray, path: str | Path) -> None:
    """Write the best route and its length to an external text file.

    The assessment brief asks for the sequence of cities for the best
    route to be recorded in an external file; this helper does that.

    Args:
        tour: The best tour found.
        dist: Distance matrix (for the recorded length).
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"Total distance: {tour_length(tour, dist):.4f}\n")
        handle.write("Route (city indices):\n")
        handle.write(" -> ".join(str(c) for c in tour))
        handle.write(f" -> {tour[0]}\n")  # Show the return to start.
