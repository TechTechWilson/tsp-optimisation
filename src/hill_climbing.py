"""Single-solution search: steepest-ascent hill climbing with random restarts.

Design choices, all of which are argued in the technical report:

* **Candidate adjustment method.** Steepest ascent. Every move in the
  neighbourhood is evaluated and the single best improving move is taken. This
  is more expensive per iteration than first-improvement, but each 2-opt move
  is scored in constant time via a delta calculation, so a full sweep of an
  ``n = 50`` tour costs only about 1,200 additions.

* **Neighbourhood structure.** 2-opt, which reverses a contiguous segment of
  the tour. For a Euclidean TSP this is the natural operator because it is
  exactly the move that removes a pair of crossing edges. The pairwise-swap
  neighbourhood used by many naive implementations disturbs four edges at once
  and is a much poorer fit.

* **Exploration versus exploitation.** A pure hill climber is pure
  exploitation: it never accepts a worsening move, so it stops at the first
  local optimum it reaches. Exploration is reintroduced at the outer level, by
  restarting from a fresh random tour and keeping the best local optimum found
  across all restarts.
"""

from __future__ import annotations

import numpy as np

from tsp_core import (
    apply_two_opt,
    random_tour,
    tour_length,
    two_opt_delta_matrix,
)


def hill_climb(tour, dist, max_iterations=10_000):
    """Run steepest-ascent hill climbing from a single starting tour.

    Args:
        tour (list[int]): Starting permutation.
        dist (numpy.ndarray): Distance matrix.
        max_iterations (int): Safety cap on improving steps. The search
            normally terminates long before this because it stops as soon as
            no improving 2-opt move exists.

    Returns:
        tuple[list[int], float, list[float]]: The local optimum, its length,
        and the history of tour lengths after each accepted move.
    """
    current = list(tour)
    current_length = tour_length(current, dist)
    history = [current_length]

    for _ in range(max_iterations):
        # Score the entire 2-opt neighbourhood, then take the steepest
        # improving move. If the best available delta is not negative we are
        # sitting in a local optimum and the climb ends.
        deltas = two_opt_delta_matrix(current, dist)
        flat_best = int(np.argmin(deltas))
        best_delta = float(deltas.flat[flat_best])

        if best_delta >= -1e-12:
            break

        i, j = np.unravel_index(flat_best, deltas.shape)
        current = apply_two_opt(current, int(i), int(j))
        current_length += best_delta
        history.append(current_length)

    return current, current_length, history


def hill_climb_random_restart(dist, restarts=20, rng=None):
    """Hill climb repeatedly from random starts and keep the best result.

    Args:
        dist (numpy.ndarray): Distance matrix.
        restarts (int): Number of independent climbs.
        rng (numpy.random.Generator | None): Seeded generator. A fresh default
            generator is created if none is supplied.

    Returns:
        tuple[list[int], float, list[float]]: Best tour found across all
        restarts, its length, and the convergence history of the restart that
        produced it.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = dist.shape[0]
    best_tour = None
    best_length = float("inf")
    best_history = []

    for _ in range(restarts):
        start = random_tour(n, rng)
        tour, length, history = hill_climb(start, dist)
        if length < best_length:
            best_tour, best_length, best_history = tour, length, history

    return best_tour, best_length, best_history
