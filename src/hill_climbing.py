"""Steepest-ascent hill climbing with random restarts.

This is the baseline single-solution search. The climb itself never accepts
a worse solution — it scans the full 2-opt neighbourhood, takes the single
best improving move, and stops when no improving move exists. Random
restarts are its only escape mechanism from a local optimum.

Design choices (all argued in the notebook):

* **Neighbourhood.** 2-opt, evaluated in constant time via
  ``two_opt_delta``, shared across all four algorithms.
* **Restarts.** 20 independent climbs from different random tours. The
  best local optimum found across all restarts is returned.
* **Steepest ascent.** Every move in the neighbourhood is evaluated; the
  steepest improving move is taken. This is more expensive per iteration
  than first-improvement but converges in fewer steps, and the O(1) delta
  evaluation makes the full sweep affordable at ``n = 50``.
"""

from __future__ import annotations

import time

import numpy as np

from .tsp import (
    Tour,
    random_route,
    tour_length,
    two_opt_apply,
    two_opt_delta,
    two_opt_moves,
)


def hill_climbing(
    D: np.ndarray,
    seed: int,
    restarts: int = 20,
    max_iter: int = 1000,
) -> tuple[Tour, float, float, list[float]]:
    """Steepest-ascent hill climbing with random restarts.

    Args:
        D: ``(n, n)`` distance matrix.
        seed: Integer seed fed to ``numpy.random.default_rng``.
        restarts: Number of independent climbs (default 20, from notebook).
        max_iter: Hard cap on improving moves per climb (default 1000).

    Returns:
        ``(best_route, best_length, elapsed_seconds, history)`` where
        ``history`` records the best length found after each restart.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    moves = two_opt_moves(n)
    t0 = time.perf_counter()

    best_route, best_len = None, float("inf")
    history: list[float] = []

    for _ in range(restarts):
        route = random_route(n, rng)
        length = tour_length(route, D)

        for _ in range(max_iter):
            best_delta, best_move = 0.0, None
            for (i, j) in moves:                          # scan whole neighbourhood
                delta = two_opt_delta(route, i, j, D)
                if delta < best_delta - 1e-12:            # strictly improving
                    best_delta, best_move = delta, (i, j)
            if best_move is None:                         # local optimum
                break
            route = two_opt_apply(route, *best_move)
            length += best_delta

        if length < best_len:
            best_len, best_route = length, route
        history.append(best_len)

    return best_route, best_len, time.perf_counter() - t0, history
