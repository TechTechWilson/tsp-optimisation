"""Simulated Annealing for the Travelling Salesman Problem.

Simulated Annealing is a single-solution metaheuristic inspired by the
cooling of metals. At high temperatures the search accepts worsening moves
freely, allowing it to roam the landscape and escape local optima. As the
temperature is reduced geometrically the acceptance probability of bad
moves collapses and the search settles into refinement.

The only difference from hill climbing is the Metropolis acceptance
criterion: a worsening move is accepted with probability
``exp(-delta / T)``.

Parameters from the notebook sweep (Section 8):
``T0 = 100``, ``alpha = 0.9998``, ``T_end = 1e-3``, 12 iterations per
temperature level.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .tsp import Tour, random_route, tour_length, two_opt_apply, two_opt_delta


def simulated_annealing(
    D: np.ndarray,
    seed: int,
    t_start: float = 100.0,
    t_end: float = 1e-3,
    alpha: float = 0.9998,
    iters_per_temp: int = 12,
) -> tuple[Tour, float, float, list[float]]:
    """Metropolis acceptance with geometric cooling.

    Args:
        D: ``(n, n)`` distance matrix.
        seed: Integer seed for ``numpy.random.default_rng``.
        t_start: Initial temperature ``T0`` (default 100.0).
        t_end: Temperature at which the search stops.
        alpha: Geometric cooling factor in ``T = alpha * T``.
        iters_per_temp: Candidate moves tried at each temperature level.

    Returns:
        ``(best_route, best_length, elapsed_seconds, history)`` where
        ``history`` records the best length after each temperature step.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    t0 = time.perf_counter()

    route = random_route(n, rng)
    length = tour_length(route, D)
    best_route, best_len = list(route), length

    T = t_start
    history: list[float] = [length]

    while T > t_end:
        for _ in range(iters_per_temp):
            i = int(rng.integers(1, n - 1))
            j = int(rng.integers(i + 1, n))
            delta = two_opt_delta(route, i, j, D)

            if delta < 0 or rng.random() < math.exp(-delta / T):
                route = two_opt_apply(route, i, j)
                length += delta
                if length < best_len:
                    best_len, best_route = length, list(route)
        history.append(best_len)
        T *= alpha                                       # geometric cooling

    return best_route, best_len, time.perf_counter() - t0, history
