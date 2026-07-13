"""Simulated Annealing (SA) for the Travelling Salesman Problem.

Simulated Annealing is the single-solution (local search) metaheuristic
chosen for this assessment. It is inspired by the cooling of metals:

* a high *temperature* makes the search willing to accept worse tours,
  which lets it escape local optima (exploration);
* as the temperature is slowly reduced by the *cooling schedule*, the
  search accepts fewer bad moves and settles on refining good tours
  (exploitation).

A worse move (one that lengthens the tour) is accepted with probability
``exp(-delta / T)``, exactly as shown in the Week 7 lecture. Because TSP
is a minimisation problem, ``delta = new_length - current_length`` and a
*negative* delta is always an improvement.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .tsp import Tour, random_tour, tour_length, two_opt_move


def simulated_annealing(
    dist: np.ndarray,
    initial_temp: float = 1000.0,
    min_temp: float = 1e-3,
    cooling_rate: float = 0.995,
    iters_per_temp: int = 100,
    neighbour_fn=two_opt_move,
    seed: int | None = None,
) -> tuple[Tour, float, list[float]]:
    """Solve a TSP instance with Simulated Annealing.

    Args:
        dist: ``(n, n)`` distance matrix for the instance.
        initial_temp: Starting temperature ``T0``. Higher values allow
            more exploration early on.
        min_temp: Temperature at which the search stops.
        cooling_rate: Geometric cooling factor ``alpha`` in ``T = alpha * T``.
            Values closer to 1.0 cool more slowly (better quality, slower).
        iters_per_temp: Number of candidate moves tried at each
            temperature level.
        neighbour_fn: Function used to generate a neighbouring tour
            (defaults to a 2-opt segment reversal).
        seed: Optional random seed for reproducible runs.

    Returns:
        A tuple ``(best_tour, best_length, history)`` where ``history``
        is the best length recorded after each temperature level (useful
        for plotting convergence).
    """
    rng = random.Random(seed)
    n_cities = dist.shape[0]

    current = random_tour(n_cities, rng)
    current_length = tour_length(current, dist)

    best = current[:]
    best_length = current_length

    history: list[float] = [best_length]
    temperature = initial_temp

    while temperature > min_temp:
        for _ in range(iters_per_temp):
            candidate = neighbour_fn(current, rng)
            candidate_length = tour_length(candidate, dist)
            delta = candidate_length - current_length

            # Accept improving moves outright; accept worsening moves
            # with a temperature-dependent probability.
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                current = candidate
                current_length = candidate_length
                if current_length < best_length:
                    best = current[:]
                    best_length = current_length

        history.append(best_length)
        temperature *= cooling_rate  # Geometric cooling schedule.

    return best, best_length, history
