"""Stochastic Hill Climbing for the Travelling Salesman Problem.

This is included as a deliberately simple *baseline* local search. It
follows the "Stochastic Hill Climber" described in the Week 5/6 lectures:
generate a random neighbour, accept it only if it improves the tour, and
never accept a worse move.

Its purpose in this project is to demonstrate the weakness the lectures
highlight - a pure hill climber has no mechanism to escape local optima,
so it tends to stall at a worse solution than Simulated Annealing or the
Genetic Algorithm. That contrast is exactly what the comparison report
needs to discuss.
"""

from __future__ import annotations

import random

import numpy as np

from .tsp import Tour, random_tour, tour_length, two_opt_move


def hill_climbing(
    dist: np.ndarray,
    max_iterations: int = 20000,
    patience: int = 2000,
    neighbour_fn=two_opt_move,
    seed: int | None = None,
) -> tuple[Tour, float, list[float]]:
    """Solve a TSP instance with stochastic hill climbing.

    Args:
        dist: ``(n, n)`` distance matrix for the instance.
        max_iterations: Hard cap on the number of candidate moves.
        patience: Stop early if no improvement is found within this many
            consecutive iterations (the search has stalled at a local
            optimum).
        neighbour_fn: Function used to generate a neighbouring tour.
        seed: Optional random seed for reproducible runs.

    Returns:
        A tuple ``(best_tour, best_length, history)``.
    """
    rng = random.Random(seed)
    n_cities = dist.shape[0]

    current = random_tour(n_cities, rng)
    current_length = tour_length(current, dist)
    history: list[float] = [current_length]

    iterations_without_improvement = 0
    for _ in range(max_iterations):
        candidate = neighbour_fn(current, rng)
        candidate_length = tour_length(candidate, dist)

        if candidate_length < current_length:
            current = candidate
            current_length = candidate_length
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        history.append(current_length)

        if iterations_without_improvement >= patience:
            break  # Stalled at a local optimum - no way to escape.

    return current, current_length, history
