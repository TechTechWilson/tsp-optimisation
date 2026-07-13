"""Sanity tests for the TSP project.

These tests do not check that the algorithms find the *optimal* tour
(that is not guaranteed for a metaheuristic). Instead they check the
properties that must always hold: tours stay valid permutations, the
objective function is computed correctly, and each algorithm improves on
a random starting tour. Run with ``pytest`` from the project root.
"""

import random

import numpy as np

from src.genetic_algorithm import (
    genetic_algorithm,
    inversion_mutation,
    order_crossover,
    pmx_crossover,
    swap_mutation,
)
from src.hill_climbing import hill_climbing
from src.simulated_annealing import simulated_annealing
from src.tabu_search import tabu_search
from src.tsp import (
    build_distance_matrix,
    random_tour,
    tour_length,
    two_opt_move,
)


def _toy_problem(n=12, seed=1):
    """Build a small reproducible coordinate problem for testing."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))
    return coords, build_distance_matrix(coords)


def _is_valid_tour(tour, n):
    """A valid tour visits every city 0..n-1 exactly once."""
    return sorted(tour) == list(range(n))


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    _, dist = _toy_problem()
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)


def test_tour_length_matches_manual_sum():
    coords = np.array([[0, 0], [0, 3], [4, 3], [4, 0]], dtype=float)
    dist = build_distance_matrix(coords)
    # Perimeter of a 3x4 rectangle = 3 + 4 + 3 + 4 = 14.
    assert abs(tour_length([0, 1, 2, 3], dist) - 14.0) < 1e-9


def test_neighbourhood_moves_return_valid_tours():
    rng = random.Random(0)
    tour = random_tour(12, rng)
    assert _is_valid_tour(two_opt_move(tour, rng), 12)
    assert _is_valid_tour(swap_mutation(tour, rng), 12)
    assert _is_valid_tour(inversion_mutation(tour, rng), 12)


def test_crossover_operators_return_valid_permutations():
    rng = random.Random(0)
    p1 = random_tour(12, rng)
    p2 = random_tour(12, rng)
    assert _is_valid_tour(order_crossover(p1, p2, rng), 12)
    assert _is_valid_tour(pmx_crossover(p1, p2, rng), 12)


def test_simulated_annealing_improves_on_random():
    _, dist = _toy_problem()
    start = tour_length(random_tour(dist.shape[0], random.Random(0)), dist)
    _, best, _ = simulated_annealing(dist, iters_per_temp=50, seed=0)
    assert best <= start


def test_hill_climbing_improves_on_random():
    _, dist = _toy_problem()
    start = tour_length(random_tour(dist.shape[0], random.Random(0)), dist)
    _, best, _ = hill_climbing(dist, max_iterations=2000, seed=0)
    assert best <= start


def test_genetic_algorithm_returns_valid_best_tour():
    _, dist = _toy_problem()
    tour, best, history = genetic_algorithm(
        dist, population_size=30, generations=50, seed=0
    )
    assert _is_valid_tour(tour, dist.shape[0])
    assert abs(tour_length(tour, dist) - best) < 1e-9
    # Convergence is monotonically non-increasing (best-so-far).
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_tabu_search_returns_valid_improved_tour():
    """Tabu Search returns a valid tour that improves on its start."""
    _, dist = _toy_problem()
    tour, best, history = tabu_search(dist, max_iterations=300, seed=0)

    assert _is_valid_tour(tour, dist.shape[0])
    assert abs(tour_length(tour, dist) - best) < 1e-9
    # Best-so-far history must be monotonically non-increasing.
    assert all(b <= a for a, b in zip(history, history[1:]))
    assert best <= history[0]
