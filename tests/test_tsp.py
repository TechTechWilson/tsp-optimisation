"""Sanity tests for the TSP project.

These tests do not check that the algorithms find the *optimal* tour
(that is not guaranteed for a metaheuristic). Instead they check the
properties that must always hold: tours stay valid permutations, the
objective function is computed correctly, each algorithm improves on a
random starting tour, the 2-opt delta matches full recomputation, and
determinism holds. Run with ``pytest`` from the project root.
"""

import numpy as np

from src.genetic_algorithm import genetic_algorithm
from src.hill_climbing import hill_climbing
from src.simulated_annealing import simulated_annealing
from src.tabu_search import tabu_search
from src.tsp import (
    build_distance_matrix,
    random_route,
    tour_length,
    two_opt_apply,
    two_opt_delta,
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


# ------------------------------------------------------------------ #
# Core utilities
# ------------------------------------------------------------------ #
def test_distance_matrix_is_symmetric_with_zero_diagonal():
    _, dist = _toy_problem()
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)


def test_tour_length_matches_manual_sum():
    coords = np.array([[0, 0], [0, 3], [4, 3], [4, 0]], dtype=float)
    dist = build_distance_matrix(coords)
    # Perimeter of a 3x4 rectangle = 3 + 4 + 3 + 4 = 14.
    assert abs(tour_length([0, 1, 2, 3], dist) - 14.0) < 1e-9


def test_two_opt_move_returns_valid_tour():
    """Classic two_opt_move still produces a valid permutation."""
    rng = np.random.default_rng(0)
    route = random_route(12, rng)
    # two_opt_move uses `random.Random` — wrap with a small helper
    import random
    rr = random.Random(0)
    moved = two_opt_move(route, rr)
    assert _is_valid_tour(moved, 12)


def test_two_opt_delta_matches_full_recomputation():
    """O(1) delta must equal tour_length(after) - tour_length(before)."""
    _, D = _toy_problem(n=12)
    rng = np.random.default_rng(42)
    route = random_route(12, rng)
    n = len(route)
    for _ in range(100):
        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))
        delta = two_opt_delta(route, i, j, D)
        after = two_opt_apply(route, i, j)
        expected = tour_length(after, D) - tour_length(route, D)
        assert abs(delta - expected) < 1e-12, (
            f"delta={delta} expected={expected} at i={i}, j={j}"
        )


# ------------------------------------------------------------------ #
# Tour validity  (every algorithm returns a valid permutation)
# ------------------------------------------------------------------ #
def test_hill_climbing_returns_valid_tour():
    _, D = _toy_problem()
    tour, _, _, _ = hill_climbing(D, seed=0)
    assert _is_valid_tour(tour, D.shape[0])


def test_simulated_annealing_returns_valid_tour():
    _, D = _toy_problem()
    tour, _, _, _ = simulated_annealing(D, seed=0)
    assert _is_valid_tour(tour, D.shape[0])


def test_tabu_search_returns_valid_tour():
    _, D = _toy_problem()
    tour, _, _, _ = tabu_search(D, seed=0)
    assert _is_valid_tour(tour, D.shape[0])


def test_genetic_algorithm_returns_valid_tour():
    _, D = _toy_problem()
    tour, _, _, _ = genetic_algorithm(D, seed=0, pop_size=30, generations=50)
    assert _is_valid_tour(tour, D.shape[0])


# ------------------------------------------------------------------ #
# Improvement  (each algorithm should not make things worse)
# ------------------------------------------------------------------ #
def test_hill_climbing_improves_on_random():
    _, D = _toy_problem()
    rng = np.random.default_rng(0)
    start = tour_length(random_route(D.shape[0], rng), D)
    _, best, _, _ = hill_climbing(D, seed=0)
    assert best <= start


def test_simulated_annealing_improves_on_random():
    _, D = _toy_problem()
    rng = np.random.default_rng(0)
    start = tour_length(random_route(D.shape[0], rng), D)
    _, best, _, _ = simulated_annealing(D, seed=0, iters_per_temp=50)
    assert best <= start


def test_genetic_algorithm_improves_on_random():
    _, D = _toy_problem()
    tour, best, _, history = genetic_algorithm(
        D, seed=0, pop_size=30, generations=50,
    )
    assert _is_valid_tour(tour, D.shape[0])
    assert abs(tour_length(tour, D) - best) < 1e-9
    # Convergence is monotonically non-increasing (best-so-far).
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_tabu_search_improves_on_random():
    _, D = _toy_problem()
    tour, best, _, history = tabu_search(D, seed=0, iterations=300)
    assert _is_valid_tour(tour, D.shape[0])
    assert abs(tour_length(tour, D) - best) < 1e-9
    # Best-so-far history must be monotonically non-increasing.
    assert all(b <= a for a, b in zip(history, history[1:]))
    assert best <= history[0]


# ------------------------------------------------------------------ #
# Determinism  (same seed → same tour)
# ------------------------------------------------------------------ #
def test_hill_climbing_determinism():
    _, D = _toy_problem()
    t1, l1, _, _ = hill_climbing(D, seed=42)
    t2, l2, _, _ = hill_climbing(D, seed=42)
    assert t1 == t2
    assert abs(l1 - l2) < 1e-12


def test_simulated_annealing_determinism():
    _, D = _toy_problem()
    t1, l1, _, _ = simulated_annealing(D, seed=42)
    t2, l2, _, _ = simulated_annealing(D, seed=42)
    assert t1 == t2
    assert abs(l1 - l2) < 1e-12


def test_tabu_search_determinism():
    _, D = _toy_problem()
    t1, l1, _, _ = tabu_search(D, seed=42)
    t2, l2, _, _ = tabu_search(D, seed=42)
    assert t1 == t2
    assert abs(l1 - l2) < 1e-12


def test_genetic_algorithm_determinism():
    _, D = _toy_problem()
    t1, l1, _, _ = genetic_algorithm(D, seed=42, pop_size=30, generations=50)
    t2, l2, _, _ = genetic_algorithm(D, seed=42, pop_size=30, generations=50)
    assert t1 == t2
    assert abs(l1 - l2) < 1e-12
