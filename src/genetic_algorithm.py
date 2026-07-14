"""Genetic Algorithm for the Travelling Salesman Problem.

A population-based method rather than a trajectory-based one. Three
operators do the work:

==========  =================  ==============================================
Operator    Choice              Reason
==========  =================  ==============================================
Selection   Tournament, k = 7  Strong pressure; directs search toward fitness
Crossover   Order Crossover    Preserves relative city order, no duplicates
Mutation    Inversion, 0.6     Inversion **is** a 2-opt move — this is what
                               keeps the GA inside the same neighbourhood as
                               the other three algorithms
==========  =================  ==============================================

Elitism carries the best 5 individuals through untouched, so the best-so-far
can never regress.

Parameters from the notebook sweep: population 100, 260 generations,
mutation rate 0.6, tournament size 7, elitism 5.
"""

from __future__ import annotations

import time

import numpy as np

from .tsp import Tour, random_route, tour_length, two_opt_apply


def _tournament(pop_fit, k, rng):
    """Pick ``k`` individuals at random; return the index of the fittest."""
    contenders = rng.integers(0, len(pop_fit), size=k)
    winner = contenders[0]
    for c in contenders[1:]:
        if pop_fit[c] < pop_fit[winner]:
            winner = c
    return int(winner)


def _ox1(p1, p2, rng):
    """Order Crossover (OX1).

    Copy a random slice of parent 1 into the child, then fill the remaining
    slots with the cities of parent 2 in the order they appear there,
    skipping anything already present. This preserves relative city order
    from one parent and absolute position from the other, and can never
    produce a duplicate.
    """
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [-1] * n
    child[a:b + 1] = p1[a:b + 1]
    taken = set(child[a:b + 1])

    fill = [c for c in p2 if c not in taken]
    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1
    return child


def _inversion_mutation(route, rate, rng):
    """Reverse a random segment.

    This is a 2-opt move applied blindly — exactly what keeps the GA inside
    the same neighbourhood as the other three algorithms.
    """
    if rng.random() >= rate:
        return route
    n = len(route)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    return two_opt_apply(route, i, j)


def genetic_algorithm(
    D: np.ndarray,
    seed: int,
    pop_size: int = 100,
    generations: int = 260,
    mutation_rate: float = 0.6,
    tournament_k: int = 7,
    elitism: int = 5,
) -> tuple[Tour, float, float, list[float]]:
    """Generational GA with elitism, OX1 crossover and inversion mutation.

    Args:
        D: ``(n, n)`` distance matrix.
        seed: Integer seed for ``numpy.random.default_rng``.
        pop_size: Population size (default 100).
        generations: Number of generations (default 260).
        mutation_rate: Probability a child is mutated (default 0.6).
        tournament_k: Tournament size for parent selection (default 7).
        elitism: Number of best individuals copied intact each generation.

    Returns:
        ``(best_route, best_length, elapsed_seconds, history)`` where
        ``history`` records the best length after each generation.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    t0 = time.perf_counter()

    pop = [random_route(n, rng) for _ in range(pop_size)]
    fit = [tour_length(r, D) for r in pop]
    history: list[float] = [min(fit)]

    for _ in range(generations):
        order = sorted(range(pop_size), key=lambda i: fit[i])
        new_pop = [list(pop[i]) for i in order[:elitism]]    # elites survive intact
        new_fit = [fit[i] for i in order[:elitism]]

        while len(new_pop) < pop_size:
            p1 = pop[_tournament(fit, tournament_k, rng)]
            p2 = pop[_tournament(fit, tournament_k, rng)]
            child = _ox1(p1, p2, rng)
            child = _inversion_mutation(child, mutation_rate, rng)
            new_pop.append(child)
            new_fit.append(tour_length(child, D))

        pop, fit = new_pop, new_fit
        history.append(min(fit))

    best_i = int(np.argmin(fit))
    return pop[best_i], fit[best_i], time.perf_counter() - t0, history
