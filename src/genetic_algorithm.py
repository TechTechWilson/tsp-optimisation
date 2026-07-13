"""Genetic Algorithm (GA) for the Travelling Salesman Problem.

This is the population-based (evolutionary) algorithm for the
assessment. A GA evolves a *population* of candidate tours over many
generations using selection, crossover (recombination) and mutation.

WHY THE BASIC LECTURE/WORKBOOK OPERATORS DO NOT WORK HERE
--------------------------------------------------------
The introductory GA example evolves a fixed-length vector such as
``[x, y, z, w]`` using single-point crossover and "replace a gene with a
random value" mutation. A TSP tour is a *permutation* - every city must
appear exactly once. If we applied single-point crossover to two tours we
would usually get an invalid tour with duplicated and missing cities, for
example::

    parent1 = [0, 1, 2, 3, 4]
    parent2 = [3, 4, 0, 1, 2]
    single-point cut after index 2 -> [0, 1, 2, 1, 2]   # 1 and 2 repeat!

Permutation problems therefore require *order-preserving* operators. This
module implements two recombination operators (Order Crossover and
Partially Mapped Crossover) and two mutation operators (swap and
inversion) that always return a valid permutation.
"""

from __future__ import annotations

import random

import numpy as np

from .tsp import Tour, random_tour, tour_length


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def tournament_selection(
    population: list[Tour],
    lengths: list[float],
    k: int,
    rng: random.Random,
) -> Tour:
    """Select one parent by tournament selection.

    ``k`` individuals are drawn at random and the shortest tour among
    them wins. Larger ``k`` increases selection pressure (the search
    favours the current best more strongly). Tournament selection works
    directly with a minimisation objective, which is why it is preferred
    here over roulette wheel.

    Args:
        population: The current list of tours.
        lengths: Tour length for each individual (same order).
        k: Tournament size.
        rng: Seeded RNG.

    Returns:
        A copy of the winning tour.
    """
    contenders = rng.sample(range(len(population)), k)
    winner = min(contenders, key=lambda i: lengths[i])
    return population[winner][:]


def roulette_wheel_selection(
    population: list[Tour],
    lengths: list[float],
    k: int,  # Unused; kept so the signature matches tournament_selection.
    rng: random.Random,
) -> Tour:
    """Select one parent by roulette wheel selection (lecture method).

    Each individual is given a slice of a wheel proportional to its
    fitness. Because TSP is a minimisation problem, fitness is taken as
    ``1 / length`` so that shorter tours occupy larger slices. Provided
    as an alternative operator for the methodology comparison. The ``k``
    argument is ignored - it exists only so this function can be dropped
    in wherever ``tournament_selection`` is expected.

    Args:
        population: The current list of tours.
        lengths: Tour length for each individual.
        k: Ignored (signature compatibility only).
        rng: Seeded RNG.

    Returns:
        A copy of the selected tour.
    """
    fitnesses = [1.0 / length for length in lengths]
    total = sum(fitnesses)
    pick = rng.random() * total
    cumulative = 0.0
    for individual, fit in zip(population, fitnesses):
        cumulative += fit
        if cumulative >= pick:
            return individual[:]
    return population[-1][:]  # Fallback for floating-point edge cases.


# --------------------------------------------------------------------------- #
# Crossover (recombination)
# --------------------------------------------------------------------------- #
def order_crossover(parent1: Tour, parent2: Tour, rng: random.Random) -> Tour:
    """Order Crossover (OX1) - a permutation-safe recombination operator.

    A random slice of ``parent1`` is copied to the child in place. The
    remaining cities are then filled in, in the order they appear in
    ``parent2``, skipping any city already taken. This preserves a
    contiguous block from one parent and the relative order of the rest
    from the other, and always yields a valid permutation.

    Args:
        parent1: First parent tour.
        parent2: Second parent tour.
        rng: Seeded RNG.

    Returns:
        A valid child tour.
    """
    size = len(parent1)
    start, end = sorted(rng.sample(range(size), 2))

    child: list[int | None] = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    taken = set(parent1[start:end + 1])

    # Walk parent2 and drop in the cities not already present.
    fill_positions = [i for i in range(size) if child[i] is None]
    fill_values = [city for city in parent2 if city not in taken]
    for pos, city in zip(fill_positions, fill_values):
        child[pos] = city

    return child  # type: ignore[return-value]


def pmx_crossover(parent1: Tour, parent2: Tour, rng: random.Random) -> Tour:
    """Partially Mapped Crossover (PMX) - an alternative recombination.

    A slice of ``parent1`` is copied to the child, then PMX uses the
    position mapping between the two parents to legally place the
    remaining cities without creating duplicates. Offered alongside OX1
    so the report can compare recombination operators.

    Args:
        parent1: First parent tour.
        parent2: Second parent tour.
        rng: Seeded RNG.

    Returns:
        A valid child tour.
    """
    size = len(parent1)
    start, end = sorted(rng.sample(range(size), 2))

    child: list[int | None] = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    for i in range(start, end + 1):
        city = parent2[i]
        if city in child[start:end + 1]:
            continue
        # Follow the mapping until we find a free slot for this city.
        pos = i
        while True:
            mapped_city = parent1[pos]
            pos = parent2.index(mapped_city)
            if child[pos] is None:
                child[pos] = city
                break

    # Any remaining empty slots take parent2's city for that position.
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    return child  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #
def swap_mutation(tour: Tour, rng: random.Random) -> Tour:
    """Swap mutation: exchange two randomly chosen cities."""
    mutated = tour[:]
    i, j = rng.sample(range(len(tour)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def inversion_mutation(tour: Tour, rng: random.Random) -> Tour:
    """Inversion mutation: reverse a random segment of the tour.

    For TSP, inversion is usually more effective than a plain swap
    because, like a 2-opt move, it can remove crossed edges in a single
    step.
    """
    mutated = tour[:]
    i, j = sorted(rng.sample(range(len(tour)), 2))
    mutated[i:j + 1] = reversed(mutated[i:j + 1])
    return mutated


# --------------------------------------------------------------------------- #
# The Genetic Algorithm
# --------------------------------------------------------------------------- #
def genetic_algorithm(
    dist: np.ndarray,
    population_size: int = 100,
    generations: int = 500,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    tournament_size: int = 7,
    elitism: int = 5,
    crossover_fn=order_crossover,
    mutation_fn=inversion_mutation,
    selection_fn=tournament_selection,
    seed: int | None = None,
) -> tuple[Tour, float, list[float]]:
    """Solve a TSP instance with a Genetic Algorithm.

    Parameter provenance
    --------------------
    The defaults below were not chosen by intuition. They are the best
    configuration found by the sweep in ``tune_ga.py``, whose full results
    are written to ``results/ga_tuning.csv``. Three findings from that
    sweep are worth stating explicitly, because they justify the design:

    * ``tournament_size=7`` with ``elitism=5`` produced the shortest mean
      tour (589.06 over five seeds) and cost no extra run time compared
      with the weaker ``k=5, elitism=2`` baseline (590.10).
    * Inversion mutation beat swap mutation by a very wide margin
      (589.06 against 709.00). Inversion reverses a segment and can
      therefore uncross edges in the same way a 2-opt move does, so it
      suits a Euclidean TSP. A plain swap cannot do this.
    * Order Crossover (OX1) beat PMX (589.06 against 600.52).

    Spending more compute did not help. A deliberately heavier
    configuration (population 200, 1000 generations) was *worse* at
    605.93 despite taking roughly four times as long, which is consistent
    with premature convergence rather than with an insufficient budget.

    Args:
        dist: ``(n, n)`` distance matrix for the instance.
        population_size: Number of tours in each generation.
        generations: Number of generations to evolve.
        crossover_rate: Probability that a pair of parents is
            recombined (otherwise a parent is copied through).
        mutation_rate: Probability that a child is mutated.
        tournament_size: ``k`` for tournament selection.
        elitism: Number of best individuals copied unchanged into the
            next generation (guarantees the best tour is never lost).
        crossover_fn: Recombination operator (default Order Crossover).
        mutation_fn: Mutation operator (default inversion).
        selection_fn: Parent selection operator (default tournament).
            ``roulette_wheel_selection`` shares the same signature and
            can be dropped in directly.
        seed: Optional random seed for reproducible runs.

    Returns:
        A tuple ``(best_tour, best_length, history)`` where ``history``
        holds the best tour length at each generation.
    """
    rng = random.Random(seed)
    n_cities = dist.shape[0]

    # Initial population of random tours.
    population = [random_tour(n_cities, rng) for _ in range(population_size)]
    lengths = [tour_length(t, dist) for t in population]

    best_index = int(np.argmin(lengths))
    best_tour = population[best_index][:]
    best_length = lengths[best_index]
    history: list[float] = [best_length]

    for _ in range(generations):
        # Rank the population so elites can be carried over.
        ranked = sorted(range(population_size), key=lambda i: lengths[i])
        new_population = [population[i][:] for i in ranked[:elitism]]

        # Fill the rest of the next generation with offspring.
        while len(new_population) < population_size:
            parent1 = selection_fn(population, lengths, tournament_size, rng)
            parent2 = selection_fn(population, lengths, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = crossover_fn(parent1, parent2, rng)
            else:
                child = parent1[:]

            if rng.random() < mutation_rate:
                child = mutation_fn(child, rng)

            new_population.append(child)

        population = new_population
        lengths = [tour_length(t, dist) for t in population]

        # Track the best solution seen so far.
        generation_best = int(np.argmin(lengths))
        if lengths[generation_best] < best_length:
            best_length = lengths[generation_best]
            best_tour = population[generation_best][:]
        history.append(best_length)

    return best_tour, best_length, history
