"""Population-based search: a genetic algorithm for the TSP.

Every operator below is written from scratch. Design choices:

* **Representation.** Path representation: a chromosome is a permutation of
  city indices, read as the visiting order. This makes every chromosome a
  valid tour by construction, so no repair step is ever needed.

* **Selection.** Tournament selection. Selection pressure is controlled by a
  single readable parameter (the tournament size ``k``), and unlike
  fitness-proportionate selection it is invariant to the scale of the
  objective, which matters here because tour lengths vary by an order of
  magnitude across problem sizes.

* **Recombination.** Order crossover (OX1). OX1 copies a contiguous slice from
  one parent and fills the remainder in the relative order given by the other
  parent, so it preserves both absolute position information and relative
  ordering, and it always emits a valid permutation.

* **Mutation.** Inversion, which reverses a random segment. This is the 2-opt
  move in disguise, so the mutation operator and the hill climber's
  neighbourhood explore the same landscape. Any performance difference between
  the two algorithms is therefore attributable to search strategy rather than
  to one algorithm being handed a better move operator.

* **Elitism.** The best ``e`` chromosomes survive unchanged into the next
  generation, which guarantees the best-so-far solution can never get worse.
"""

from __future__ import annotations

import numpy as np

from tsp_core import population_lengths, random_tour


def tournament_select(population, fitness, k, rng):
    """Pick one parent by tournament selection.

    Args:
        population (list[list[int]]): The current generation.
        fitness (numpy.ndarray): Tour length of each chromosome. Lower is
            better, so the tournament winner is the minimum.
        k (int): Tournament size. Larger values raise selection pressure.
        rng (numpy.random.Generator): Seeded generator.

    Returns:
        list[int]: A copy of the winning chromosome.
    """
    contenders = rng.integers(0, len(population), size=k)
    winner = contenders[int(np.argmin(fitness[contenders]))]
    return list(population[winner])


def order_crossover(parent_a, parent_b, rng):
    """Order crossover (OX1) producing one child from two parents.

    A random slice of ``parent_a`` is copied into the child at the same
    positions. The remaining positions are filled, left to right starting after
    the slice, with the cities of ``parent_b`` in the order they appear there,
    skipping any city already present.

    Args:
        parent_a (Sequence[int]): First parent.
        parent_b (Sequence[int]): Second parent.
        rng (numpy.random.Generator): Seeded generator.

    Returns:
        list[int]: A valid permutation inheriting from both parents.
    """
    n = len(parent_a)
    cut_one, cut_two = sorted(rng.choice(n, size=2, replace=False))

    child = [-1] * n
    child[cut_one:cut_two + 1] = list(parent_a[cut_one:cut_two + 1])
    taken = set(child[cut_one:cut_two + 1])

    fill_position = (cut_two + 1) % n
    for offset in range(n):
        city = parent_b[(cut_two + 1 + offset) % n]
        if city not in taken:
            child[fill_position] = city
            taken.add(city)
            fill_position = (fill_position + 1) % n

    return child


def inversion_mutation(chromosome, rng):
    """Reverse a randomly chosen segment of the chromosome, in place-safe form.

    Args:
        chromosome (list[int]): The tour to mutate.
        rng (numpy.random.Generator): Seeded generator.

    Returns:
        list[int]: A new mutated tour; the input is left unmodified.
    """
    n = len(chromosome)
    start, end = sorted(rng.choice(n, size=2, replace=False))
    return chromosome[:start] + chromosome[start:end + 1][::-1] + chromosome[end + 1:]


def genetic_algorithm(dist, population_size=150, generations=400,
                      crossover_rate=0.9, mutation_rate=0.2,
                      tournament_size=5, elitism=4, rng=None):
    """Run a generational GA with elitism on a TSP distance matrix.

    Args:
        dist (numpy.ndarray): Distance matrix.
        population_size (int): Number of chromosomes per generation.
        generations (int): Number of generations to evolve.
        crossover_rate (float): Probability that a pair of parents is
            recombined rather than copied.
        mutation_rate (float): Probability that a child is mutated.
        tournament_size (int): Selection pressure.
        elitism (int): Number of best chromosomes carried over unchanged.
        rng (numpy.random.Generator | None): Seeded generator.

    Returns:
        tuple[list[int], float, list[float]]: Best tour found, its length, and
        the best-so-far length recorded at every generation.

    Raises:
        ValueError: If ``elitism`` is not smaller than ``population_size``.
    """
    if elitism >= population_size:
        raise ValueError("elitism must be smaller than population_size")
    if rng is None:
        rng = np.random.default_rng()

    n = dist.shape[0]
    population = [random_tour(n, rng) for _ in range(population_size)]
    fitness = population_lengths(population, dist)

    best_index = int(np.argmin(fitness))
    best_tour = list(population[best_index])
    best_length = float(fitness[best_index])
    history = [best_length]

    for _ in range(generations):
        # Elitism: carry the best chromosomes over untouched.
        order = np.argsort(fitness)
        next_population = [list(population[i]) for i in order[:elitism]]

        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitness, tournament_size, rng)
            parent_b = tournament_select(population, fitness, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = order_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a

            if rng.random() < mutation_rate:
                child = inversion_mutation(child, rng)

            next_population.append(child)

        population = next_population
        fitness = population_lengths(population, dist)

        generation_best = int(np.argmin(fitness))
        if fitness[generation_best] < best_length:
            best_length = float(fitness[generation_best])
            best_tour = list(population[generation_best])
        history.append(best_length)

    return best_tour, best_length, history
