"""Tabu Search for the Travelling Salesman Problem.

Tabu Search is a single-member (local) search metaheuristic. Like Hill
Climbing it holds one current solution and repeatedly moves to a
neighbour, but it differs in two decisive ways:

1. It moves to the *best* neighbour in a sampled neighbourhood **even if
   that neighbour is worse** than the current tour. This is what lets it
   walk out of a local optimum, whereas a hill climber simply stops.
2. It keeps a short-term memory, the *tabu list*, of recently-performed
   moves. Those moves are forbidden for a number of iterations (the
   *tabu tenure*), which stops the search from immediately undoing the
   move it has just made and cycling between the same two tours.

An *aspiration criterion* overrides the tabu status: if a forbidden move
would produce a tour better than any found so far, it is allowed anyway.
Without this the memory can block genuinely excellent moves.

Design notes for this implementation
------------------------------------
* Move type: 2-opt segment reversal, the same neighbourhood used by
  Simulated Annealing and Hill Climbing. Using an identical neighbourhood
  across all three single-member algorithms means the comparison isolates
  the *search strategy* rather than the move operator.
* Tabu attribute: the unordered pair of city indices ``(i, j)`` whose
  segment was reversed. Storing the move rather than the whole tour keeps
  memory small and lookups fast.
* Neighbourhood sampling: evaluating all n(n-1)/2 2-opt moves each
  iteration is expensive. A random sample of candidate moves is scored
  instead, which keeps the cost per iteration bounded and is standard
  practice on larger instances.
"""

from __future__ import annotations

import random

import numpy as np

from .tsp import Tour, random_tour, tour_length


def tabu_search(
    dist: np.ndarray,
    max_iterations: int = 2000,
    tabu_tenure: int = 15,
    neighbourhood_size: int = 60,
    seed: int | None = None,
) -> tuple[Tour, float, list[float]]:
    """Solve a TSP instance with Tabu Search.

    Args:
        dist: ``(n, n)`` distance matrix for the instance.
        max_iterations: Number of search iterations to perform.
        tabu_tenure: Number of iterations for which a move stays
            forbidden. Small tenures risk cycling; very large tenures
            over-restrict the search and starve it of good moves.
        neighbourhood_size: Number of candidate 2-opt moves sampled and
            scored per iteration.
        seed: Optional random seed for reproducibility.

    Returns:
        A tuple ``(best_tour, best_length, history)`` where ``history``
        records the best length found so far at each iteration, so the
        convergence curve can be plotted on the same axes as the other
        algorithms.
    """
    rng = random.Random(seed)
    n = len(dist)

    current = random_tour(n, rng)
    current_len = tour_length(current, dist)

    best = current[:]
    best_len = current_len

    # Maps a move (i, j) -> the iteration at which it stops being tabu.
    tabu: dict[tuple[int, int], int] = {}
    history: list[float] = [best_len]

    for iteration in range(max_iterations):
        best_neighbour: Tour | None = None
        best_neighbour_len = float("inf")
        best_move: tuple[int, int] | None = None

        # Score a random sample of 2-opt moves from the neighbourhood.
        for _ in range(neighbourhood_size):
            i, j = sorted(rng.sample(range(n), 2))
            if j - i < 1:
                continue

            candidate = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
            candidate_len = tour_length(candidate, dist)

            move = (i, j)
            is_tabu = tabu.get(move, 0) > iteration

            # Aspiration: allow a tabu move if it beats the global best.
            if is_tabu and candidate_len >= best_len:
                continue

            if candidate_len < best_neighbour_len:
                best_neighbour = candidate
                best_neighbour_len = candidate_len
                best_move = move

        # Every sampled move was tabu; carry on and let the list expire.
        if best_neighbour is None:
            history.append(best_len)
            continue

        # Move to the best admissible neighbour even if it is worse than
        # the current tour. This is the escape mechanism.
        current = best_neighbour
        current_len = best_neighbour_len
        if best_move is not None:
            tabu[best_move] = iteration + tabu_tenure

        if current_len < best_len:
            best = current[:]
            best_len = current_len

        history.append(best_len)

    return best, best_len, history
