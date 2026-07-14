"""Tabu Search for the Travelling Salesman Problem.

Tabu Search always moves to the best candidate available, **even when that
move makes the tour worse**. What stops it immediately undoing that move is
the tabu list: a recently used move is forbidden for ``tenure`` iterations.

The aspiration criterion overrides the ban when a forbidden move would
yield a new global best — refusing a record-breaking tour on a
technicality would be perverse.

The neighbourhood is *sampled* rather than scanned in full, which keeps the
per-iteration cost comparable to the other three algorithms.

Parameters from the notebook: 1200 iterations, tabu tenure 15,
neighbourhood sample size 260.
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


def tabu_search(
    D: np.ndarray,
    seed: int,
    iterations: int = 1200,
    tenure: int = 15,
    sample_size: int = 260,
) -> tuple[Tour, float, float, list[float]]:
    """Best-of-neighbourhood search with short-term memory.

    Args:
        D: ``(n, n)`` distance matrix.
        seed: Integer seed for ``numpy.random.default_rng``.
        iterations: Search iterations (default 1200).
        tenure: Tabu list length — a recently used move is forbidden for
            this many iterations.
        sample_size: Number of 2-opt moves sampled and scored per
            iteration.

    Returns:
        ``(best_route, best_length, elapsed_seconds, history)`` where
        ``history`` records the best length after each iteration.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    all_moves = two_opt_moves(n)
    t0 = time.perf_counter()

    route = random_route(n, rng)
    length = tour_length(route, D)
    best_route, best_len = list(route), length

    tabu: dict[tuple[int, int], int] = {}                # move -> iteration it expires
    history: list[float] = [length]

    for it in range(iterations):
        idx = rng.choice(
            len(all_moves),
            size=min(sample_size, len(all_moves)),
            replace=False,
        )
        cand_move, cand_delta = None, float("inf")

        for k in idx:
            i, j = all_moves[k]
            delta = two_opt_delta(route, i, j, D)
            is_tabu = tabu.get((i, j), 0) > it
            aspires = (length + delta) < best_len - 1e-12

            if is_tabu and not aspires:                   # forbidden, and not special
                continue
            if delta < cand_delta:
                cand_move, cand_delta = (i, j), delta

        if cand_move is None:                             # everything was tabu
            continue

        route = two_opt_apply(route, *cand_move)
        length += cand_delta
        tabu[cand_move] = it + tenure                     # serve your sentence

        if length < best_len:
            best_len, best_route = length, list(route)
        history.append(best_len)

    return best_route, best_len, time.perf_counter() - t0, history
