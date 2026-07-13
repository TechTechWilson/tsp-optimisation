"""Core utilities for the Travelling Salesman Problem.

This module holds everything that is shared between the search algorithms:
loading the city data, building the Euclidean distance matrix, evaluating a
tour, and writing the best route out to an external file.

Nothing in this project imports a ready-made TSP solver. Every algorithm is
written from first principles; NumPy is used only for array storage and
vectorised arithmetic.
"""

from __future__ import annotations

import csv
import os

import numpy as np


def load_cities(path):
    """Read a city CSV file into a coordinate array.

    The file is expected to have a header row and the columns
    ``City, X, Y`` (case-insensitive).

    Args:
        path (str): Path to the CSV file.

    Returns:
        tuple[list[str], numpy.ndarray]: City names, and an ``(n, 2)`` array
        of float coordinates.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains fewer than three cities or any
            coordinate cannot be parsed as a float.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"City data file not found: {path}")

    names = []
    coords = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        lookup = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        for required in ("x", "y"):
            if required not in lookup:
                raise ValueError(f"Column '{required}' missing from {path}")
        name_key = lookup.get("city")
        for row_number, row in enumerate(reader, start=2):
            try:
                x = float(row[lookup["x"]])
                y = float(row[lookup["y"]])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Bad coordinate on line {row_number} of {path}"
                ) from error
            names.append(row[name_key] if name_key else f"City_{row_number - 1}")
            coords.append((x, y))

    if len(coords) < 3:
        raise ValueError("A TSP instance needs at least three cities")

    return names, np.asarray(coords, dtype=float)


def distance_matrix(coords):
    """Build the symmetric Euclidean distance matrix for a set of points.

    Pre-computing the matrix once turns every later tour evaluation into a
    handful of table look-ups rather than repeated square roots, which is the
    single biggest speed win available to all four algorithms.

    Args:
        coords (numpy.ndarray): An ``(n, 2)`` array of coordinates.

    Returns:
        numpy.ndarray: An ``(n, n)`` array where entry ``[i, j]`` is the
        straight-line distance between city ``i`` and city ``j``.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def tour_length(tour, dist):
    """Return the total length of a closed tour.

    Args:
        tour (Sequence[int]): A permutation of city indices.
        dist (numpy.ndarray): The distance matrix.

    Returns:
        float: Sum of the edges of the tour, including the edge that returns
        from the final city to the starting city.
    """
    order = np.asarray(tour, dtype=int)
    return float(dist[order, np.roll(order, -1)].sum())


def two_opt_delta(tour, dist, i, j):
    """Return the change in tour length caused by a single 2-opt move.

    A 2-opt move reverses the segment ``tour[i:j + 1]``. Doing so removes the
    two edges entering and leaving that segment and replaces them with two
    new edges. Because the rest of the tour is untouched, the effect on the
    objective can be computed in constant time from four table look-ups
    instead of re-summing all ``n`` edges. This is what makes an exhaustive
    steepest-ascent sweep affordable.

    Args:
        tour (Sequence[int]): The current permutation.
        dist (numpy.ndarray): The distance matrix.
        i (int): Start index of the segment to reverse (``i >= 1``).
        j (int): End index of the segment to reverse (``j > i``).

    Returns:
        float: The signed change in tour length. Negative means an
        improvement, because this is a minimisation problem.
    """
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])


def two_opt_delta_matrix(tour, dist):
    """Score every 2-opt move on a tour in one vectorised pass.

    This is the same arithmetic as :func:`two_opt_delta`, evaluated for all
    ``(i, j)`` pairs simultaneously. It is what the hill climber actually calls,
    because a Python-level double loop over roughly 1,200 candidate moves per
    sweep would dominate the runtime and make the scalability experiment
    impractical.

    Args:
        tour (Sequence[int]): The current permutation.
        dist (numpy.ndarray): The distance matrix.

    Returns:
        numpy.ndarray: An ``(n, n)`` array of deltas. Entries outside the valid
        range ``1 <= i < j <= n - 1`` are set to ``+inf`` so they can never be
        selected as the best move.
    """
    order = np.asarray(tour, dtype=int)
    n = order.size

    previous = order[np.arange(n) - 1]   # city before position i
    following = np.roll(order, -1)       # city after position j

    added = dist[np.ix_(previous, order)] + dist[np.ix_(order, following)]
    removed = (dist[previous, order][:, np.newaxis]
               + dist[order, following][np.newaxis, :])
    delta = added - removed

    rows = np.arange(n)[:, np.newaxis]
    cols = np.arange(n)[np.newaxis, :]
    valid = (rows >= 1) & (cols > rows) & (cols <= n - 1)
    return np.where(valid, delta, np.inf)


def population_lengths(population, dist):
    """Evaluate a whole population of tours at once.

    Args:
        population (Sequence[Sequence[int]]): Chromosomes, all of equal length.
        dist (numpy.ndarray): The distance matrix.

    Returns:
        numpy.ndarray: A 1-D array of tour lengths, one per chromosome.
    """
    arr = np.asarray(population, dtype=int)
    return dist[arr, np.roll(arr, -1, axis=1)].sum(axis=1)


def apply_two_opt(tour, i, j):
    """Return a new tour with the segment ``tour[i:j + 1]`` reversed.

    Args:
        tour (list[int]): The current permutation.
        i (int): Start index of the segment.
        j (int): End index of the segment.

    Returns:
        list[int]: A new list; the input is left unmodified.
    """
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def random_tour(n, rng):
    """Return a uniformly random permutation of ``0 .. n - 1``.

    Args:
        n (int): Number of cities.
        rng (numpy.random.Generator): Seeded random generator.

    Returns:
        list[int]: A random tour.
    """
    return list(rng.permutation(n))


def save_route(path, tour, names, dist):
    """Write the best route found to an external plain-text file.

    The assessment brief requires the sequence of cities for the best route to
    be recorded in an external file, so this writes both the human-readable
    city names and the raw indices, along with the tour length.

    Args:
        path (str): Destination file path.
        tour (Sequence[int]): The permutation to record.
        names (Sequence[str]): City names, indexed the same way as ``tour``.
        dist (numpy.ndarray): The distance matrix, used to report the length.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    length = tour_length(tour, dist)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Best tour found\n")
        handle.write(f"# Cities: {len(tour)}\n")
        handle.write(f"# Total distance: {length:.4f}\n")
        handle.write("# order,city_index,city_name\n")
        for position, city in enumerate(tour):
            handle.write(f"{position},{city},{names[city]}\n")
        handle.write(f"{len(tour)},{tour[0]},{names[tour[0]]}  # return to start\n")
