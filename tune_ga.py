"""Parameter sweep for the Genetic Algorithm.

The assessment rewards *justified* parameter choices. This script measures
several (tournament_size, mutation_rate) combinations on the real 50-city
instance and writes the comparison to ``results/ga_tuning.csv``.

Each configuration is run with five seeds and summarised by mean best tour
length. The notebook's chosen configuration (k=7, mutation_rate=0.6) is
included so the table shows how it compares against alternatives.

Run from the project root::

    python tune_ga.py
"""

import time

import numpy as np
import pandas as pd

from src.genetic_algorithm import genetic_algorithm
from src.tsp import build_distance_matrix, load_cities

N_SEEDS = 5

coords = load_cities("data/cities.csv")
D = build_distance_matrix(coords)

tournament_sizes = [3, 5, 7, 9]
mutation_rates = [0.2, 0.4, 0.6, 0.8]

rows = []
for ts in tournament_sizes:
    for mr in mutation_rates:
        lengths, times = [], []
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()
            _, length, _, _ = genetic_algorithm(
                D, seed=seed,
                pop_size=100, generations=260,
                mutation_rate=mr, tournament_k=ts, elitism=5,
            )
            times.append(time.perf_counter() - t0)
            lengths.append(length)

        rows.append({
            "tournament_size": ts,
            "mutation_rate": mr,
            "mean_distance": np.mean(lengths),
            "std_distance": np.std(lengths),
            "mean_time_s": np.mean(times),
        })
        print(
            f"k={ts}  mut={mr:.1f}  "
            f"mean={np.mean(lengths):.2f}  std={np.std(lengths):.2f}  "
            f"time={np.mean(times):.2f}s"
        )

table = pd.DataFrame(rows).sort_values("mean_distance")
table.to_csv("results/ga_tuning.csv", index=False)

print(
    "\nRanked by mean tour length (lower is better):"
)
print(table.to_string(index=False))
print("\nWritten to results/ga_tuning.csv")
