"""Small parameter sweep for the Genetic Algorithm.

The assessment rewards *justified* parameter choices. Rather than assert
that a configuration is good, this script measures several configurations
on the real 50-city instance and writes the comparison to
``results/ga_tuning.csv``, so the report can cite evidence for the
settings that were adopted.

Each configuration is run with several seeds and summarised by the mean
best tour length. Run time is also recorded, because a configuration that
is better only by spending far more compute is not a fair winner.

Run from the project root:  python tune_ga.py
"""

import time

import pandas as pd

from src.genetic_algorithm import (
    genetic_algorithm,
    pmx_crossover,
    swap_mutation,
)
from src.tsp import build_distance_matrix, load_cities

N_SEEDS = 5

coords = load_cities("data/cities.csv")
dist = build_distance_matrix(coords)

# Each entry: (label, kwargs). The first is the original baseline.
configs = [
    ("Baseline (pop=100, gen=500, mut=0.2)",
     dict(population_size=100, generations=500, mutation_rate=0.2)),

    ("Larger population (pop=200, gen=500)",
     dict(population_size=200, generations=500, mutation_rate=0.2)),

    ("Longer run (pop=100, gen=1000)",
     dict(population_size=100, generations=1000, mutation_rate=0.2)),

    ("Higher mutation (mut=0.4)",
     dict(population_size=100, generations=500, mutation_rate=0.4)),

    ("Stronger selection (k=7, elitism=5)",
     dict(population_size=100, generations=500, mutation_rate=0.2,
          tournament_size=7, elitism=5)),

    ("Swap mutation instead of inversion",
     dict(population_size=100, generations=500, mutation_rate=0.2,
          mutation_fn=swap_mutation)),

    ("PMX crossover instead of OX1",
     dict(population_size=100, generations=500, mutation_rate=0.2,
          crossover_fn=pmx_crossover)),

    ("Tuned (pop=200, gen=1000, mut=0.3, k=7, elit=5)",
     dict(population_size=200, generations=1000, mutation_rate=0.3,
          tournament_size=7, elitism=5)),
]

rows = []
for label, kwargs in configs:
    lengths, times = [], []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()
        _, length, _ = genetic_algorithm(dist, seed=seed, **kwargs)
        times.append(time.perf_counter() - t0)
        lengths.append(length)

    series = pd.Series(lengths)
    rows.append({
        "configuration": label,
        "mean_length": series.mean(),
        "std_length": series.std(),
        "best_length": series.min(),
        "mean_time_s": sum(times) / len(times),
    })
    print(f"{label:48s} mean={series.mean():7.2f}  "
          f"std={series.std():6.2f}  time={sum(times)/len(times):.2f}s")

table = pd.DataFrame(rows).sort_values("mean_length")
table.to_csv("results/ga_tuning.csv", index=False)

print("\nRanked by mean tour length (lower is better):")
print(table.to_string(index=False))
print("\nWritten to results/ga_tuning.csv")
