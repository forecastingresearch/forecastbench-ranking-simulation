#!/usr/bin/env python
"""Run the ranking simulation."""

import sys

sys.path.append("src")
import numpy as np
import pandas as pd

from ranking_sim import (
    evaluate_ranking_methods,
    median_displacement,
    process_raw_data,
    rank_by_brier,
    rank_by_bss,
    rank_by_peer_score,
    simulate_random_sampling,
    spearman_correlation,
    top_k_retention,
)

# EXPECTED RUNTIME: ~2 minutes on a standard laptop with N_SIMULATIONS = 1000

np.random.seed(20250527)

# Configuration
INPUT_FOLDER = "./data/raw"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"
REF_MODEL = "GPT-4 (zero shot)"
N_SIMULATIONS = 1000
N_QUESTIONS_PER_MODEL = (
    125  # Ensures ~25 overlapping questions between two models given the dataset
)
DATASET_WEIGHT = 0.5
SIMULATION_METHOD = "random_sampling"

# Define simulation methods
simulation_methods = {
    "random_sampling": (
        simulate_random_sampling,
        {"n_questions_per_model": N_QUESTIONS_PER_MODEL},
    )
}


def main():
    print("Loading data...")
    df = process_raw_data(f"{INPUT_FOLDER}/leaderboard_human.pkl")
    df.to_csv(f"{PROCESSED_FOLDER}/processed_dataset.csv", index=False)

    # Load the processed dataset
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_dataset.csv")
    print(f"Loaded {len(df)} records")

    # Define ranking methods, with the following tuple elements:
    # 1: ranking method name
    # 2: metric name
    # 3: is a lower score better
    # 4: Additional kwargs to the ranking function
    ranking_methods = {
        "Brier": (rank_by_brier, "avg_brier", True, {}),
        "BSS": (rank_by_bss, "avg_bss", False, {"ref_model": REF_MODEL}),
        "Peer Score": (rank_by_peer_score, "avg_peer_score", False, {}),
    }

    # Define evaluation metrics, with the following tuple elements:
    # 1: evaluation metric name
    # 2: additional kwargs to the evaluation metric function
    evaluation_metrics = {
        "Spearman": (spearman_correlation, {}),
        "Top-20 Retention": (top_k_retention, {"k": 20}),
        "Top-50 Retention": (top_k_retention, {"k": 50}),
        "Median Displacement": (median_displacement, {}),
    }

    # Get simulation method
    simulation_func, simulation_kwargs = simulation_methods[SIMULATION_METHOD]
    print(f"Using simulation method: {SIMULATION_METHOD}")

    print(f"Running {N_SIMULATIONS} simulations...")
    results = evaluate_ranking_methods(
        df=df,
        ranking_methods=ranking_methods,
        evaluation_metrics=evaluation_metrics,
        simulation_func=simulation_func,
        simulation_kwargs=simulation_kwargs,
        n_simulations=N_SIMULATIONS,
        dataset_weight=DATASET_WEIGHT,
        ref_model=REF_MODEL,
    )

    # Save results
    results.to_csv(f"{RESULTS_FOLDER}/simulation_output.csv", index=False)

    # Print summary
    summary = (
        results.groupby("method")
        .mean()[
            ["Spearman", "Top-20 Retention", "Top-50 Retention", "Median Displacement"]
        ]
        .sort_values(by="Spearman", ascending=False)
    )

    print("\nResults:")
    print(summary)


if __name__ == "__main__":
    main()
