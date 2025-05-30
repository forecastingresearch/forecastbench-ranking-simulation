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
    rank_by_diff_adj_brier,
    rank_by_peer_score,
    simulate_random_sampling,
    simulate_round_based,
    spearman_correlation,
    top_k_retention,
)

# EXPECTED RUNTIME: ~3-4 minutes on a standard laptop with N_SIMULATIONS = 1000

np.random.seed(20250527)

# Configuration
INPUT_FOLDER = "./data/raw"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"
REF_MODEL = "GPT-4 (zero shot)"
N_SIMULATIONS = 1000

# Parameters for random sampling
N_QUESTIONS_PER_MODEL = (
    125  # Ensures ~25 overlapping questions between two models given the dataset
)

# Parameters for round-based sampling
N_ROUNDS = 15
QUESTIONS_PER_ROUND = 100
MODELS_PER_ROUND_MEAN = 40
DATASET_WEIGHT = 0.5
SIMULATION_METHOD = "round_based"

# Define simulation methods
simulation_methods = {
    "random_sampling": (
        simulate_random_sampling,
        {"n_questions_per_model": N_QUESTIONS_PER_MODEL},
    ),
    "round_based": (
        simulate_round_based,
        {
            "n_rounds": N_ROUNDS,
            "questions_per_round": QUESTIONS_PER_ROUND,
            "models_per_round_mean": MODELS_PER_ROUND_MEAN,
        },
    ),
}


def validate_processed_data(df):
    """Validate the processed dataset for consistency and completeness."""
    print("\nValidating processed data...")

    # 1. Check for duplicates at [question_id, model] level
    duplicates = df.duplicated(subset=["question_id", "model"], keep=False)
    if duplicates.any():
        dup_data = df[duplicates].sort_values(["question_id", "model"])
        raise ValueError(
            f"Found {duplicates.sum()} duplicate entries at \
                [question_id, model] level:\n{dup_data.head(10)}"
        )
    print("✓ No duplicates at [question_id, model] level")

    # 2. Check perfect coverage: each model should predict every question
    models = df["model"].unique()
    questions = df["question_id"].unique()
    expected_entries = len(models) * len(questions)
    actual_entries = len(df)

    if actual_entries != expected_entries:
        # Find which model-question pairs are missing
        from itertools import product

        expected_pairs = set(product(models, questions))
        actual_pairs = set(zip(df["model"], df["question_id"]))
        missing_pairs = expected_pairs - actual_pairs

        if missing_pairs:
            missing_sample = list(missing_pairs)[:5]
            print(
                f"WARNING: Missing {len(missing_pairs)} model-question pairs. \
                    Examples: {missing_sample}"
            )
        else:
            print(
                f"WARNING: Found {actual_entries} entries, expected {expected_entries}"
            )
    else:
        print("✓ Perfect coverage: all models predict all questions")

    # 3. For market questions, check single forecast horizon
    df_market = df[df["question_type"] == "market"]
    if len(df_market) > 0:
        # Check if any question has multiple horizons by looking at unique combinations
        question_horizons = df_market.groupby(["source", "id"])["horizon"].nunique()
        multi_horizon = question_horizons[question_horizons > 1]

        if len(multi_horizon) > 0:
            print(
                f"WARNING: {len(multi_horizon)} market questions \
                    have multiple horizons:"
            )
            print(multi_horizon.head())
        else:
            print("✓ All market questions have single forecast horizon")

    # 4. Check for missing values in critical columns
    critical_columns = [
        "model",
        "question_id",
        "forecast",
        "resolved_to",
        "question_type",
    ]
    missing_values = df[critical_columns].isnull().sum()
    if missing_values.any():
        raise ValueError(
            f"Found missing values in critical \
                columns:\n{missing_values[missing_values > 0]}"
        )
    print("✓ No missing values in critical columns")

    # 5. Check forecast values are in [0, 1]
    invalid_forecasts = df[(df["forecast"] < 0) | (df["forecast"] > 1)]
    if len(invalid_forecasts) > 0:
        raise ValueError(
            f"Found {len(invalid_forecasts)} forecasts outside [0, 1] range"
        )
    print("✓ All forecasts are in valid range [0, 1]")

    # 6. Check resolved_to is binary
    unique_resolved = df["resolved_to"].unique()
    if not set(unique_resolved).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"resolved_to contains non-binary values: {unique_resolved}")
    print("✓ All resolved_to values are binary")

    # 7. Check question_type values
    valid_types = {"dataset", "market"}
    actual_types = set(df["question_type"].unique())
    if not actual_types.issubset(valid_types):
        raise ValueError(f"Invalid question types found: {actual_types - valid_types}")
    print("✓ All question_type values are valid")

    # 8. Check that only data from the first forecasting round (2024 July)
    # is present
    mask = pd.to_datetime(df["forecast_due_date"]) == "2024-07-21"
    if not mask.all():
        raise ValueError(
            f"Invalid forecast_due_date's found: \
                  {df.loc[~mask, "forecast_due_date"].values[0:5]}"
        )
    print("✓ Only data from 2024-07-21 present")

    # 9. Summary statistics
    print("\nDataset summary:")
    print(f"- Total entries: {len(df):,}")
    print(f"- Unique models: {len(models)}")
    print(f"- Unique questions: {len(questions)}")
    print(f"- Dataset questions: {(df['question_type'] == 'dataset').sum():,}")
    print(f"- Market questions: {(df['question_type'] == 'market').sum():,}")
    return True


def main():
    print("Loading data...")
    df = process_raw_data(f"{INPUT_FOLDER}/leaderboard_llm.pkl")
    df.to_csv(f"{PROCESSED_FOLDER}/processed_dataset.csv", index=False)

    # Load the processed dataset
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_dataset.csv")
    print(f"Loaded {len(df)} records")

    # Validate the processed data
    try:
        validate_processed_data(df)
        print("\n✅ Data validation passed!")
    except ValueError as e:
        print(f"\n❌ Data validation failed: {e}")
        sys.exit(1)

    # Define ranking methods, with the following tuple elements:
    # 1: ranking method name
    # 2: metric name
    # 3: is a lower score better
    # 4: Additional kwargs to the ranking function
    ranking_methods = {
        "Brier": (rank_by_brier, "avg_brier", True, {}),
        "Diff-Adj. Brier": (rank_by_diff_adj_brier, "avg_diff_adj_brier", True, {}),
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
