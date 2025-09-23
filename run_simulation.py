#!/usr/bin/env python
"""Run the ranking simulation."""

import json
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
    ranking_sanity_check,
    simulate_random_sampling,
    simulate_round_based,
    spearman_correlation,
    top_k_retention,
)

# EXPECTED RUNTIME: ~3-4 minutes per scenario on a standard
# laptop with N_SIMULATIONS = 1000; with 12 scenarios, total
# expected runtime is around 40 minutes.

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================
N_SIMULATIONS = 1000  # Number of simulations for each scenario
DATASET_WEIGHT = 0.5  # Weight for dataset vs market questions
RANDOM_SEED = 20250527  # Random seed for replicability

INPUT_FOLDER = "./data/raw"
DATAFILE_NAME = "llm_and_human_leaderboard_overall.pkl"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"

# =====================================================
# SIMULATION SCENARIOS
# =====================================================
# Add new scenarios to this list. Each scenario is a dictionary with:
# - name: identifier for output files
# - description: what this scenario tests
# - ref_model: reference model that answers all questions (used for simulation and BSS)
# - simulation_func: the simulation function to use
# - simulation_kwargs: parameters specific to that simulation function

SIMULATION_SCENARIOS = [
    {
        "name": "random_sampling_baseline",
        "description": "Random sampling with ~30 overlapping questions between models",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_random_sampling,
        "simulation_kwargs": {
            # With 473 questions to sample from, total epxected
            # number of overlapping questinos between two models is
            # 473 * (n_questions_per_model / 473) * (n_questions_per_model / 473) =
            "n_questions_per_model": 125,
        },
    },
    {
        "name": "round_based_baseline",
        "description": "Round-based sampling with ~30 overlapping questions per round, \
            and ~100 questions per model overall; retention between rounds is ~30%",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            # With 473 questions to sample from and 141 models
            # to sample from, the total expected number of overlapping
            # questions between two models is:
            # n_rounds * (models_per_round_mean / 141)
            #   * (models_per_round_mean - 1) / 141) * questions_per_round;
            # (-1) because we sample models without replacement
            # Total number of expected questions per model is
            # n_rounds * questions_per_round * (models_per_round_mean / 141)
            # Probability that a model participats in round R + 1, conditional
            # on participating in round R is:
            # models_per_round_mean / 141
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
        },
    },
    {
        "name": "random_sampling_GPT4_reference",
        "description": "Random sampling with GPT-4 as reference model",
        "ref_model": "GPT-4 (zero shot)",
        "simulation_func": simulate_random_sampling,
        "simulation_kwargs": {
            "n_questions_per_model": 125,
        },
    },
    {
        "name": "round_based_GPT4_reference",
        "description": "Round-based sampling with 25 overlapping questions per round",
        "ref_model": "GPT-4 (zero shot)",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
        },
    },
    {
        "name": "random_sampling_small_sample",
        "description": "Random sampling with a small sample size \
              with ~2 overlapping questions between models",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_random_sampling,
        "simulation_kwargs": {
            "n_questions_per_model": 30,
        },
    },
    {
        "name": "round_based_small_sample",
        "description": "Round-based sampling with a small sample size \
              with ~2 overlapping questinos between models",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 5,
            "questions_per_round": 5,
            "models_per_round_mean": 15,
        },
    },
    {
        "name": "random_sampling_always_half_ref",
        "description": "Random sampling using Always 0.5 as reference. \
            Included for sanity testing purposes.",
        "ref_model": "Always 0.5",
        "simulation_func": simulate_random_sampling,
        "simulation_kwargs": {
            "n_questions_per_model": 125,
        },
    },
    {
        "name": "round_based_model_drift",
        "description": "Round-based sampling with increasing model quality over time; \
            calibrated to yield a ~0.05 Brier improvement from first to final round in \
            average model performance.",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "skill_temperature": lambda round_id: (-15 + 30.0 / 14.0 * round_id),
            # Linear increase from -15 to 15 from round_id = 0 to 14
        },
    },
    {
        "name": "round_based_question_drift",
        "description": "Round-based sampling with some easier rounds; easier rounds \
            have questions with ~0.07 better Brier scores.",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "difficulty_temperature": lambda round_id: -10.0 if round_id <= 5 else 0.0,
        },
    },
    {
        "name": "round_based_with_model_persistence",
        "description": "Round-based sampling with baseline parameters \
              and 70% model persistence across rounds",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_with_model_persistence_and_drift",
        "description": "Round-based sampling with model drift \
              and 70% model persistence across rounds",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "skill_temperature": lambda round_id: (-15 + 30.0 / 14.0 * round_id),
            # Linear increase from -15 to 15 from round_id = 0 to 14
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_with_model_persistence_and_drift_sample_ample",
        "description": "Round-based sampling with model drift \
              and 70% model persistence across rounds, small sample",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 5,
            "questions_per_round": 25,
            "models_per_round_mean": 15,
            "skill_temperature": lambda round_id: (-15 + 30.0 / 14.0 * round_id),
            # Linear increase from -15 to 15 from round_id = 0 to 14
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_with_model_persistence_and_drift_GTP_4_reference",
        "description": "Round-based sampling with model drift \
              and 70% model persistence across rounds",
        "ref_model": "GPT-4 (zero shot)",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "skill_temperature": lambda round_id: (-15 + 30.0 / 14.0 * round_id),
            # Linear increase from -15 to 15 from round_id = 0 to 14
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_with_model_persistence_and_drift_real_world_sampling",
        "description": "Round-based sampling with model drift, \
              70% model persistence across rounds, and real-world data sampling",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
            "skill_temperature": lambda round_id: (-15 + 30.0 / 14.0 * round_id),
            # Linear increase from -15 to 15 from round_id = 0 to 14
            "model_persistence": 0.70,
            "fixed_dataset_market_question_sampling": True,
        },
    },
]

# Define RANKING METHODS (shared across all scenarios)
# NOTE: ref_model will be updated per scenario for BSS, to allow
# for easy changes to ref_model
BASE_RANKING_METHODS = {
    "Brier": (rank_by_brier, "avg_brier", True, {}),
    "Diff-Adj. Brier (w_mkt=0.00)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {},
    ),
    "Diff-Adj. Brier (w_mkt=0.25)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.25},
    ),
    "Diff-Adj. Brier (w_mkt=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.50},
    ),
    "Diff-Adj. Brier (w_mkt=0.75)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.75},
    ),
    "Diff-Adj. Brier (w_mkt=1.00)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 1.00},
    ),
    "Diff-Adj. Brier (w_mkt=0.00, fe_frac=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.00, "fe_models_frac": 0.5},
    ),
    "Diff-Adj. Brier (w_mkt=0.25, fe_frac=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.25, "fe_models_frac": 0.5},
    ),
    "Diff-Adj. Brier (w_mkt=0.50, fe_frac=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.50, "fe_models_frac": 0.5},
    ),
    "Diff-Adj. Brier (w_mkt=0.75, fe_frac=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.75, "fe_models_frac": 0.5},
    ),
    "Diff-Adj. Brier (w_mkt=1.00, fe_frac=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 1.00, "fe_models_frac": 0.5},
    ),
    "BSS (Pct.)": (
        rank_by_bss,
        "avg_bss",
        False,
        {"type": "percent"},
    ),
    "BSS (Abs.)": (
        rank_by_bss,
        "avg_bss",
        False,
        {"type": "absolute"},
    ),
    "Peer Score": (rank_by_peer_score, "avg_peer_score", False, {}),
}

# Define EVALUATION METRICS (shared across all scenarios)
EVALUATION_METRICS = {
    "Spearman": (spearman_correlation, {}),
    "Top-20 Retention": (top_k_retention, {"k": 20}),
    "Top-50 Retention": (top_k_retention, {"k": 50}),
    "Median Displacement": (median_displacement, {}),
    "Passed Sanity": (
        ranking_sanity_check,
        {
            "model_list": [
                "Superforecaster median forecast",
                "Public median forecast",
                "Random Uniform",
                "Always 0",
                "Always 1",
            ],
            "pct_point_tol": 0.25,
            "verbose": False,
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
    if not set(unique_resolved).issubset({int(0), int(1), 0.0, 1.0}):
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


def get_ranking_methods_for_scenario(scenario):
    """Create ranking methods with the correct ref_model for BSS calculations."""
    ranking_methods = {}

    for method_name, (func, metric, is_lower, kwargs) in BASE_RANKING_METHODS.items():
        if "BSS" in method_name:
            # Update kwargs with scenario's ref_model
            updated_kwargs = kwargs.copy()
            updated_kwargs["ref_model"] = scenario["ref_model"]
            ranking_methods[method_name] = (func, metric, is_lower, updated_kwargs)
        else:
            # Non-BSS methods don't need ref_model
            ranking_methods[method_name] = (func, metric, is_lower, kwargs)

    return ranking_methods


def save_scenario_config(scenario, results_folder):
    """Save scenario configuration for reproducibility."""
    # Create a serializable version of simulation_kwargs
    serializable_kwargs = {}
    for key, value in scenario["simulation_kwargs"].items():
        if callable(value):
            # Convert function to a string representation
            if hasattr(value, "__name__"):
                serializable_kwargs[key] = f"<function {value.__name__}>"
            else:
                # For lambda functions, try to get source code
                import inspect

                try:
                    serializable_kwargs[key] = (
                        f"<lambda: {inspect.getsource(value).strip()}>"
                    )
                except Exception:
                    serializable_kwargs[key] = "<function: not serializable>"
        else:
            serializable_kwargs[key] = value

    # Create a complete config including global parameters
    full_config = {
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "ref_model": scenario["ref_model"],
        "simulation_function": scenario["simulation_func"].__name__,
        "simulation_kwargs": serializable_kwargs,
        "global_parameters": {
            "n_simulations": N_SIMULATIONS,
            "dataset_weight": DATASET_WEIGHT,
            "random_seed": RANDOM_SEED,
        },
        "ranking_methods": list(BASE_RANKING_METHODS.keys()),
        "evaluation_metrics": list(EVALUATION_METRICS.keys()),
    }

    config_filename = f"{results_folder}/config_{scenario['name']}.json"
    with open(config_filename, "w") as f:
        json.dump(full_config, f, indent=2)

    return config_filename


def run_scenario(df, scenario):
    """Run a single simulation scenario and return results."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"Method: {scenario['simulation_func'].__name__}")
    print(f"Reference model: {scenario['ref_model']}")
    print(f"Simulations: {N_SIMULATIONS}")
    print(f"Parameters: {scenario['simulation_kwargs']}")

    # Save scenario configuration
    config_file = save_scenario_config(scenario, RESULTS_FOLDER)
    print(f"Saved configuration to: {config_file}")

    # Get ranking methods with correct ref_model
    ranking_methods = get_ranking_methods_for_scenario(scenario)

    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Run evaluation
    results, error_count = evaluate_ranking_methods(
        df=df,
        ranking_methods=ranking_methods,
        evaluation_metrics=EVALUATION_METRICS,
        simulation_func=scenario["simulation_func"],
        simulation_kwargs=scenario["simulation_kwargs"],
        n_simulations=N_SIMULATIONS,
        dataset_weight=DATASET_WEIGHT,
        ref_model=scenario["ref_model"],  # Used by simulation functions
    )

    # Print status with emoji
    if error_count == 0:
        print("\n✅ Scenario completed successfully!")
    else:
        print("\n❌ Scenario completed with {error_count} errors")

    return results, error_count


def main():
    print("Loading data...")
    df = process_raw_data(f"{INPUT_FOLDER}/{DATAFILE_NAME}")
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

    # Run all scenarios
    all_summaries = []

    for scenario in SIMULATION_SCENARIOS:
        # Run scenario
        results, _ = run_scenario(df, scenario)

        # Save detailed results
        results_filename = f"{RESULTS_FOLDER}/simulation_output_{scenario['name']}.csv"
        results.to_csv(results_filename, index=False)
        print(f"\nSaved detailed results to: {results_filename}")

        # Calculate and save summary
        summary = (
            results.groupby("method")
            .mean()[
                [
                    "Spearman",
                    "Top-20 Retention",
                    "Top-50 Retention",
                    "Median Displacement",
                    "Passed Sanity",
                ]
            ]
            .sort_values(by="Spearman", ascending=False)
        )

        # Add scenario metadata
        summary["scenario"] = scenario["name"]

        # Save individual summary
        summary_filename = f"{RESULTS_FOLDER}/summary_{scenario['name']}.csv"
        summary.to_csv(summary_filename)
        print(f"Saved summary to: {summary_filename}")

        print(f"\nResults for {scenario['name']}:")
        print(
            summary[
                [
                    "Spearman",
                    "Top-20 Retention",
                    "Top-50 Retention",
                    "Median Displacement",
                    "Passed Sanity",
                ]
            ]
        )

        # Collect for combined summary
        summary_reset = summary.reset_index()
        all_summaries.append(summary_reset)

    # Save combined summary
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(f"{RESULTS_FOLDER}/all_scenarios_summary.csv", index=False)
    print(f"\n\nSaved combined summary to: {RESULTS_FOLDER}/all_scenarios_summary.csv")


if __name__ == "__main__":
    main()
