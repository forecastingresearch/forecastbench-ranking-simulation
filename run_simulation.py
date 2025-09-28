#!/usr/bin/env python
"""Run the ranking simulation."""

import json
import os
import shutil
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

# EXPECTED RUNTIME: ~12 hours with N_SIMULATIONS = 1000.

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================
N_SIMULATIONS = 2  # Number of simulations for each scenario
DATASET_WEIGHT = 0.5  # Weight for dataset vs market questions
RANDOM_SEED = 20250527  # Random seed for replicability
FE_MODELS_FRAC = 0.5  # Fraction of models used for FE estimatino

INPUT_FOLDER = "./data/raw"
DATAFILE_NAME = "llm_and_human_leaderboard_overall.pkl"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"

# If True, delete any existing output from previous runs
CLEANUP_OUTPUT = True

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
        "description": "Random sampling",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_random_sampling,
        "simulation_kwargs": {
            "n_questions_per_model": 500,
        },
    },
    {
        "name": "round_based_baseline",
        "description": "Round-based sampling, baseline",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 30,
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_drift",
        "description": "Round-based sampling, model and question drift",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 30,
            "model_persistence": 0.70,
            # Linear model skill temparature increase from -15 to 15 from round_id = 0
            # to 14. Calibrated to yield an improvement of ~0.06 in the avg. Brier
            # score from the first to the final round
            "skill_temperature": lambda round_id: (-15 + 30.0 / 9.0 * round_id),
            # Even rounds are more difficult than odd rounds. Calibrated so that
            # the difference between rounds is ~0.09 in the avg. Brier score
            "difficulty_temperature": lambda round_id: (
                -5.0 if round_id % 2 == 0 else 5.0
            ),
        },
    },
    {
        "name": "round_based_low_models",
        "description": "Round-based sampling, low number of models",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 10,
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_high_models",
        "description": "Round-based sampling, high number of models",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 50,
            "model_persistence": 0.70,
        },
    },
    {
        "name": "round_based_high_discrepancy",
        "description": "Round-based sampling, high-discrepancy questions",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 30,
            "model_persistence": 0.70,
        },
        "filter_func": lambda df: (df["question_type"] == "dataset")
        | (
            df["question_market_discrepancy"] > df["median_question_market_discrepancy"]
        ),
    },
    {
        "name": "round_based_baseline_low_discrepancy",
        "description": "Round-based sampling, low-discrepancy questions",
        "ref_model": "Naive Forecaster",
        "simulation_func": simulate_round_based,
        "simulation_kwargs": {
            "n_rounds": 10,
            "questions_per_round": 500,
            "models_per_round_mean": 30,
            "model_persistence": 0.70,
        },
        "filter_func": lambda df: (df["question_type"] == "dataset")
        | (
            df["question_market_discrepancy"] < df["median_question_market_discrepancy"]
        ),
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
        {"market_weight": 0.00, "fe_models_frac": FE_MODELS_FRAC},
    ),
    "Diff-Adj. Brier (w_mkt=0.25)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.25, "fe_models_frac": FE_MODELS_FRAC},
    ),
    "Diff-Adj. Brier (w_mkt=0.50)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.50, "fe_models_frac": FE_MODELS_FRAC},
    ),
    "Diff-Adj. Brier (w_mkt=0.75)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 0.75, "fe_models_frac": FE_MODELS_FRAC},
    ),
    "Diff-Adj. Brier (w_mkt=1.00)": (
        rank_by_diff_adj_brier,
        "avg_diff_adj_brier",
        True,
        {"market_weight": 1.00, "fe_models_frac": FE_MODELS_FRAC},
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


def cleanup_output_directories():
    """Clean up existing output directories to ensure fresh results"""
    directories_to_clean = [PROCESSED_FOLDER, RESULTS_FOLDER]

    for directory in directories_to_clean:
        if os.path.exists(directory):
            print(f"Cleaning directory: {directory}...", end="", flush=True)
            shutil.rmtree(directory)
            print(" ✅")

        # Recreate the directory
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}", end="", flush=True)
        print(" ✅")


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


def run_scenario(df, scenario, filter_msg=None):
    """Run a single simulation scenario and return results."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"Method: {scenario['simulation_func'].__name__}")
    print(f"Reference model: {scenario['ref_model']}")
    print(f"Simulations: {N_SIMULATIONS}")
    print(f"Parameters: {scenario['simulation_kwargs']}")
    if filter_msg:
        print(filter_msg)

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


def generate_latex_tables(combined_summary):
    """Generate LaTeX tables for the simulation results."""
    print("\n" + "=" * 60)
    print("Generating LaTeX tables...")

    # Define scenario groups
    basic_scenarios = [
        "random_sampling_baseline",
        "round_based_baseline",
        "round_based_drift",
    ]

    complex_scenarios = [
        "round_based_low_models",
        "round_based_high_models",
        "round_based_high_discrepancy",
        "round_based_baseline_low_discrepancy",
    ]

    # Define method order and display names
    methods_order = [
        "Brier",
        "BSS (Pct.)",
        "BSS (Abs.)",
        "Peer Score",
        "Diff-Adj. Brier (w_mkt=0.00)",
        "Diff-Adj. Brier (w_mkt=0.25)",
        "Diff-Adj. Brier (w_mkt=0.50)",
        "Diff-Adj. Brier (w_mkt=0.75)",
        "Diff-Adj. Brier (w_mkt=1.00)",
    ]

    # Method display names for table
    method_display_names = {
        "Brier": "Raw Brier",
        "BSS (Pct.)": "BSS (Pct.)",
        "BSS (Abs.)": "BSS (Abs.)",
        "Peer Score": "Peer Score",
        "Diff-Adj. Brier (w_mkt=0.00)": "Diff.-Adj. Brier ($w_{\\text{mkt}} = 0.00$)",
        "Diff-Adj. Brier (w_mkt=0.25)": "Diff.-Adj. Brier ($w_{\\text{mkt}} = 0.25$)",
        "Diff-Adj. Brier (w_mkt=0.50)": "Diff.-Adj. Brier ($w_{\\text{mkt}} = 0.50$)",
        "Diff-Adj. Brier (w_mkt=0.75)": "Diff.-Adj. Brier ($w_{\\text{mkt}} = 0.75$)",
        "Diff-Adj. Brier (w_mkt=1.00)": "Diff.-Adj. Brier ($w_{\\text{mkt}} = 1.00$)",
    }

    # Scenario display names
    scenario_display_names = {
        "random_sampling_baseline": "Random sampling",
        "round_based_baseline": "Round-based sampling",
        "round_based_drift": "Round-based sampling with drift",
        "round_based_low_models": "Round-based sampling, low models",
        "round_based_high_models": "Round-based sampling, high models",
        "round_based_high_discrepancy": "Round-based sampling, high discrepancy",
        "round_based_baseline_low_discrepancy": "Round-based sampling, low discrepancy",
    }

    def create_latex_table(scenarios, table_title, filename):
        """Create LaTeX table for given scenarios."""
        # Filter data for these scenarios
        df_filtered = combined_summary[
            combined_summary["scenario"].isin(scenarios)
        ].copy()

        if len(df_filtered) == 0:
            print(f"Warning: No data found for scenarios: {scenarios}")
            return

        latex_content = []
        latex_content.append("\\begin{table}[ht]")
        latex_content.append("\\centering")
        caption_text = (
            table_title + ". \\emph{Spearman}: Spearman rank correlation coefficient "
            "relative to the true ranking. "
            "\\emph{Top-20}: Top-20 retention rate relative to the true ranking. "
            "\\emph{Top-50}: Top-50 retention rate relative to the true ranking. "
            "\\emph{Med. Disp.}: Median displacement relative to the true ranking, "
            "in ranks. "
            "See Section~\\ref{sec:simulation_framework} for additional details."
        )
        latex_content.append("\\caption{" + caption_text + "}")

        # Create column specification: Method name + 4 metrics
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")

        # Create header
        latex_content.append(" & Spearman & Top-20 & Top-50 & Med. Disp. \\\\")
        latex_content.append("\\midrule")

        # Group scenarios and add data rows
        for i, scenario in enumerate(scenarios):
            # Add scenario header
            scenario_display = scenario_display_names.get(scenario, scenario)
            latex_content.append(f"\\textit{{{scenario_display}}} & & & & \\\\")

            # Add methods for this scenario
            for method in methods_order:
                scenario_data = df_filtered[
                    (df_filtered["scenario"] == scenario)
                    & (df_filtered["method"] == method)
                ]

                if len(scenario_data) == 0:
                    continue

                data = scenario_data.iloc[0]
                method_display = method_display_names.get(method, method)

                spearman = (
                    f"{data['Spearman']:.2f}" if pd.notna(data["Spearman"]) else "-"
                )
                top20 = (
                    f"{data['Top-20 Retention']:.2f}"
                    if pd.notna(data["Top-20 Retention"])
                    else "-"
                )
                top50 = (
                    f"{data['Top-50 Retention']:.2f}"
                    if pd.notna(data["Top-50 Retention"])
                    else "-"
                )
                med_disp = (
                    f"{data['Median Displacement']:.0f}"
                    if pd.notna(data["Median Displacement"])
                    else "-"
                )

                latex_content.append(
                    f"{method_display} & {spearman} & {top20} "
                    f"& {top50} & {med_disp} \\\\"
                )

            # Add spacing between scenarios (except for the last one)
            if i < len(scenarios) - 1:
                latex_content.append("\\midrule")

        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")

        # Write to file
        filepath = f"{RESULTS_FOLDER}/{filename}"
        with open(filepath, "w") as f:
            f.write("\n".join(latex_content))

        print(f"LaTeX table saved to: {filepath}")
        return "\n".join(latex_content)

    # Generate both tables
    create_latex_table(
        basic_scenarios,
        "Simulation Results: Basic Scenarios",
        "latex_table_basic_scenarios.tex",
    )

    create_latex_table(
        complex_scenarios,
        "Simulation Results: Complex Scenarios",
        "latex_table_complex_scenarios.tex",
    )


def main():
    # Clean up previous outputs before starting
    if CLEANUP_OUTPUT:
        cleanup_output_directories()

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
        # Apply data filtering if specified
        if "filter_func" in scenario:
            mask = scenario["filter_func"](df)
            df_scenario = df[mask].copy()
            filter_msg = f"Applied filter: {len(df_scenario)} records (from {len(df)})"
        else:
            df_scenario = df.copy()
            filter_msg = None

        # Run scenario
        results, _ = run_scenario(
            df=df_scenario, scenario=scenario, filter_msg=filter_msg
        )

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

    # Generate LaTeX tables
    generate_latex_tables(combined_summary)


if __name__ == "__main__":
    main()
