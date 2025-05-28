import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("../src")

from ranking_sim import (
    brier_score,
    combine_rankings,
    evaluate_ranking_methods,
    median_displacement,
    rank_by_brier,
    rank_by_bss,
    rank_by_peer_score,
    rank_with_weighting,
    simulate_random_sampling,
    simulate_round_based,
    spearman_correlation,
    top_k_retention,
)


def test_brier_score():
    """Test that Brier score is calculated correctly."""
    df = pd.DataFrame({"forecast": [0.7, 0.3, 1.0], "resolved_to": [1, 0, 1]})
    scores = brier_score(df)
    assert np.isclose(scores[0], 0.09)
    assert np.isclose(scores[1], 0.09)
    assert np.isclose(scores[2], 0.0)


def test_rank_by_brier():
    """Test basic ranking by Brier score."""
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C"],
            "forecast": [0.9, 0.1, 0.5, 0.5, 1.0, 0.0],
            "resolved_to": [1, 0, 1, 0, 1, 0],
        }
    )
    rankings = rank_by_brier(df)
    assert len(rankings) == 3  # Three models
    assert "rank" in rankings.columns
    assert rankings[rankings["model"] == "A"]["rank"].values[0] == 2
    assert rankings[rankings["model"] == "C"]["rank"].values[0] == 1


def test_rank_by_bss():
    """Test basic ranking by Brier Skill Score (BSS)."""
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C"],
            "forecast": [0.9, 0.1, 0.5, 0.5, 1.0, 0.0],
            "question_id": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "resolved_to": [1, 0, 1, 0, 1, 0],
        }
    )
    rankings = rank_by_bss(df, ref_model="A")
    assert len(rankings) == 3  # Three models
    assert "rank" in rankings.columns
    assert rankings[rankings["model"] == "A"]["rank"].values[0] == 2
    assert rankings[rankings["model"] == "C"]["rank"].values[0] == 1
    mask = rankings["model"] == "B"
    assert np.isclose(rankings.loc[mask, "avg_bss"].values[0], -24.0)


def test_rank_by_bss_ref_model_missing():
    """Test basic ranking by Brier Skill Score (BSS)."""
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C"],
            "forecast": [0.9, np.nan, 0.5, 0.5, 1.0, 0.0],
            "question_id": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "resolved_to": [1, 0, 1, 0, 1, 0],
        }
    )
    rankings = rank_by_bss(df, ref_model="A")
    assert len(rankings) == 3  # Three models
    assert "rank" in rankings.columns
    assert rankings[rankings["model"] == "A"]["rank"].values[0] == 2
    assert rankings[rankings["model"] == "B"]["rank"].values[0] == 3
    assert rankings[rankings["model"] == "C"]["rank"].values[0] == 1
    mask = rankings["model"] == "A"
    assert np.isclose(rankings.loc[mask, "avg_bss"].values[0], 0.0, atol=0.0001)
    mask = rankings["model"] == "C"
    print(rankings.loc[mask, "avg_bss"].values[0])
    assert np.isclose(rankings.loc[mask, "avg_bss"].values[0], 1.0)


def test_rank_by_peer_score():
    """Test basic ranking by Brier peer score"""
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C"],
            "forecast": [0.9, 0.1, 0.5, 0.5, 1.0, 0.0],
            "question_id": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "resolved_to": [1, 0, 1, 0, 1, 0],
        }
    )
    rankings = rank_by_peer_score(df)
    assert len(rankings) == 3  # Three models
    assert "rank" in rankings.columns
    assert rankings[rankings["model"] == "A"]["rank"].values[0] == 2
    assert rankings[rankings["model"] == "C"]["rank"].values[0] == 1
    mask = rankings["model"] == "B"
    assert np.isclose(
        rankings.loc[mask, "avg_peer_score"].values[0], -0.1633, atol=0.0001
    )


def test_rank_by_peer_score_missing():
    """Test basic ranking by Brier peer score"""
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C"],
            "forecast": [0.9, np.nan, np.nan, 0.5, np.nan, 0.0],
            "question_id": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "resolved_to": [1, 0, 1, 0, 1, 0],
        }
    )
    rankings = rank_by_peer_score(df)
    mask = rankings["model"] == "A"
    assert np.isclose(rankings.loc[mask, "avg_peer_score"].values[0], 0.0)
    mask = rankings["model"] == "B"
    assert np.isclose(rankings.loc[mask, "avg_peer_score"].values[0], -0.125)
    mask = rankings["model"] == "C"
    assert np.isclose(rankings.loc[mask, "avg_peer_score"].values[0], 0.125)


def test_spearman_correlation():
    """Test Spearman correlation calculation."""
    # Perfect correlation (same ranking)
    df_true = pd.DataFrame({"model": ["A", "B", "C", "D"], "rank_true": [1, 2, 3, 4]})
    df_sim = pd.DataFrame({"model": ["A", "B", "C", "D"], "rank_sim": [1, 2, 3, 4]})
    corr = spearman_correlation(df_true, df_sim)
    assert np.isclose(corr, 1.0)

    # Perfect negative correlation (reversed ranking)
    df_sim_reversed = pd.DataFrame(
        {"model": ["A", "B", "C", "D"], "rank_sim": [4, 3, 2, 1]}
    )
    corr_neg = spearman_correlation(df_true, df_sim_reversed)
    assert np.isclose(corr_neg, -1.0)

    # Partial correlation
    df_sim_partial = pd.DataFrame(
        {"model": ["A", "B", "C", "D"], "rank_sim": [1, 3, 2, 4]}  # B and C swapped
    )
    corr_partial = spearman_correlation(df_true, df_sim_partial)
    assert np.isclose(corr_partial, 0.8)


def test_top_k_retention():
    """Test top-k retention calculation."""
    df_true = pd.DataFrame(
        {"model": ["A", "B", "C", "D", "E"], "rank_true": [1, 2, 3, 4, 5]}
    )

    # All top-3 models retained
    df_sim_all = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [2, 1, 3, 4, 5],  # Top 3 are still A, B, C
        }
    )
    retention = top_k_retention(df_true, df_sim_all, k=3)
    assert retention == 1.0

    # Only 2 out of top-3 retained
    df_sim_partial = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [1, 2, 4, 3, 5],  # C dropped out, D entered top 3
        }
    )
    retention = top_k_retention(df_true, df_sim_partial, k=3)
    assert np.isclose(retention, 2 / 3)

    # Only 1 out of top-3 retained
    df_sim_partial_2 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [5, 4, 3, 2, 1],  # Only C remained
        }
    )
    retention = top_k_retention(df_true, df_sim_partial_2, k=3)
    assert np.isclose(retention, 1 / 3)

    # None retained
    df_sim_none = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [5, 4, 3, 2, 1],  # Only C remained
        }
    )
    retention = top_k_retention(df_true, df_sim_none, k=2)
    assert retention == 0.0


def test_median_displacement():
    """Test median displacement calculation."""
    df_true = pd.DataFrame(
        {"model": ["A", "B", "C", "D", "E"], "rank_true": [1, 2, 3, 4, 5]}
    )

    # No displacement
    df_sim_same = pd.DataFrame(
        {"model": ["A", "B", "C", "D", "E"], "rank_sim": [1, 2, 3, 4, 5]}
    )
    displacement = median_displacement(df_true, df_sim_same)
    assert displacement == 0.0

    # Uniform displacement of 1
    df_sim_shift = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [2, 3, 4, 5, 1],  # Each moved by 1
        }
    )
    displacement = median_displacement(df_true, df_sim_shift)
    assert displacement == 1.0

    # Mixed displacements
    df_sim_mixed = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [1, 4, 3, 2, 5],
            # Displacements: A=0, B=2, C=0, D=2, E=0
            # Median of [0, 0, 0, 2, 2] = 0
        }
    )
    displacement = median_displacement(df_true, df_sim_mixed)
    assert displacement == 0.0

    # Different mixed case
    df_sim_mixed2 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [3, 1, 2, 5, 4],
            # Displacements: A=2, B=1, C=1, D=1, E=1
            # Median of [1, 1, 1, 1, 2] = 1
        }
    )
    displacement = median_displacement(df_true, df_sim_mixed2)
    assert displacement == 1.0


def test_combine_rankings():
    """Test combining dataset and market rankings with weighting."""
    # Dataset rankings
    df_dataset = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "avg_brier": [0.1, 0.2, 0.3],  # A is best
            "rank": [1, 2, 3],
        }
    )

    # Market rankings
    df_market = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "avg_brier": [0.3, 0.1, 0.2],  # B is best
            "rank": [3, 1, 2],
        }
    )

    # Test 50-50 weighting
    combined = combine_rankings(
        df_dataset,
        df_market,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    # Check weighted metrics
    # A: 0.5 * 0.1 + 0.5 * 0.3 = 0.2
    # B: 0.5 * 0.2 + 0.5 * 0.1 = 0.15
    # C: 0.5 * 0.3 + 0.5 * 0.2 = 0.25
    assert np.isclose(
        combined[combined["model"] == "A"]["avg_brier_weighted"].values[0], 0.2
    )
    assert np.isclose(
        combined[combined["model"] == "B"]["avg_brier_weighted"].values[0], 0.15
    )
    assert np.isclose(
        combined[combined["model"] == "C"]["avg_brier_weighted"].values[0], 0.25
    )

    # Check rankings (B=1, A=2, C=3)
    assert combined[combined["model"] == "B"]["rank"].values[0] == 1
    assert combined[combined["model"] == "A"]["rank"].values[0] == 2
    assert combined[combined["model"] == "C"]["rank"].values[0] == 3

    # Test 100% dataset weight
    combined_dataset = combine_rankings(
        df_dataset,
        df_market,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=1.0,
    )

    # Should match dataset rankings
    assert combined_dataset[combined_dataset["model"] == "A"]["rank"].values[0] == 1
    assert combined_dataset[combined_dataset["model"] == "B"]["rank"].values[0] == 2
    assert combined_dataset[combined_dataset["model"] == "C"]["rank"].values[0] == 3

    # Test with missing model in market (outer join)
    df_market_missing = pd.DataFrame(
        {"model": ["A", "B"], "avg_brier": [0.3, 0.1], "rank": [2, 1]}  # C is missing
    )

    combined_missing = combine_rankings(
        df_dataset,
        df_market_missing,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    # C should still appear with its dataset value used for both
    assert len(combined_missing) == 3
    assert "C" in combined_missing["model"].values
    # C's weighted metric should be 0.3 (using dataset value for both)
    assert np.isclose(
        combined_missing[combined_missing["model"] == "C"]["avg_brier_weighted"].values[
            0
        ],
        0.3,
    )
    assert np.isclose(
        combined_missing[combined_missing["model"] == "C"]["avg_brier_market"].values[
            0
        ],
        0.3,
    )


def test_rank_with_weighting():
    """Test ranking with dataset/market weighting."""
    # Create mixed dataset
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
            "question_id": [
                "d1",
                "d2",
                "d1",
                "d2",
                "d1",
                "d2",
                "m1",
                "m2",
                "m1",
                "m2",
                "m1",
                "m2",
            ],
            "question_type": ["dataset"] * 6 + ["market"] * 6,
            "forecast": [0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.5, 0.5, 0.9, 0.1],
            "resolved_to": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    # Calculate expected Brier scores by hand
    # Dataset - Model A: (0.9-1)² + (0.1-0)² = 0.01 + 0.01 = 0.02, avg = 0.01
    # Dataset - Model B: (0.7-1)² + (0.3-0)² = 0.09 + 0.09 = 0.18, avg = 0.09
    # Dataset - Model C: (0.6-1)² + (0.4-0)² = 0.16 + 0.16 = 0.32, avg = 0.16

    # Market - Model A: (0.8-1)² + (0.2-0)² = 0.04 + 0.04 = 0.08, avg = 0.04
    # Market - Model B: (0.5-1)² + (0.5-0)² = 0.25 + 0.25 = 0.50, avg = 0.25
    # Market - Model C: (0.9-1)² + (0.1-0)² = 0.01 + 0.01 = 0.02, avg = 0.01

    # Test with 50-50 weighting
    result = rank_with_weighting(
        df=df,
        ranking_func=rank_by_brier,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    # Expected weighted scores:
    # A: 0.5 * 0.01 + 0.5 * 0.04 = 0.025
    # B: 0.5 * 0.09 + 0.5 * 0.25 = 0.17
    # C: 0.5 * 0.16 + 0.5 * 0.01 = 0.085

    # Check weighted metrics
    assert np.isclose(
        result[result["model"] == "A"]["avg_brier_weighted"].values[0], 0.025
    )
    assert np.isclose(
        result[result["model"] == "B"]["avg_brier_weighted"].values[0], 0.17
    )
    assert np.isclose(
        result[result["model"] == "C"]["avg_brier_weighted"].values[0], 0.085
    )

    # Check rankings (A=1, C=2, B=3)
    assert result[result["model"] == "A"]["rank"].values[0] == 1
    assert result[result["model"] == "C"]["rank"].values[0] == 2
    assert result[result["model"] == "B"]["rank"].values[0] == 3

    # Test error when question_type is missing
    df_no_type = df.drop("question_type", axis=1)
    with pytest.raises(ValueError, match="question_type not found"):
        rank_with_weighting(
            df=df_no_type,
            ranking_func=rank_by_brier,
            metric_name="avg_brier",
            is_lower_metric_better=True,
        )

    # Test with 100% market weight
    result_market = rank_with_weighting(
        df=df,
        ranking_func=rank_by_brier,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.0,
    )

    # Should match market rankings (C=1, A=2, B=3)
    assert result_market[result_market["model"] == "C"]["rank"].values[0] == 1
    assert result_market[result_market["model"] == "A"]["rank"].values[0] == 2
    assert result_market[result_market["model"] == "B"]["rank"].values[0] == 3


def test_simulate_random_sampling():
    """Test dataset simulation with controlled overlap."""
    # Create a small dataset
    df = pd.DataFrame(
        {
            "model": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "question_id": ["q1", "q1", "q1", "q2", "q2", "q2", "q3", "q3", "q3"],
            "question_type": ["dataset"] * 9,
            "forecast": [0.7, 0.8, 0.9] * 3,
            "resolved_to": [1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )

    # Set seed for reproducibility
    np.random.seed(42)

    # Test with 20% overlap
    df_sim = simulate_random_sampling(df, n_questions_per_model=2, ref_model="A")

    # Check that ref model A has all questions
    a_questions = df_sim[df_sim["model"] == "A"]["question_id"].unique()
    assert len(a_questions) == 3  # All 3 questions
    assert set(a_questions) == {"q1", "q2", "q3"}

    # Check that other models have fewer questions
    b_questions = df_sim[df_sim["model"] == "B"]["question_id"].values
    c_questions = df_sim[df_sim["model"] == "C"]["question_id"].values
    assert len(b_questions) == 2
    assert len(c_questions) == 2

    # Check that all data is preserved correctly
    for _, row in df_sim.iterrows():
        # Find corresponding row in original
        mask = (df["model"] == row["model"]) & (df["question_id"] == row["question_id"])
        orig_row = df[mask].iloc[0]
        assert row["forecast"] == orig_row["forecast"]
        assert row["resolved_to"] == orig_row["resolved_to"]
        assert row["question_type"] == orig_row["question_type"]

    # Test with n_questions_per_model=3
    df_sim_full = simulate_random_sampling(df, n_questions_per_model=3, ref_model="A")

    # Each model should have all 3 questions
    for model in ["A", "B", "C"]:
        model_samples = len(df_sim_full[df_sim_full["model"] == model])
        assert model_samples == 3

    # Test that ref_model must exist
    with pytest.raises(ValueError, match="Reference model not provided"):
        simulate_random_sampling(df, n_questions_per_model=2, ref_model="NonExistent")


def test_simulate_random_sampling_overlap():
    """Test that overlap between models matches theoretical expectations."""
    np.random.seed(42)

    # Create a large dataset
    n_questions = 100

    # Create dummy data
    models = ["A", "B", "RefModel"]
    questions = [f"q{i}" for i in range(n_questions)]

    # Create all combinations
    data = []
    for model in models:
        for q in questions:
            data.append(
                {
                    "model": model,
                    "question_id": q,
                    "question_type": "dataset",
                    "forecast": 0.5,
                    "resolved_to": 1,
                }
            )
    df = pd.DataFrame(data)

    # Test parameters
    n_questions_per_model = 20
    n_simulations = 500

    # Track overlap across simulations
    overlaps = []

    for _ in range(n_simulations):
        df_sim = simulate_random_sampling(
            df, n_questions_per_model, ref_model="RefModel"
        )

        # Get questions answered by each model
        a_questions = set(df_sim[df_sim["model"] == "A"]["question_id"].unique())
        b_questions = set(df_sim[df_sim["model"] == "B"]["question_id"].unique())

        # Count overlap
        overlap = len(a_questions & b_questions)
        overlaps.append(overlap)

    # Calculate empirical statistics
    empirical_mean_overlap = np.mean(overlaps)

    # Calculate theoretical values
    # Probability that a specific question is selected by one model
    p_selected = 1 - (1 - 1 / n_questions) ** n_questions_per_model

    # Probability that both models select a specific question (independent sampling)
    p_both_select = p_selected**2

    # Expected number of (unique) questions selected by both
    theoretical_mean_overlap = n_questions * p_both_select

    # Theoretical variance (using binomial variance)
    theoretical_var = n_questions * p_both_select * (1 - p_both_select)
    theoretical_std = np.sqrt(theoretical_var)

    # Check that empirical mean is within 2 standard errors of theoretical
    standard_error = theoretical_std / np.sqrt(n_simulations)
    assert abs(empirical_mean_overlap - theoretical_mean_overlap) < 2 * standard_error

    # Assert that empirical mean is close to the theoretical mean
    assert abs(empirical_mean_overlap - 3.315) < 0.1


def test_evaluate_ranking_methods_oracle():
    """Test with one perfect predictor using all ranking methods."""
    np.random.seed(42)

    # Oracle always predicts correctly, others have varying skill
    n_questions = 50
    models = ["Oracle", "Good", "Average", "Poor", "Random"]

    data = []
    for i in range(n_questions):
        outcome = np.random.binomial(1, 0.5)

        for model in models:
            if model == "Oracle":
                forecast = float(outcome)  # Perfect prediction
            elif model == "Good":
                # Good but not perfect
                forecast = 0.8 if outcome else 0.2
                forecast += np.random.normal(0, 0.1)
                forecast = np.clip(forecast, 0.01, 0.99)
            elif model == "Average":
                # Average predictor
                forecast = 0.65 if outcome else 0.35
                forecast += np.random.normal(0, 0.15)
                forecast = np.clip(forecast, 0.01, 0.99)
            elif model == "Poor":
                # Poor predictor
                forecast = 0.55 if outcome else 0.45
                forecast += np.random.normal(0, 0.2)
                forecast = np.clip(forecast, 0.01, 0.99)
            else:  # Random
                forecast = np.random.uniform(0.1, 0.9)

            data.append(
                {
                    "model": model,
                    "question_id": f"q{i}",
                    "question_type": "dataset",
                    "forecast": forecast,
                    "resolved_to": outcome,
                }
            )

    df = pd.DataFrame(data)

    # Test Brier and Peer Score first (without BSS)
    ranking_methods_no_bss = {
        "Brier": (rank_by_brier, "avg_brier", True, {}),
        "Peer Score": (rank_by_peer_score, "avg_peer_score", False, {}),
    }
    evaluation_metrics = {
        "Top-1 Retention": (top_k_retention, {"k": 1}),
        "Spearman": (spearman_correlation, {}),
    }

    simulation_methods = {
        "random_sampling": (
            simulate_random_sampling,
            {"n_questions_per_model": 20},
        )
    }
    simulation_func, simulation_kwargs = simulation_methods["random_sampling"]

    results_no_bss = evaluate_ranking_methods(
        df=df,
        ranking_methods=ranking_methods_no_bss,
        evaluation_metrics=evaluation_metrics,
        simulation_func=simulation_func,
        simulation_kwargs=simulation_kwargs,
        n_simulations=50,
        dataset_weight=1.0,
        ref_model="Random",  # Ensure Random is always present
    )

    # Check results for Brier and Peer Score
    for method in ["Brier", "Peer Score"]:
        method_results = results_no_bss[results_no_bss["method"] == method]

        # Oracle should almost always be ranked #1
        top1_retention = method_results["Top-1 Retention"].mean()
        assert (
            top1_retention > 0.90
        ), f"Oracle should almost always be top-1 \
             with {method}, got {top1_retention}"

        # Spearman correlation should be high
        spearman = method_results["Spearman"].mean()
        assert (
            spearman > 0.7
        ), f"Spearman correlation should be high \
             with {method}, got {spearman}"


def test_bss_brier_identical_with_constant_reference():
    """Test that BSS and Brier give identical rankings
    when reference has constant Brier score."""
    np.random.seed(42)

    # Create a dataset with various models including "Always 0.5"
    models = ["Good", "Average", "Poor", "Random", "Always 0.5"]
    n_questions = 30

    data = []
    for i in range(n_questions):
        outcome = np.random.binomial(1, 0.5)

        for model in models:
            if model == "Always 0.5":
                forecast = 0.5  # Constant prediction
            elif model == "Good":
                forecast = 0.8 if outcome else 0.2
                forecast += np.random.normal(0, 0.05)
            elif model == "Average":
                forecast = 0.65 if outcome else 0.35
                forecast += np.random.normal(0, 0.1)
            elif model == "Poor":
                forecast = 0.55 if outcome else 0.45
                forecast += np.random.normal(0, 0.15)
            else:  # Random
                forecast = np.random.uniform(0.2, 0.8)

            forecast = np.clip(forecast, 0.01, 0.99)

            data.append(
                {
                    "model": model,
                    "question_id": f"q{i}",
                    "question_type": "dataset",
                    "forecast": forecast,
                    "resolved_to": outcome,
                }
            )

    df = pd.DataFrame(data)

    # Simulate a dataset
    df_sim = simulate_random_sampling(
        df, n_questions_per_model=20, ref_model="Always 0.5"
    )

    # Get rankings from both methods
    brier_ranking = rank_by_brier(df_sim)
    bss_ranking = rank_by_bss(df_sim, ref_model="Always 0.5")

    # Merge rankings
    merged = pd.merge(
        brier_ranking[["model", "rank"]],
        bss_ranking[["model", "rank"]],
        on="model",
        suffixes=("_brier", "_bss"),
    )

    # Check that ranks are identical
    assert (
        merged["rank_brier"] == merged["rank_bss"]
    ).all(), f"Rankings should be identical:\n{merged}"

    # Also check that the order of avg_brier matches the inverse order of avg_bss
    # (since BSS = 1 - 4*brier when ref has constant brier of 0.25)
    merged_with_scores = pd.merge(
        brier_ranking[["model", "avg_brier"]],
        bss_ranking[["model", "avg_bss"]],
        on="model",
    )

    # Verify the linear relationship: BSS = 1 - 4*brier
    for _, row in merged_with_scores.iterrows():
        expected_bss = 1 - 4 * row["avg_brier"]
        assert np.isclose(
            row["avg_bss"], expected_bss, atol=1e-10
        ), f"BSS should equal 1 - 4*brier for\
             {row['model']}: {row['avg_bss']} vs {expected_bss}"

    # Test with a different reference model to show rankings differ
    if "Random" in df_sim["model"].unique():  # Check Random is in the simulated data
        bss_ranking_different = rank_by_bss(df_sim, ref_model="Random")

        merged_different = pd.merge(
            brier_ranking[["model", "rank"]],
            bss_ranking_different[["model", "rank"]],
            on="model",
            suffixes=("_brier", "_bss_random"),
        )

        # With a different reference, rankings should NOT all be identical
        assert not (
            merged_different["rank_brier"] == merged_different["rank_bss_random"]
        ).all(), "Rankings should differ with non-constant reference model"


def test_evaluate_ranking_methods_correlation_vs_coverage():
    """Test that correlation increases with question coverage."""
    np.random.seed(42)

    # Create a dataset where models have clear performance differences but with noise
    n_questions = 50
    models = ["Excellent", "Good", "Average", "Poor", "Reference"]

    # Create data with clear performance hierarchy plus noise
    data = []
    for i in range(n_questions):
        outcome = np.random.binomial(1, 0.5)  # Random binary outcomes

        # Each model has different calibration + noise
        for model in models:
            if model == "Excellent":
                base_forecast = 0.9 if outcome else 0.1
                noise = np.random.normal(0, 0.1)
            elif model == "Good":
                base_forecast = 0.8 if outcome else 0.2
                noise = np.random.normal(0, 0.15)
            elif model == "Average":
                base_forecast = 0.7 if outcome else 0.3
                noise = np.random.normal(0, 0.2)
            elif model == "Poor":
                base_forecast = 0.6 if outcome else 0.4
                noise = np.random.normal(0, 0.2)
            else:  # Reference
                base_forecast = 0.5
                noise = np.random.normal(0, 0.1)

            # Clip forecast to [0, 1]
            forecast = np.clip(base_forecast + noise, 0.01, 0.99)

            data.append(
                {
                    "model": model,
                    "question_id": f"q{i}",
                    "forecast": forecast,
                    "resolved_to": outcome,
                    "question_type": "dataset",
                }
            )

    df = pd.DataFrame(data)

    ranking_methods = {"Brier": (rank_by_brier, "avg_brier", True, {})}
    evaluation_metrics = {"Spearman": (spearman_correlation, {})}

    # Test with different coverage levels
    coverage_levels = [2, 50, 500]  # Number of questions per model
    mean_correlations = []

    for n_q in coverage_levels:
        # Define simulation
        simulation_methods = {
            "random_sampling": (
                simulate_random_sampling,
                {"n_questions_per_model": n_q},
            )
        }
        simulation_func, simulation_kwargs = simulation_methods["random_sampling"]

        # Perform evaluation
        results = evaluate_ranking_methods(
            df=df,
            ranking_methods=ranking_methods,
            evaluation_metrics=evaluation_metrics,
            simulation_func=simulation_func,
            simulation_kwargs=simulation_kwargs,
            n_simulations=100,
            dataset_weight=1.0,
            ref_model="Reference",
        )

        mean_corr = results["Spearman"].mean()
        mean_correlations.append(mean_corr)

    # Check that correlation generally increases with coverage
    # Allow for small fluctuations due to randomness
    increasing_pairs = sum(
        1
        for i in range(len(mean_correlations) - 1)
        if mean_correlations[i] > mean_correlations[i + 1]
    )
    assert (
        increasing_pairs <= 0
    ), f"Correlation should generally increase with coverage: {mean_correlations}"

    # With many questions, correlation should be very high
    assert (
        mean_correlations[-1] > 0.99
    ), f"With 80% coverage, correlation should be > 0.99, got {mean_correlations[-1]}"


def test_simulate_round_based_questions_per_round():
    """Test that each round has exactly questions_per_round questions."""
    # Create test data
    df = pd.DataFrame(
        {
            "model": ["RefModel", "A", "B"] * 30,
            "question_id": ["q1"] * 90,
            "forecast": [0.5] * 90,
            "resolved_to": [1] * 90,
            "question_type": ["dataset"] * 90,
        }
    )

    # Create unique questions
    for i in range(30):
        df.loc[i * 3 : (i + 1) * 3 - 1, "question_id"] = f"q{i}"

    np.random.seed(42)
    questions_per_round = 7
    df_sim = simulate_round_based(
        df,
        n_rounds=4,
        questions_per_round=questions_per_round,
        models_per_round_mean=2,
        ref_model="RefModel",
    )

    # Check each round has the correct number of questions
    for round_id in df_sim["round_id"].unique():
        round_data = df_sim[df_sim["round_id"] == round_id]
        # Get questions for any model in this round (they should all be the same)
        model = round_data["model"].iloc[0]
        questions_in_round = round_data[round_data["model"] == model][
            "question_id"
        ].values
        assert len(questions_in_round) == questions_per_round


def test_simulate_round_based_ref_model_all_rounds():
    """Test that reference model participates in all rounds."""
    # Create test data
    df = pd.DataFrame(
        {
            "model": ["RefModel", "A", "B", "C"] * 10,
            "question_id": ["q1", "q1", "q1", "q1"] * 10,
            "forecast": [0.5] * 40,
            "resolved_to": [1] * 40,
            "question_type": ["dataset"] * 40,
        }
    )

    # Make unique questions
    for i in range(10):
        df.loc[i * 4 : (i + 1) * 4 - 1, "question_id"] = f"q{i}"

    np.random.seed(42)
    df_sim = simulate_round_based(
        df,
        n_rounds=5,
        questions_per_round=3,
        models_per_round_mean=2,
        ref_model="RefModel",
    )

    # Check that RefModel appears in all rounds
    ref_model_rounds = df_sim[df_sim["model"] == "RefModel"]["round_id"].unique()
    assert len(ref_model_rounds) == 5
    assert set(ref_model_rounds) == {0, 1, 2, 3, 4}


def test_simulate_round_based_all_models_answer_all_questions_in_round():
    """Test that all models in a round answer all questions in that round."""
    # Create test data
    df = pd.DataFrame(
        {
            "model": ["RefModel", "A", "B", "C", "D"] * 20,
            "question_id": ["q1"] * 100,
            "forecast": [0.5] * 100,
            "resolved_to": [1] * 100,
            "question_type": ["dataset"] * 100,
        }
    )

    # Create unique questions
    for i in range(20):
        df.loc[i * 5 : (i + 1) * 5 - 1, "question_id"] = f"q{i}"

    np.random.seed(42)
    questions_per_round = 5
    df_sim = simulate_round_based(
        df,
        n_rounds=3,
        questions_per_round=questions_per_round,
        models_per_round_mean=3,
        ref_model="RefModel",
    )

    # For each round, check that all models answer all questions
    for round_id in df_sim["round_id"].unique():
        round_data = df_sim[df_sim["round_id"] == round_id]

        # Get unique models in this round
        models_in_round = round_data["model"].unique()

        # Get all questions in this round (including duplicates)
        # We'll check using the first model's questions as reference
        first_model = models_in_round[0]
        reference_questions = round_data[round_data["model"] == first_model][
            "question_id"
        ].values

        # Each model should answer exactly questions_per_round questions
        for model in models_in_round:
            model_questions = round_data[round_data["model"] == model][
                "question_id"
            ].values

            # Check that this model answered the correct number of questions
            assert (
                len(model_questions) == questions_per_round
            ), f"Model {model} answered {len(model_questions)} questions, \
                expected {questions_per_round}"

            # Check that this model answered the same questions as the reference
            # (order might differ, but content should be the same)
            assert sorted(model_questions.tolist()) == sorted(
                reference_questions.tolist()
            ), f"Model {model} answered different questions than expected"


def test_simulate_round_based_total_rounds():
    """Test that there are exactly n_rounds rounds."""
    # Create test data
    df = pd.DataFrame(
        {
            "model": ["RefModel", "A", "B", "C"] * 10,
            "question_id": ["q1"] * 40,
            "forecast": [0.5] * 40,
            "resolved_to": [1] * 40,
            "question_type": ["dataset"] * 40,
        }
    )

    # Create unique questions
    for i in range(10):
        df.loc[i * 4 : (i + 1) * 4 - 1, "question_id"] = f"q{i}"

    np.random.seed(42)
    n_rounds = 8
    df_sim = simulate_round_based(
        df,
        n_rounds=n_rounds,
        questions_per_round=3,
        models_per_round_mean=2,
        ref_model="RefModel",
    )

    # Check total number of rounds
    unique_rounds = df_sim["round_id"].unique()
    assert len(unique_rounds) == n_rounds
    assert set(unique_rounds) == set(range(n_rounds))


def test_simulate_round_based_models_per_round_mean():
    """Test that empirical mean of models per round matches the parameter."""
    # Create test data with many models
    n_models = 50
    models = ["RefModel"] + [f"Model_{i}" for i in range(n_models - 1)]

    data = []
    for model in models:
        for q in range(20):
            data.append(
                {
                    "model": model,
                    "question_id": f"q{q}",
                    "forecast": 0.5,
                    "resolved_to": 1,
                    "question_type": "dataset",
                }
            )
    df = pd.DataFrame(data)

    # Run many simulations
    np.random.seed(42)
    n_simulations = 100
    models_per_round_mean = 15
    all_models_per_round = []

    for i in range(n_simulations):
        np.random.seed(i)
        df_sim = simulate_round_based(
            df,
            n_rounds=10,
            questions_per_round=5,
            models_per_round_mean=models_per_round_mean,
            ref_model="RefModel",
        )

        # Count models per round (excluding reference model)
        for round_id in df_sim["round_id"].unique():
            round_data = df_sim[df_sim["round_id"] == round_id]
            models_in_round = round_data["model"].unique()
            # Subtract 1 for reference model
            n_models_in_round = len(models_in_round) - 1
            all_models_per_round.append(n_models_in_round)

    # Check empirical mean
    empirical_mean = np.mean(all_models_per_round)

    # The empirical mean should be close to the parameter
    # Allow for some deviation due to Poisson sampling
    assert (
        abs(empirical_mean - models_per_round_mean) < 1.0
    ), f"Empirical mean {empirical_mean} too far from parameter {models_per_round_mean}"

    # Also check that the distribution looks roughly Poisson
    # by checking the variance is close to the mean
    empirical_var = np.var(all_models_per_round)
    assert (
        abs(empirical_var - empirical_mean) < 2.0
    ), f"Variance {empirical_var} too different from mean {empirical_mean} for Poisson"


def test_simulate_round_based_perfect_ranking_when_all_questions_sampled():
    """Test that ranking is perfect when questions_per_round equals total questions."""
    # Create test data with clear performance differences
    np.random.seed(42)
    n_questions = 20
    models = ["RefModel", "Excellent", "Good", "Average", "Poor"]

    data = []
    for i in range(n_questions):
        outcome = np.random.binomial(1, 0.5)

        for model in models:
            if model == "RefModel":
                forecast = 0.5  # Constant reference
            elif model == "Excellent":
                forecast = 0.9 if outcome else 0.1
            elif model == "Good":
                forecast = 0.8 if outcome else 0.2
            elif model == "Average":
                forecast = 0.7 if outcome else 0.3
            elif model == "Poor":
                forecast = 0.6 if outcome else 0.4

            data.append(
                {
                    "model": model,
                    "question_id": f"q{i}",
                    "forecast": forecast,
                    "resolved_to": outcome,
                    "question_type": "dataset",
                }
            )

    df = pd.DataFrame(data)

    # Calculate true ranking using all data
    true_ranking = rank_by_brier(df)

    # Run simulation with questions_per_round = total questions
    # and high models_per_round_mean to ensure all models participate
    np.random.seed(123)
    df_sim = simulate_round_based(
        df,
        n_rounds=5,
        questions_per_round=n_questions,  # Sample ALL questions each round
        models_per_round_mean=len(models),  # High mean to get all models
        ref_model="RefModel",
    )

    # Calculate ranking from simulation
    sim_ranking = rank_by_brier(df_sim)

    # Check which models participated in the simulation
    models_in_sim = set(df_sim["model"].unique())
    models_in_true = set(true_ranking["model"].unique())

    # All models should have participated (with high models_per_round_mean)
    assert (
        models_in_sim == models_in_true
    ), f"Not all models participated. Missing: {models_in_true - models_in_sim}"

    # Merge rankings
    comparison = pd.merge(
        true_ranking[["model", "avg_brier", "rank"]],
        sim_ranking[["model", "avg_brier", "rank"]],
        on="model",
        suffixes=("_true", "_sim"),
    )

    # Check that rankings match perfectly
    assert (
        comparison["rank_true"] == comparison["rank_sim"]
    ).all(), f"Rankings don't match:\n{comparison}"

    # Check that Brier scores are identical (within floating point tolerance)
    assert np.allclose(
        comparison["avg_brier_true"], comparison["avg_brier_sim"]
    ), f"Brier scores don't match:\n{comparison}"


def test_simulate_round_based_perfect_ranking_multiple_simulations():
    """Test perfect ranking across multiple simulations when sampling all questions."""
    # Create test data
    np.random.seed(42)
    n_questions = 15
    models = ["RefModel", "A", "B", "C"]

    data = []
    for i in range(n_questions):
        outcome = np.random.binomial(1, 0.6)

        for model in models:
            if model == "RefModel":
                forecast = 0.5
            elif model == "A":
                forecast = 0.85 if outcome else 0.15
            elif model == "B":
                forecast = 0.75 if outcome else 0.25
            elif model == "C":
                forecast = 0.65 if outcome else 0.35

            data.append(
                {
                    "model": model,
                    "question_id": f"q{i}",
                    "forecast": forecast,
                    "resolved_to": outcome,
                    "question_type": "dataset",
                }
            )

    df = pd.DataFrame(data)

    # Calculate true ranking
    true_ranking = rank_by_brier(df)

    # Run multiple simulations
    n_simulations = 10
    for sim in range(n_simulations):
        np.random.seed(sim + 1000)

        df_sim = simulate_round_based(
            df,
            n_rounds=3,
            questions_per_round=n_questions,
            models_per_round_mean=100,  # Very high to ensure all models participate
            ref_model="RefModel",
        )

        sim_ranking = rank_by_brier(df_sim)

        # Verify all models participated
        assert set(sim_ranking["model"]) == set(true_ranking["model"])

        # Merge and compare
        comparison = pd.merge(
            true_ranking[["model", "rank"]],
            sim_ranking[["model", "rank"]],
            on="model",
            suffixes=("_true", "_sim"),
        )

        # Rankings should match perfectly in every simulation
        assert (
            comparison["rank_true"] == comparison["rank_sim"]
        ).all(), f"Rankings don't match in simulation {sim}:\n{comparison}"
