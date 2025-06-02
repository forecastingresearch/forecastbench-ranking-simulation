import contextlib
import io
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
    process_raw_data,
    rank_by_brier,
    rank_by_bss,
    rank_by_diff_adj_brier,
    rank_by_peer_score,
    rank_with_weighting,
    ranking_sanity_check,
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


def test_rank_by_diff_adj_brier_edge_cases():
    """Test edge cases like single model, single question, or missing data."""

    # Test 1: Single model
    df_single_model = pd.DataFrame(
        {
            "model": ["Model_A"] * 5,
            "question_id": [f"q{i}" for i in range(5)],
            "forecast": [0.7, 0.8, 0.6, 0.9, 0.5],
            "resolved_to": [1, 1, 0, 1, 0],
            "question_type": ["dataset"] * 5,
        }
    )

    # Should not raise an error
    ranking = rank_by_diff_adj_brier(df_single_model)
    assert len(ranking) == 1
    assert ranking["model"].iloc[0] == "Model_A"
    assert ranking["rank"].iloc[0] == 1

    # Test 2: All models predict the same for all questions
    df_identical = pd.DataFrame(
        {
            "model": ["A", "B", "C"] * 3,
            "question_id": ["q1", "q1", "q1", "q2", "q2", "q2", "q3", "q3", "q3"],
            "forecast": [0.7] * 9,
            "resolved_to": [1, 1, 1, 0, 0, 0, 1, 1, 1],
            "question_type": ["dataset"] * 9,
        }
    )

    ranking = rank_by_diff_adj_brier(df_identical)

    # All models should have the same score and rank
    assert len(ranking["avg_diff_adj_brier"].unique()) == 1
    assert (ranking["rank"] == 1).all()  # All tied for first


def test_rank_by_diff_adj_brier_balanced_dataset():
    """Test that all ranking methods agree on a perfectly balanced dataset."""
    # Create a balanced dataset where each model has consistent skill
    # Model skill levels: A=0.9, B=0.7, C=0.5, D=0.3 (calibration)

    np.random.seed(42)
    n_questions = 20
    models = {
        "Model_A": 0.9,  # Best calibration
        "Model_B": 0.7,
        "Model_C": 0.5,
        "Model_D": 0.3,  # Worst calibration
    }

    data = []
    for q_idx in range(n_questions):
        # Random outcome for each question
        outcome = np.random.binomial(1, 0.5)

        for model, skill in models.items():
            # Model predicts: skill if outcome=1, (1-skill) if outcome=0
            if outcome == 1:
                forecast = skill
            else:
                forecast = 1 - skill

            data.append(
                {
                    "model": model,
                    "question_id": f"q{q_idx}",
                    "forecast": forecast,
                    "resolved_to": outcome,
                    "question_type": "dataset",
                }
            )

    df = pd.DataFrame(data)

    # Get rankings from all methods
    brier_ranking = rank_by_brier(df)
    diff_adj_ranking = rank_by_diff_adj_brier(df)
    peer_ranking = rank_by_peer_score(df)
    bss_ranking = rank_by_bss(df, ref_model="Model_C")

    # All methods should produce the same ranking
    for ranking_df in [brier_ranking, diff_adj_ranking, peer_ranking, bss_ranking]:
        assert ranking_df[ranking_df["model"] == "Model_A"]["rank"].iloc[0] == 1
        assert ranking_df[ranking_df["model"] == "Model_B"]["rank"].iloc[0] == 2
        assert ranking_df[ranking_df["model"] == "Model_C"]["rank"].iloc[0] == 3
        assert ranking_df[ranking_df["model"] == "Model_D"]["rank"].iloc[0] == 4

    # Check that score differences are preserved
    # The difference between consecutive models should be the same
    # for brier and diff-adj brier
    brier_scores = brier_ranking.set_index("model")["avg_brier"]
    diff_adj_scores = diff_adj_ranking.set_index("model")["avg_diff_adj_brier"]

    # Calculate differences between consecutive models
    models_ordered = ["Model_A", "Model_B", "Model_C", "Model_D"]
    for i in range(len(models_ordered) - 1):
        model1, model2 = models_ordered[i], models_ordered[i + 1]

        brier_diff = brier_scores[model2] - brier_scores[model1]
        diff_adj_diff = diff_adj_scores[model2] - diff_adj_scores[model1]

        # Differences should be identical in balanced case
        assert np.isclose(
            brier_diff, diff_adj_diff, atol=1e-10
        ), f"Score differences should match: {brier_diff} vs {diff_adj_diff}"


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


def test_top_k_retention_with_proper_tie_handling():
    """Test that top_k_retention properly handles ties by
    including all models with rank <= k."""

    # Test 1: Simple tie at boundary
    df_true = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_true": [1, 2, 2, 4, 5],  # B and C tied for 2nd
        }
    )

    df_sim = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [1, 2, 2, 4, 5],  # Same ranking
        }
    )

    # With k=2, should include A, B, and C (all with rank <= 2)
    retention = top_k_retention(df_true, df_sim, k=2)
    assert retention == 1.0  # All 3 models (A, B, C) retained

    # Test 2: Tie at boundary with different sim ranking
    df_sim2 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [1, 3, 2, 4, 5],  # B dropped to 3rd
        }
    )

    retention2 = top_k_retention(df_true, df_sim2, k=2)
    # True top-2: {A, B, C}, Sim top-2: {A, C}
    # Retained: {A, C}, so 2/3
    assert np.isclose(retention2, 2 / 3)

    # Test 3: Multiple ties
    df_true3 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F"],
            "rank_true": [1, 1, 3, 3, 3, 6],  # A,B tied for 1st; C,D,E tied for 3rd
        }
    )

    df_sim3 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F"],
            "rank_sim": [2, 1, 3, 4, 5, 6],  # B wins tie, A second
        }
    )

    retention3 = top_k_retention(df_true3, df_sim3, k=2)
    # True top-2: {A, B} (both rank 1 <= 2)
    # Sim top-2: {B, A} (ranks 1 and 2)
    # All retained: 2/2 = 1.0
    assert retention3 == 1.0

    retention3_k3 = top_k_retention(df_true3, df_sim3, k=3)
    # True top-3: {A, B, C, D, E} (all with rank <= 3)
    # Sim top-3: {B, A, C} (ranks 1, 2, 3)
    # Retained: {A, B, C}, so 3/5
    assert np.isclose(retention3_k3, 3 / 5)


def test_top_k_retention_edge_cases():
    """Test edge cases for top_k_retention."""

    # Test 1: All models tied
    df_all_tied = pd.DataFrame(
        {"model": ["A", "B", "C", "D"], "rank_true": [1, 1, 1, 1]}  # All tied for 1st
    )

    df_sim_different = pd.DataFrame(
        {"model": ["A", "B", "C", "D"], "rank_sim": [1, 2, 3, 4]}  # All different ranks
    )

    retention = top_k_retention(df_all_tied, df_sim_different, k=1)
    # True top-1: {A, B, C, D} (all rank 1)
    # Sim top-1: {A}
    # Retained: {A}, so 1/4
    assert np.isclose(retention, 1 / 4)

    # Test 2: k larger than number of models
    retention_large_k = top_k_retention(df_all_tied, df_sim_different, k=10)
    # All models are in top-10 for both, so retention = 1.0
    assert retention_large_k == 1.0


def test_top_k_retention_integration_with_tied_scores():
    """Integration test: Test full pipeline from scores
    to retention with ties across all ranking methods."""

    # Create realistic data with natural ties
    df = pd.DataFrame(
        {
            "model": ["RefModel", "A", "B", "C", "D", "E"] * 8,
            "question_id": (
                ["q1", "q1", "q1", "q1", "q1", "q1"] * 4
                + ["q5", "q5", "q5", "q5", "q5", "q5"] * 4
            ),
            "question_type": ["dataset"] * 24 + ["market"] * 24,
            "forecast": (
                [0.5, 0.7, 0.8, 0.8, 0.5, 0.3] * 4  # Dataset questions
                + [0.5, 0.6, 0.7, 0.7, 0.5, 0.4] * 4
            ),  # Market questions
            "resolved_to": ([1, 1, 1, 1, 1, 1] * 2 + [0, 0, 0, 0, 0, 0] * 2) * 2,
        }
    )

    # Add more questions to make patterns clearer
    for i in range(2, 5):
        df.loc[i * 6 : (i + 1) * 6 - 1, "question_id"] = f"q{i}"
    for i in range(6, 8):
        df.loc[i * 6 : (i + 1) * 6 - 1, "question_id"] = f"q{i}"

    # Test 1: Brier Score ranking
    brier_ranking = rank_by_brier(df)

    # Verify B and C have identical Brier scores
    b_score = brier_ranking[brier_ranking["model"] == "B"]["avg_brier"].iloc[0]
    c_score = brier_ranking[brier_ranking["model"] == "C"]["avg_brier"].iloc[0]
    assert np.isclose(b_score, c_score), "B and C should have same Brier score"

    # Verify B and C have same rank
    b_rank = brier_ranking[brier_ranking["model"] == "B"]["rank"].iloc[0]
    c_rank = brier_ranking[brier_ranking["model"] == "C"]["rank"].iloc[0]
    assert b_rank == c_rank, f"B and C should have same rank, got {b_rank} and {c_rank}"

    # Verify RefModel and D are tied
    ref_score = brier_ranking[brier_ranking["model"] == "RefModel"]["avg_brier"].iloc[0]
    d_score = brier_ranking[brier_ranking["model"] == "D"]["avg_brier"].iloc[0]
    assert np.isclose(ref_score, d_score), "RefModel and D should have same Brier score"

    ref_rank = brier_ranking[brier_ranking["model"] == "RefModel"]["rank"].iloc[0]
    d_rank = brier_ranking[brier_ranking["model"] == "D"]["rank"].iloc[0]
    assert ref_rank == d_rank, "RefModel and D should have same rank"

    # Verify proper rank gaps (e.g., if two models tied for 2nd, next is 4th)
    rank_counts = brier_ranking["rank"].value_counts().sort_index()

    expected_next_rank = 1
    for rank, count in rank_counts.items():
        assert (
            rank == expected_next_rank
        ), f"Expected rank {expected_next_rank}, got {rank}"
        expected_next_rank = rank + count

    # Test 2: BSS ranking (using RefModel as reference)
    bss_ranking = rank_by_bss(df, ref_model="RefModel")

    # RefModel should have BSS of 0
    ref_bss = bss_ranking[bss_ranking["model"] == "RefModel"]["avg_bss"].iloc[0]
    assert np.isclose(ref_bss, 0.0), f"Reference model should have BSS=0, got {ref_bss}"

    # Verify BSS ranking uses method='min' for ties
    bss_rank_counts = bss_ranking["rank"].value_counts().sort_index()
    expected_next_rank = 1
    for rank, count in bss_rank_counts.items():
        assert (
            rank == expected_next_rank
        ), f"BSS ranking: expected rank {expected_next_rank}, got {rank}"
        expected_next_rank = rank + count

    # Test 3: Peer Score ranking
    peer_ranking = rank_by_peer_score(df)

    # Models with same Brier score should have same peer score
    # Get average Brier per question for verification
    df_with_brier = df.copy()
    df_with_brier["brier"] = (
        df_with_brier["forecast"] - df_with_brier["resolved_to"]
    ) ** 2

    # Verify peer score ranking uses method='min'
    peer_rank_counts = peer_ranking["rank"].value_counts().sort_index()
    expected_next_rank = 1
    for rank, count in peer_rank_counts.items():
        assert (
            rank == expected_next_rank
        ), f"Peer ranking: expected rank {expected_next_rank}, got {rank}"
        expected_next_rank = rank + count

    # Test 4: Dataset/Market weighting with ties
    weighted_ranking = rank_with_weighting(
        df=df,
        ranking_func=rank_by_brier,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    # Verify weighted calculation
    for _, row in weighted_ranking.iterrows():
        expected_weighted = (
            0.5 * row["avg_brier_dataset"] + 0.5 * row["avg_brier_market"]
        )
        assert np.isclose(
            row["avg_brier_weighted"], expected_weighted
        ), f"Weighted score calculation error for {row['model']}"

    # Verify ties are preserved in weighted ranking
    weighted_scores = weighted_ranking.set_index("model")["avg_brier_weighted"]
    for i, model1 in enumerate(weighted_ranking["model"]):
        for j, model2 in enumerate(weighted_ranking["model"]):
            if i < j and np.isclose(
                weighted_scores[model1], weighted_scores[model2], atol=1e-10
            ):
                rank1 = weighted_ranking[weighted_ranking["model"] == model1][
                    "rank"
                ].iloc[0]
                rank2 = weighted_ranking[weighted_ranking["model"] == model2][
                    "rank"
                ].iloc[0]
                assert (
                    rank1 == rank2
                ), f"{model1} and {model2} have same score but different ranks"

    # Test 5: Top-k retention with each ranking method
    df_sim = df.copy()
    # Change C's forecasts slightly to break the tie
    df_sim.loc[df_sim["model"] == "C", "forecast"] = (
        df_sim.loc[df_sim["model"] == "C", "forecast"] + 0.02
    )

    # Test Brier ranking retention
    true_brier = rank_by_brier(df).rename(columns={"rank": "rank_true"})
    sim_brier = rank_by_brier(df_sim).rename(columns={"rank": "rank_sim"})

    # Get models in top-2 for both
    true_top_2_brier = set(true_brier[true_brier["rank_true"] <= 2]["model"])
    sim_top_2_brier = set(sim_brier[sim_brier["rank_sim"] <= 2]["model"])

    # Calculate expected retention
    retained_brier = true_top_2_brier & sim_top_2_brier
    expected_retention_brier = len(retained_brier) / len(true_top_2_brier)

    # Verify top_k_retention function gives same result
    actual_retention_brier = top_k_retention(true_brier, sim_brier, k=2)
    assert np.isclose(
        actual_retention_brier, expected_retention_brier
    ), f"Brier retention mismatch: expected {expected_retention_brier}, \
        got {actual_retention_brier}"

    # Test Peer Score retention
    true_peer = rank_by_peer_score(df).rename(columns={"rank": "rank_true"})
    sim_peer = rank_by_peer_score(df_sim).rename(columns={"rank": "rank_sim"})

    true_top_2_peer = set(true_peer[true_peer["rank_true"] <= 2]["model"])
    sim_top_2_peer = set(sim_peer[sim_peer["rank_sim"] <= 2]["model"])

    retained_peer = true_top_2_peer & sim_top_2_peer
    expected_retention_peer = len(retained_peer) / len(true_top_2_peer)

    actual_retention_peer = top_k_retention(true_peer, sim_peer, k=2)
    assert np.isclose(
        actual_retention_peer, expected_retention_peer
    ), f"Peer retention mismatch: expected {expected_retention_peer}, \
        got {actual_retention_peer}"

    # Verify that when there are ties, top-k includes more than k models
    assert (
        len(true_top_2_brier) >= 2
    ), "When ties exist at boundary, top-k should include all tied models"
    assert (
        len(true_top_2_peer) >= 2
    ), "When ties exist at boundary, top-k should include all tied models"


def test_ranking_sanity_check():
    """Test ranking_sanity_check function with missing models and known results."""

    # Test 1: Model missing from df_true_ranking
    df_true = pd.DataFrame({"model": ["A", "B", "C"], "rank_true": [1, 2, 3]})

    df_sim = pd.DataFrame({"model": ["A", "B", "C", "D"], "rank_sim": [1, 2, 3, 4]})

    # Test with verbose=True (should print warning)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = ranking_sanity_check(
            df_true,
            df_sim,
            model_list=["A", "B", "D"],  # D is missing from df_true
            pct_point_tol=0.05,
            verbose=True,
        )

    assert np.isnan(result), "Should return np.nan when model is missing"
    assert "Warning: Model 'D' not found in df_true_ranking" in f.getvalue()

    # Test with verbose=False (should not print warning)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = ranking_sanity_check(
            df_true,
            df_sim,
            model_list=["A", "B", "D"],
            pct_point_tol=0.05,
            verbose=False,
        )

    assert np.isnan(result), "Should return np.nan when model is missing"
    assert f.getvalue() == "", "Should not print warning when verbose=False"

    # Test 2: Model missing from df_sim_ranking
    df_true2 = pd.DataFrame({"model": ["A", "B", "C", "D"], "rank_true": [1, 2, 3, 4]})

    df_sim2 = pd.DataFrame({"model": ["A", "B", "C"], "rank_sim": [1, 2, 3]})

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = ranking_sanity_check(
            df_true2,
            df_sim2,
            model_list=["B", "D"],  # D is missing from df_sim
            pct_point_tol=0.05,
            verbose=True,
        )

    assert np.isnan(result), "Should return np.nan when model is missing from sim"
    assert "Warning: Model 'D' not found in df_sim_ranking" in f.getvalue()

    # Test 3: All models present, all pass the test
    df_true3 = pd.DataFrame(
        {"model": ["A", "B", "C", "D", "E"], "rank_true": [1, 2, 3, 4, 5]}
    )

    df_sim3 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "rank_sim": [1, 2, 3, 4, 5],  # Exact same ranking
        }
    )

    result = ranking_sanity_check(
        df_true3, df_sim3, model_list=["A", "C", "E"], pct_point_tol=0.05
    )

    assert result == 1.0, "Should return 1.0 when all models have same percentile rank"

    # Test 4: All models present, some fail the test
    df_true4 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "rank_true": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    df_sim4 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "rank_sim": [1, 9, 3, 4, 5, 6, 7, 8, 2, 10],  # B and I swapped positions
        }
    )

    # B: true rank 2/10 = 0.2, sim rank 9/10 = 0.9, diff = 0.7 > 0.05
    # I: true rank 9/10 = 0.9, sim rank 2/10 = 0.2, diff = 0.7 > 0.05
    result = ranking_sanity_check(
        df_true4, df_sim4, model_list=["A", "B", "I"], pct_point_tol=0.05
    )

    assert result == 0.0, "Should return 0.0 when some models fail the test"

    # Test 5: Edge case with tolerance
    df_true5 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "rank_true": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    df_sim5 = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "rank_sim": [1, 2, 4, 3, 5, 6, 7, 8, 9, 10],  # C and D swapped
        }
    )

    # C: true rank 3/10 = 0.3, sim rank 4/10 = 0.4, diff = 0.1
    # D: true rank 4/10 = 0.4, sim rank 3/10 = 0.3, diff = 0.1
    # With tolerance 0.15, both should pass
    result = ranking_sanity_check(
        df_true5, df_sim5, model_list=["C", "D"], pct_point_tol=0.15
    )

    assert result == 1.0, "Should return 1.0 when differences are within tolerance"

    # With tolerance 0.05, both should fail
    result = ranking_sanity_check(
        df_true5, df_sim5, model_list=["C", "D"], pct_point_tol=0.05
    )

    assert result == 0.0, "Should return 0.0 when differences exceed tolerance"

    # Test 6: Real-world scenario with specific models
    df_true6 = pd.DataFrame(
        {
            "model": [
                "Superforecaster median forecast",
                "Public median forecast",
                "Random Uniform",
                "Always 0",
                "Always 1",
                "Other Model",
            ],
            "rank_true": [5, 15, 25, 48, 50, 30],  # Out of 50 models
        }
    )

    df_sim6 = pd.DataFrame(
        {
            "model": [
                "Superforecaster median forecast",
                "Public median forecast",
                "Random Uniform",
                "Always 0",
                "Always 1",
                "Other Model",
            ],
            "rank_sim": [8, 18, 26, 47, 49, 25],  # Slightly different ranks
        }
    )

    # Calculate expected percentile differences:
    # Superforecaster: |5/50 - 8/50| = |0.1 - 0.16| = 0.06
    # Public median: |15/50 - 18/50| = |0.3 - 0.36| = 0.06
    # Random Uniform: |25/50 - 26/50| = |0.5 - 0.52| = 0.02
    # Always 0: |48/50 - 47/50| = |0.96 - 0.94| = 0.02
    # Always 1: |50/50 - 49/50| = |1.0 - 0.98| = 0.02

    # With tolerance 0.25, all should pass
    result = ranking_sanity_check(
        df_true6,
        df_sim6,
        model_list=[
            "Superforecaster median forecast",
            "Public median forecast",
            "Random Uniform",
            "Always 0",
            "Always 1",
        ],
        pct_point_tol=0.25,
        verbose=False,
    )

    assert result == 1.0, "Should return 1.0 with large tolerance"

    # With tolerance 0.05, Superforecaster and Public median should fail
    result = ranking_sanity_check(
        df_true6,
        df_sim6,
        model_list=[
            "Superforecaster median forecast",
            "Public median forecast",
            "Random Uniform",
            "Always 0",
            "Always 1",
        ],
        pct_point_tol=0.05,
        verbose=False,
    )

    assert result == 0.0, "Should return 0.0 when some exceed tight tolerance"

    # Test only models within tolerance
    result = ranking_sanity_check(
        df_true6,
        df_sim6,
        model_list=["Random Uniform", "Always 0", "Always 1"],
        pct_point_tol=0.05,
        verbose=False,
    )

    assert (
        result == 1.0
    ), "Should return 1.0 when all selected models are within tolerance"


def test_combine_rankings_with_ties():
    """Test that combine_rankings properly handles ties when re-ranking."""

    # Also need to update combine_rankings to use method='min'
    # Here's a focused test for that function

    # Dataset rankings with ties
    df_dataset = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D"],
            "avg_brier": [0.1, 0.2, 0.2, 0.4],  # B and C tied
            "rank": [1, 2, 2, 4],
        }
    )

    # Market rankings with different ties
    df_market = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D"],
            "avg_brier": [0.3, 0.1, 0.3, 0.3],  # A, C, and D tied
            "rank": [2, 1, 2, 2],
        }
    )

    # Combine with 50/50 weight
    combined = combine_rankings(
        df_dataset,
        df_market,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    print("\nCombined rankings:")
    print(combined)

    # Check weighted scores
    # A: 0.5 * 0.1 + 0.5 * 0.3 = 0.2
    # B: 0.5 * 0.2 + 0.5 * 0.1 = 0.15
    # C: 0.5 * 0.2 + 0.5 * 0.3 = 0.25
    # D: 0.5 * 0.4 + 0.5 * 0.3 = 0.35

    # So B < A < C < D, no ties in final ranking
    assert combined[combined["model"] == "B"]["rank"].iloc[0] == 1
    assert combined[combined["model"] == "A"]["rank"].iloc[0] == 2
    assert combined[combined["model"] == "C"]["rank"].iloc[0] == 3
    assert combined[combined["model"] == "D"]["rank"].iloc[0] == 4

    # Test with ties in combined scores
    df_dataset2 = pd.DataFrame(
        {"model": ["A", "B", "C"], "avg_brier": [0.1, 0.3, 0.5], "rank": [1, 2, 3]}
    )

    df_market2 = pd.DataFrame(
        {"model": ["A", "B", "C"], "avg_brier": [0.5, 0.3, 0.1], "rank": [3, 2, 1]}
    )

    combined2 = combine_rankings(
        df_dataset2,
        df_market2,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=0.5,
    )

    # All models should have weighted score of 0.3
    assert np.allclose(combined2["avg_brier_weighted"], 0.3)

    # All should be tied for rank 1
    assert (
        combined2["rank"] == 1
    ).all(), f"All models should be rank 1, got {combined2['rank'].values}"


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


@pytest.mark.slow
def test_simulation_regression_results():
    """
    Regression test to ensure simulation results remain consistent.

    These are NOT ground truth values, but results from a known good run
    that we want to preserve. If these tests fail, it means the simulation
    behavior has changed and we need to verify if the change is intentional.

    Expected values generated on 2025-06-02 with:
    - numpy seed: 20250527
    - n_simulations: 50
    - n_questions_per_model: 125
    - dataset_weight: 0.5
    - simulation_method: "random_sampling"
    - ref_model = "GPT-4 (zero shot)"
    """
    np.random.seed(20250527)

    # Load and process data
    df = process_raw_data("../data/raw/llm_and_human_leaderboard_overall.pkl")

    # Define methods
    ranking_methods = {
        "Brier": (rank_by_brier, "avg_brier", True, {}),
        "Diff-Adj. Brier": (rank_by_diff_adj_brier, "avg_diff_adj_brier", True, {}),
        "BSS": (rank_by_bss, "avg_bss", False, {"ref_model": "Naive Forecaster"}),
        "Peer Score": (rank_by_peer_score, "avg_peer_score", False, {}),
    }

    evaluation_metrics = {
        "Spearman": (spearman_correlation, {}),
        "Top-20 Retention": (top_k_retention, {"k": 20}),
    }

    # Run simulation
    results = evaluate_ranking_methods(
        df=df,
        ranking_methods=ranking_methods,
        evaluation_metrics=evaluation_metrics,
        simulation_func=simulate_random_sampling,
        simulation_kwargs={"n_questions_per_model": 125},
        n_simulations=50,
        dataset_weight=0.5,
        ref_model="Naive Forecaster",
    )

    # Expected results (from known good run)
    expected_results = {
        "Brier": {"Spearman": 0.726376, "Top-20 Retention": 0.503},
        "Diff-Adj. Brier": {"Spearman": 0.813342, "Top-20 Retention": 0.582},
        "BSS": {"Spearman": 0.134529, "Top-20 Retention": 0.321},
        "Peer Score": {"Spearman": 0.813029, "Top-20 Retention": 0.590},
    }

    # Check results
    summary = results.groupby("method")[["Spearman", "Top-20 Retention"]].mean()

    for method, expected in expected_results.items():
        for metric, expected_value in expected.items():
            actual_value = summary.loc[method, metric]

            # Use reasonable tolerance for random simulations
            assert np.isclose(
                actual_value, expected_value, atol=0.00001
            ), f"{method} {metric}: expected {expected_value:.6f}, \
                got {actual_value:.6f}"


@pytest.mark.slow
def test_simulation_regression_round_based_results():
    """
    Regression test to ensure simulation results remain consistent.

    These are NOT ground truth values, but results from a known good run
    that we want to preserve. If these tests fail, it means the simulation
    behavior has changed and we need to verify if the change is intentional.

    Expected values generated on 2025-06-02 with:
    - numpy seed: 20250527
    - n_simulations: 50
    - n_rounds: 15
    - questions_per_round = 25
    - models_per_round_mean = 40
    - dataset_weight: 0.5
    - simulation_method: "round_based"
    - ref_model = "Naive Forecaster"
    """
    np.random.seed(20250527)

    # Load and process data
    df = process_raw_data("../data/raw/llm_and_human_leaderboard_overall.pkl")

    # Define methods
    ranking_methods = {
        "Brier": (rank_by_brier, "avg_brier", True, {}),
        "Diff-Adj. Brier": (rank_by_diff_adj_brier, "avg_diff_adj_brier", True, {}),
        "BSS": (rank_by_bss, "avg_bss", False, {"ref_model": "Naive Forecaster"}),
        "Peer Score": (rank_by_peer_score, "avg_peer_score", False, {}),
    }

    evaluation_metrics = {
        "Spearman": (spearman_correlation, {}),
        "Top-20 Retention": (top_k_retention, {"k": 20}),
    }

    # Run simulation
    results = evaluate_ranking_methods(
        df=df,
        ranking_methods=ranking_methods,
        evaluation_metrics=evaluation_metrics,
        simulation_func=simulate_round_based,
        simulation_kwargs={
            "n_rounds": 15,
            "questions_per_round": 25,
            "models_per_round_mean": 40,
        },
        n_simulations=50,
        dataset_weight=0.5,
        ref_model="Naive Forecaster",
    )

    # Expected results (from known good run)
    expected_results = {
        "Brier": {"Spearman": 0.694564, "Top-20 Retention": 0.476},
        "Diff-Adj. Brier": {"Spearman": 0.771530, "Top-20 Retention": 0.557},
        "BSS": {"Spearman": 0.145300, "Top-20 Retention": 0.319},
        "Peer Score": {"Spearman": 0.770672, "Top-20 Retention": 0.564},
    }

    # Check results
    summary = results.groupby("method")[["Spearman", "Top-20 Retention"]].mean()

    for method, expected in expected_results.items():
        for metric, expected_value in expected.items():
            actual_value = summary.loc[method, metric]

            # Use reasonable tolerance for random simulations
            assert np.isclose(
                actual_value, expected_value, atol=0.00001
            ), f"{method} {metric}: expected {expected_value:.6f}, \
                got {actual_value:.6f}"
