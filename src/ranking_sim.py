import numpy as np
import pandas as pd
import pyfixest as pf

# ================
# Data preparation
# ================


def process_raw_data(input_name):
    # Read the pickle file & convert to pandas df
    pkl = pd.read_pickle(input_name)
    df = pd.DataFrame()
    for ii in range(len(pkl)):
        df_temp = pkl[ii]["df"]
        df_temp["model"] = pkl[ii]["model"]
        df_temp["organization"] = pkl[ii]["organization"]
        df = pd.concat([df, df_temp])
    df = df.reset_index(drop=True)

    # Define question type
    df["question_type"] = df["source"].apply(
        lambda x: (
            "market"
            if x in ["manifold", "infer", "metaculus", "polymarket"]
            else "dataset"
        )
    )

    # Calculate forecast horizon
    df["horizon"] = np.nan
    mask = df["question_type"] == "dataset"
    df.loc[mask, "horizon"] = (
        pd.to_datetime(df.loc[mask, "resolution_date"])
        - pd.to_datetime(df.loc[mask, "forecast_due_date"])
    ).astype(int)
    if (df["horizon"] < 0).any():
        raise ValueError("Some resolution dates are before forecast_due_date.")

    # Create a new column 'question_id' by concatenating 'source', 'id',
    # and 'horizon' columns. This is done to create a unique identifier
    # for each question/prediction
    df["question_id"] = (
        df["source"].astype(str)
        + "-"
        + df["id"].astype(str)
        + "-"
        + df["horizon"].astype(str)
        + "-"
        + df["forecast_due_date"].astype(str)
    )

    # Filter out unresolved questions
    mask = df["resolved"].eq(True)
    df = df.loc[mask,]

    return df


# ===============
# Ranking methods
# ===============


def brier_score(df):
    return (df["forecast"] - df["resolved_to"]) ** 2


def rank_by_brier(df):
    df = df.copy()
    df["brier_score"] = brier_score(df)

    # Calculate average Brier score per model
    model_scores = df.groupby("model")["brier_score"].mean().reset_index()
    model_scores.rename(columns={"brier_score": "avg_brier"}, inplace=True)

    # Rank models (lower Brier score is better)
    model_scores["rank"] = model_scores["avg_brier"].rank(ascending=True, method="min")
    model_scores = model_scores.sort_values(by="rank")

    return model_scores


def rank_by_diff_adj_brier(df):
    df = df.copy()
    df["brier_score"] = brier_score(df)

    # Calculate "question difficulty" by estimating
    # a two-way fixed effect model:
    #
    #   brier{i, j} = a_i + b_j + u_{i,j}
    #
    # where i = forecaster, and j = question. Question difficulty
    # is estimated with b_j. In pyfixest, question_id should be
    # provided as the first FE variable, to ensure we have an estimate
    # for each question_id (otherwise one question may be dropped to
    # avoid perfect multicolinearity)
    mod = pf.feols("brier_score ~ 1 | question_id + model", data=df)
    dict_question_fe = mod.fixef()["C(question_id)"]
    if len(dict_question_fe) != len(df["question_id"].unique()):
        raise ValueError(
            f"Estimated num. of question fixed effects ({len(dict_question_fe)}) \
                not equal to num. of questions ({len(df['question_id'].unique())})"
        )

    # Merge question FE back to the original df
    df_question_fe = pd.DataFrame(
        [
            {"question_id": key, "question_fe": value}
            for key, value in dict_question_fe.items()
        ]
    )
    df = pd.merge(df, df_question_fe, on="question_id", how="left")
    df["diff_adj_brier"] = df["brier_score"] - df["question_fe"]

    # Calculate average difficulty-adj. Brier score per model
    model_scores = df.groupby("model")["diff_adj_brier"].mean().reset_index()
    model_scores.rename(columns={"diff_adj_brier": "avg_diff_adj_brier"}, inplace=True)

    # Rank models (lower Brier score is better)
    model_scores["rank"] = model_scores["avg_diff_adj_brier"].rank(
        ascending=True, method="min"
    )
    model_scores = model_scores.sort_values(by="rank")

    return model_scores


def rank_by_bss(df, ref_model="Naive Forecaster", type="percent"):
    df = df.copy()
    df["brier_score"] = brier_score(df)

    # Get reference model's predictions
    mask = df["model"] == ref_model
    ref_data = df.loc[mask,].copy()
    if len(ref_data) == 0:
        raise ValueError(f"Reference model '{ref_model}' not found in data")

    # Create mapping of question_id to reference brier score
    ref_brier_by_question = ref_data.set_index("question_id")["brier_score"].to_dict()

    # Only keep questions that the reference model predicted
    mask = df["question_id"].isin(ref_brier_by_question.keys())
    df_filtered = df.loc[mask,].copy()

    # Calculate Brier skill score per question
    df_filtered["ref_brier"] = df_filtered["question_id"].map(ref_brier_by_question)
    if type == "percent":
        df_filtered["bss"] = np.where(
            df_filtered["ref_brier"] > 0,
            1 - (df_filtered["brier_score"] / df_filtered["ref_brier"]),
            np.nan,
        )
    elif type == "absolute":
        df_filtered["bss"] = df_filtered["ref_brier"] - df_filtered["brier_score"]
    else:
        raise ValueError(f"Unkown type: '{type}'")

    # Calculate average scores by model
    model_scores = df_filtered[["model", "bss"]].groupby("model").mean().reset_index()
    model_scores.rename(columns={"bss": "avg_bss"}, inplace=True)

    # Rank by BSS (higher is better)
    model_scores["rank"] = model_scores["avg_bss"].rank(ascending=False, method="min")
    model_scores = model_scores.sort_values(by="rank").reset_index(drop=True)

    return model_scores


def rank_by_peer_score(df):
    df = df.copy()
    df["brier_score"] = brier_score(df)

    # For each question, calculate average Brier score
    df["question_avg_brier"] = df.groupby("question_id")["brier_score"].transform(
        "mean"
    )

    # Calculate peer score (positive is better than average)
    df["peer_score"] = df["question_avg_brier"] - df["brier_score"]

    # Average peer score per model
    model_scores = df[["model", "peer_score"]].groupby("model").mean().reset_index()
    model_scores.rename(columns={"peer_score": "avg_peer_score"}, inplace=True)
    model_scores["rank"] = model_scores["avg_peer_score"].rank(
        ascending=False, method="min"
    )
    model_scores = model_scores.sort_values(by="rank")

    return model_scores


# ==================
# Evaluation metrics
# ==================


def spearman_correlation(df_true_ranking, df_sim_ranking):
    # Create a merged df
    df_merged = pd.merge(
        df_true_ranking,
        df_sim_ranking,
        on="model",
        how="left",
        validate="1:1",
        suffixes=("_true", "_sim"),
    )

    # Calculate correlation coefficient
    corr = df_merged[["rank_true", "rank_sim"]].corr().values[0, 1]
    return corr


def top_k_retention(df_true_ranking, df_sim_ranking, k=20):
    """Calculate percentage of true top-k models that remain in simulated top-k"""

    # Get all models with rank <= k (handles ties properly)
    # Note that there may be more than k such models if there are ties
    true_top_k = set(df_true_ranking[df_true_ranking["rank_true"] <= k]["model"])
    sim_top_k = set(df_sim_ranking[df_sim_ranking["rank_sim"] <= k]["model"])

    # Calculate retention rate
    true_top_k_in_sim = true_top_k & sim_top_k
    retention = len(true_top_k_in_sim) / len(true_top_k)
    return retention


def median_displacement(df_true_ranking, df_sim_ranking):
    # Create a merged df
    df_merged = pd.merge(
        df_true_ranking,
        df_sim_ranking,
        on="model",
        how="left",
        validate="1:1",
        suffixes=("_true", "_sim"),
    )

    # Calculate correlation coefficient
    df_merged["displacement"] = (df_merged["rank_true"] - df_merged["rank_sim"]).abs()
    median_displacement = df_merged["displacement"].median()
    return median_displacement


def ranking_sanity_check(
    df_true_ranking, df_sim_ranking, model_list, pct_point_tol=0.05, verbose=True
):
    """Simple sanity test that checks whether models in model_list
    are close to their "true" position in the simulated ranking"""

    def get_pct_rank(df, model, rank_name):
        """Helper function to get percentage rank for a model in a dataframe"""
        mask = df["model"] == model
        model_rank = df.loc[mask, rank_name].values[0]
        max_rank = df[rank_name].max()
        return model_rank / max_rank

    test_passed = True

    for model in model_list:
        # Check if model exists in both dataframes
        if model not in df_true_ranking["model"].values:
            if verbose:
                print(f"Warning: Model '{model}' not found in df_true_ranking")
            return np.nan

        if model not in df_sim_ranking["model"].values:
            if verbose:
                print(f"Warning: Model '{model}' not found in df_sim_ranking")
            return np.nan

        # If model exists in both, proceed with the check
        model_pct_true_rank = get_pct_rank(
            df_true_ranking, model, rank_name="rank_true"
        )
        model_pct_sim_rank = get_pct_rank(df_sim_ranking, model, rank_name="rank_sim")

        # Check if the percentile ranks are within tolerance
        if np.abs(model_pct_true_rank - model_pct_sim_rank) < pct_point_tol:
            test = True
        else:
            test = False
        test_passed = test_passed & test

    return test_passed * 1.0


# ==========
# Simulation
# ==========


def simulate_random_sampling(df, n_questions_per_model, ref_model="Always 0.5"):
    """Simulate a dataset by drawing a n_questions_per_model random questions
    (sampled with replacement) from the full sample of questions. ref_model
    answers all questions"""
    # Get parameters
    models = df["model"].unique()
    questions = df["question_id"].unique()
    n_models = len(models)
    n_questions = len(questions)

    # Check if ref_model exists
    if ref_model is not None and ref_model in models:
        # Find ref_model index
        ref_model_idx = np.where(models == ref_model)[0][0]

        # Create indices for all OTHER models
        other_model_indices = np.arange(n_models)
        other_model_indices = other_model_indices[other_model_indices != ref_model_idx]

        # Sample for other models
        all_model_indices = np.repeat(other_model_indices, n_questions_per_model)
        all_question_indices = np.concatenate(
            [
                np.random.choice(n_questions, size=n_questions_per_model, replace=True)
                for _ in range(len(other_model_indices))
            ]
        )

        # Append reference model with ALL questions at the end
        all_model_indices = np.concatenate(
            [all_model_indices, np.repeat(ref_model_idx, n_questions)]
        )
        all_question_indices = np.concatenate(
            [all_question_indices, np.arange(n_questions)]
        )
    else:
        raise ValueError("Reference model not provided.")

    # Convert indices to actual model/question values
    sampled_models = models[all_model_indices]
    sampled_questions = questions[all_question_indices]

    # Create DataFrame of sampled (model, question) pairs
    # Include a sample_id to handle duplicates from sampling with replacement
    df_samples = pd.DataFrame(
        {
            "model": sampled_models,
            "question_id": sampled_questions,
            "sample_id": np.arange(len(sampled_models)),  # Unique ID for each sample
        }
    )

    # Single merge operation to get all data
    # The sample_id ensures we keep duplicates when sampling with replacement
    df_results = df_samples.merge(
        df[["model", "question_id", "forecast", "resolved_to", "question_type"]],
        on=["model", "question_id"],
        how="left",
    )

    # Clean up
    df_results = df_results.reset_index(drop=True)

    return df_results


def simulate_round_based(
    df,
    n_rounds=15,
    questions_per_round=25,
    models_per_round_mean=40,
    ref_model="Always 0.5",
):
    """
    Simulate dataset using round-based sampling.

    Each round samples questions_per_round questions (with replacement)
    from the full set of questions. For each round, we sample
    which models participate. When a model participates in a round,
    it answers ALL questions in that round.
    """
    # Get parameters
    models = df["model"].unique()
    questions = df["question_id"].unique()

    # Check if ref_model exists
    if ref_model is None or ref_model not in models:
        raise ValueError("Reference model not provided or not found in data.")

    # Create list of non-reference models
    other_models = [m for m in models if m != ref_model]
    n_non_reference_models = len(other_models)

    # Create rounds by sampling questions with replacement
    rounds = []
    for round_id in range(n_rounds):
        # Sample questions with replacement for this round
        round_questions = np.random.choice(
            questions, size=questions_per_round, replace=True
        )

        # Sample number of models for this round (Poisson, but
        # at least 1 non-ref model, and less than total available
        # models)
        n_models_this_round = max(1, np.random.poisson(models_per_round_mean))
        n_models_this_round = min(n_models_this_round, n_non_reference_models)

        # Randomly select which models participate in this round;
        # sampling without replacement
        selected_models = np.random.choice(
            other_models, size=n_models_this_round, replace=False
        )

        # Add reference model to the list
        round_models = [ref_model] + selected_models.tolist()

        rounds.append(
            {
                "round_id": round_id,
                "round_questions": round_questions,
                "round_models": round_models,
            }
        )

    # Create a clean pandas df
    data_rows = []

    for round_info in rounds:
        round_id = round_info["round_id"]
        round_questions = round_info["round_questions"]
        round_models = round_info["round_models"]

        # Create entries for each model-question pair in this round
        for model in round_models:
            for question_id in round_questions:
                data_rows.append(
                    {"model": model, "question_id": question_id, "round_id": round_id}
                )

    # Create DataFrame from all samples
    df_samples = pd.DataFrame(data_rows)

    # Merge with original data to get forecasts and outcomes
    # round_id ensures uniqueness during merge
    df_results = df_samples.merge(
        df[["model", "question_id", "forecast", "resolved_to", "question_type"]],
        on=["model", "question_id"],
        how="left",
    )

    return df_results


def evaluate_ranking_methods(
    df,
    ranking_methods,
    evaluation_metrics,
    simulation_func,
    simulation_kwargs,
    n_simulations=1000,
    dataset_weight=0.5,
    ref_model="Always 0.5",
):
    # Calculate true ranking with dataset/market weighting
    df_true_ranking = rank_with_weighting(
        df=df,
        ranking_func=rank_by_brier,
        metric_name="avg_brier",
        is_lower_metric_better=True,
        dataset_weight=dataset_weight,
    )
    df_true_ranking = df_true_ranking.rename(columns={"rank": "rank_true"})

    # Run simulations
    results_list = []
    error_count = 0

    for sim in range(n_simulations):
        # Generate simulated dataset using the provided simulation function
        df_sim = simulation_func(df=df, ref_model=ref_model, **simulation_kwargs)

        # Evaluate each ranking method
        for method_name, method_info in ranking_methods.items():
            try:
                # Unpack method information
                method_func, metric_name, is_lower_better, method_kwargs = method_info

                # Apply ranking with dataset/market weighting
                df_sim_ranking = rank_with_weighting(
                    df=df_sim,
                    ranking_func=method_func,
                    metric_name=metric_name,
                    is_lower_metric_better=is_lower_better,
                    dataset_weight=dataset_weight,
                    **method_kwargs,
                )
                df_sim_ranking = df_sim_ranking.rename(columns={"rank": "rank_sim"})

                # Create result row for this simulation and method
                result_row = {"simulation_id": sim, "method": method_name}

                # Evaluate using each metric
                for metric_name, metric_info in evaluation_metrics.items():
                    metric_func, metric_kwargs = metric_info
                    value = metric_func(
                        df_true_ranking=df_true_ranking,
                        df_sim_ranking=df_sim_ranking,
                        **metric_kwargs,
                    )
                    result_row[metric_name] = value

                results_list.append(result_row)

            except Exception as e:
                error_count += 1
                print(f"Error in {method_name} at simulation {sim}: {e}")
                continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)

    # Reorder columns: simulation_id, method, then metrics
    metric_cols = list(evaluation_metrics.keys())
    col_order = ["simulation_id", "method"] + metric_cols
    df_results = df_results[col_order]

    return df_results, error_count


# =================
# Utility functions
# =================


def combine_rankings(
    df_dataset_ranking,
    df_market_ranking,
    metric_name,
    is_lower_metric_better=False,
    dataset_weight=0.5,
):
    # Merge rankings
    df_combined = pd.merge(
        df_dataset_ranking,
        df_market_ranking,
        on="model",
        how="outer",  # To capture the edge case when the model does not
        # not answer any dataset questions
        suffixes=("_dataset", "_market"),
        validate="1:1",
    )

    # For metrics: use the available value
    df_combined[f"{metric_name}_dataset"] = df_combined[
        f"{metric_name}_dataset"
    ].fillna(df_combined[f"{metric_name}_market"])
    df_combined[f"{metric_name}_market"] = df_combined[f"{metric_name}_market"].fillna(
        df_combined[f"{metric_name}_dataset"]
    )

    # Calculate weighted average for the metric
    market_weight = 1 - dataset_weight
    df_combined[f"{metric_name}_weighted"] = (
        dataset_weight * df_combined[f"{metric_name}_dataset"]
        + market_weight * df_combined[f"{metric_name}_market"]
    )

    # Rank based on the weighted metric
    if is_lower_metric_better:
        ascending = True
    else:
        ascending = False
    df_combined["rank"] = df_combined[f"{metric_name}_weighted"].rank(
        ascending=ascending, method="min"
    )

    # Prepare output
    df_combined = df_combined[
        [
            "model",
            f"{metric_name}_dataset",
            f"{metric_name}_market",
            f"{metric_name}_weighted",
            "rank",
        ]
    ]
    df_combined = df_combined.sort_values(by="rank").reset_index(drop=True)

    return df_combined


def rank_with_weighting(
    df,
    ranking_func,
    metric_name,
    is_lower_metric_better=False,
    dataset_weight=0.5,
    **kwargs,
):
    if "question_type" not in df.columns:
        raise ValueError("question_type not found in the data")

    # Split by question type
    df_dataset = df[df["question_type"] == "dataset"].copy()
    df_market = df[df["question_type"] == "market"].copy()

    # Apply ranking to each subset
    df_dataset_ranking = ranking_func(df_dataset, **kwargs)
    df_market_ranking = ranking_func(df_market, **kwargs)

    return combine_rankings(
        df_dataset_ranking=df_dataset_ranking,
        df_market_ranking=df_market_ranking,
        metric_name=metric_name,
        is_lower_metric_better=is_lower_metric_better,
        dataset_weight=dataset_weight,
    )
