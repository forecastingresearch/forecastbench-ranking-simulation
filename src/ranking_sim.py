import numpy as np
import pandas as pd
import pyfixest as pf
from tqdm import tqdm

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
        # drop combo questions
        df_temp = df_temp[df_temp["direction"] == ()]
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

    # Return empty results for empty input
    if len(df) == 0:
        return pd.DataFrame(
            {
                "model": pd.Series(dtype="object"),
                "avg_diff_adj_brier": pd.Series(dtype="float64"),
                "rank": pd.Series(dtype="int64"),
            }
        )

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
    df["brier_score"] = brier_score(df)

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

    # Return empty results for empty input
    if len(df) == 0:
        return pd.DataFrame(
            {
                "model": pd.Series(dtype="object"),
                "avg_bss": pd.Series(dtype="float64"),
                "rank": pd.Series(dtype="int64"),
            }
        )

    # Otherwise, calculate Brier skill score (BSS)
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
    # Extract variables
    questions = df["question_id"].unique()
    models = df["model"].unique()

    # Check if ref_model exists
    if ref_model is None or ref_model not in models:
        raise ValueError("Reference model not provided.")

    other_models = [model for model in models if model != ref_model]
    n_other_models = len(other_models)

    # Draw questions for non-reference models
    df_samples = pd.DataFrame()
    df_samples["model"] = np.repeat(other_models, n_questions_per_model)
    df_samples["question_id"] = np.random.choice(
        questions, size=n_questions_per_model * n_other_models, replace=True
    )

    # Calculate number of occurences of question_id for a given model.
    # This is for getting a unique primary key for simulated questions.
    # This approach treats the same question_id (from the original dataset)
    # resampled k times as k different questions (i.e., k different
    # sim_question_id's)
    df_samples = df_samples.sort_values(["model", "question_id"]).reset_index(drop=True)
    df_samples["occ_question_id"] = (
        df_samples.groupby(["model", "question_id"]).cumcount() + 1
    )

    # Create a unique primary key for simulated questions
    df_samples["sim_question_id"] = (
        df_samples["question_id"] + "-" + df_samples["occ_question_id"].astype(str)
    )

    # Add reference model
    df_temp = df_samples[["question_id", "sim_question_id"]].copy().drop_duplicates()
    df_temp["model"] = ref_model
    df_samples = pd.concat([df_samples, df_temp], ignore_index=True)

    # Get data on forecasts and realizations from the original dataset
    df_results = df_samples.merge(
        df[["model", "question_id", "forecast", "resolved_to", "question_type"]],
        on=["model", "question_id"],
        how="left",
    )

    # Clean up
    df_results["question_id"] = df_results["sim_question_id"]
    df_results = df_results.drop(["occ_question_id", "sim_question_id"], axis=1)
    df_results = df_results.reset_index(drop=True)

    return df_results


def fixed_dataset_market_question_sample(df, n):
    """
    Sample questions in a similar way to sampling done on ForecastBench.

    Aim for a 50:50 split between dataset questions and market questions when there is
    only one forecast horizon. As the number of available horizons increases, the
    proportion of dataset questions to market questions increases.
    """
    groups = df.groupby(["question_type", "horizon"], dropna=False)[
        "question_id"
    ].unique()

    dataset_groups = groups["dataset"]
    n_horizons = len(dataset_groups)

    # Number of full sets of dataset (n_horizons) and market questions (1) that fit in
    # `n`
    n_dataset = n // (n_horizons + 1) * n_horizons

    # Number of dataset questions per horizon
    n_dataset_horizon = n_dataset // n_horizons

    # The remainder
    n_market = n - n_dataset

    # Since sampling is done with replacement, we need at least
    # 1 market and at least 1 dataset questions, and the required
    # number of market questions to sample should be >= 1
    if (
        n_dataset_horizon < 1
        or n_market < 1
        or len(dataset_groups.values[0]) < 1  # At least one dataset question exists
        or len(df[df["question_type"] == "market"])
        < 1  # At least one market question exists
    ):
        raise ValueError(
            f"`fixed_dataset_market_question_sample()` needs a bigger `n`. It was "
            f"provided n={n}, which caused n_dataset_horizon=={n_dataset_horizon} and "
            f"n_market=={n_market}.\n"
        )

    # Market questions: choose randomly across all market questions
    all_market_questions = np.concatenate([g for g in groups["market"].values])
    market_questions = np.random.choice(
        all_market_questions, size=n_market, replace=True
    )

    # Dataset questions: choose randomly for one horizon, then get the same questions
    # at all horizons
    df0 = df[df["question_id"].isin(dataset_groups.values[0])]
    sampled_rows = df0.sample(n=n_dataset_horizon, replace=True)
    dataset_questions_list = []
    for _, row in sampled_rows.iterrows():
        subset = df[
            (df["source"] == row["source"])
            & (df["id"] == row["id"])
            & (df["forecast_due_date"] == row["forecast_due_date"])
        ]
        dataset_questions_list.extend(subset["question_id"].unique())
    dataset_questions = np.array(dataset_questions_list)

    return np.concatenate([market_questions, dataset_questions])


def simulate_round_based(
    df,
    n_rounds=15,
    questions_per_round=25,
    models_per_round_mean=40,
    ref_model="Always 0.5",
    skill_temperature=None,
    difficulty_temperature=None,
    model_persistence=0.0,
    fixed_models_per_round=False,
    fixed_dataset_market_question_sampling=False,
):
    """
    Simulate dataset using round-based sampling.

    Each round samples questions_per_round questions (with replacement)
    from the full set of questions. For each round, we sample
    which models participate. When a model participates in a round,
    it answers ALL questions in that round.

    In this function, skill_temperature and difficulty_temperature
    are used to model potential model and/or question drift over time,
    while model_persistence is used to control how likely a model
    to participate in round R + 1, given that it participated in round R.

    In particular:

    - skill_temperature: Controls bias toward higher-skill models.
        - None or 0 = uniform sampling
        - > 0 = favor better models (lower Brier scores)
        - Can be a float (constant) or function of round number
    - difficulty_temperature: Controls question difficulty bias.
        - None or 0 = uniform sampling
        - > 0 = favor harder questions
        - < 0 = favor easier questions
        - Can be a float (constant) or function of round number
    - model_persistence: Probability (0-1) that a model continues to the next round.
        - 0 = no persistence
        - 0.7 = 70% of models continue (randomly selected)
        - 1 = all models continue

    """

    # Get parameters
    models = df["model"].unique()
    questions = df["question_id"].unique()

    # Check input data
    if not isinstance(model_persistence, float):
        raise ValueError("model_persistence must be a float")

    if not (0 <= model_persistence <= 1):
        raise ValueError("model_persistence must be between 0 and 1")

    if ref_model is None or ref_model not in models:
        raise ValueError("Reference model not provided or not found in data.")

    if fixed_dataset_market_question_sampling and difficulty_temperature is not None:
        raise ValueError(
            "Cannot use `fixed_dataset_market_question_sampling` and "
            "`difficulty_temperature`."
        )

    # Create list of non-reference models
    other_models = [m for m in models if m != ref_model]
    n_non_reference_models = len(other_models)

    # Pre-compute skill and difficulty if drift is enabled
    use_model_drift = skill_temperature is not None
    use_question_drift = difficulty_temperature is not None

    if use_model_drift or use_question_drift:
        df_temp = df.copy()
        df_temp["brier_score"] = brier_score(df_temp)
        df_temp["brier_skill"] = (-1) * df_temp["brier_score"]  # Lower Brier is better

    if use_model_drift:
        # TODO: Add weighting by dataset/market questions;
        # Currently, the codebase doesn't handle missing cases when no
        # dataset or market questions are present, causing some unit
        # tests to fail
        model_skills = (
            df_temp[["model", "brier_skill"]].groupby("model")["brier_skill"].mean()
        )
        other_model_skills = model_skills.loc[other_models].to_numpy(float)

    if use_question_drift:
        question_difficulties = (
            df_temp[["question_id", "brier_score"]]
            .groupby("question_id")["brier_score"]
            .mean()
        )
        question_difficulties = question_difficulties.loc[questions].to_numpy(float)

    # Create rounds by sampling questions with replacement
    rounds = []
    previous_round_models = []

    for round_id in range(n_rounds):
        # Get temperature values for this round
        alpha_r = temperature_lookup(skill_temperature, round_id)
        beta_r = temperature_lookup(difficulty_temperature, round_id)

        # Sample questions with replacement for this round
        if fixed_dataset_market_question_sampling:
            round_questions = fixed_dataset_market_question_sample(
                df=df, n=questions_per_round
            )
        else:
            question_probs = (
                softmax_weights(question_difficulties, beta_r)
                if use_question_drift
                else None
            )
            round_questions = np.random.choice(
                questions, size=questions_per_round, replace=True, p=question_probs
            )

        # Sample number of models for this round (either Poisson-distributed
        # or fixed)
        if not fixed_models_per_round:
            n_models_this_round = max(1, np.random.poisson(models_per_round_mean))
            n_models_this_round = min(n_models_this_round, n_non_reference_models)
        elif fixed_models_per_round:
            n_models_this_round = models_per_round_mean
            if n_models_this_round > n_non_reference_models:
                raise ValueError(
                    "Fixed models per round exceeds available non-reference models."
                )

        # Sample models that participate in this round;
        # sampling WITHOUT replacement
        if round_id == 0:
            other_model_probs = (
                softmax_weights(other_model_skills, alpha_r)
                if use_model_drift
                else None
            )
            selected_models = np.random.choice(
                other_models,
                size=n_models_this_round,
                replace=False,
                p=other_model_probs,
            ).tolist()
        else:
            n_continuing_target = int(
                np.floor(model_persistence * len(previous_round_models))
            )
            if n_continuing_target > 0:
                continuing_models = np.random.choice(
                    previous_round_models,
                    size=n_continuing_target,
                    replace=False,
                ).tolist()
            else:
                continuing_models = []
            n_continuing_models = len(continuing_models)
            n_new_models = max(0, n_models_this_round - n_continuing_models)

            if n_new_models > 0:
                # Known edge case: Sometimes a model from previous_round_models
                # may be selected as a new model as it is included in
                # available_models below; that can increase the actual
                # model persistence rate above the nominal rate when
                # we need to sample many models
                available_models = [
                    m for m in other_models if m not in continuing_models
                ]
                if len(available_models) < n_new_models:
                    selected_models = continuing_models + available_models
                else:
                    if use_model_drift:
                        available_models_skills = model_skills.loc[
                            available_models
                        ].to_numpy(float)
                    available_models_probs = (
                        softmax_weights(available_models_skills, alpha_r)
                        if use_model_drift
                        else None
                    )
                    selected_new_models = np.random.choice(
                        available_models,
                        size=n_new_models,
                        replace=False,
                        p=available_models_probs,
                    ).tolist()
                    selected_models = continuing_models + selected_new_models
            elif n_new_models == 0:
                # Note that this can mean that the actual number
                # of models in this round is higher than the
                # randomly drawn n_models_this_round
                selected_models = continuing_models

        # Update previous round models for the next iteration
        previous_round_models = selected_models

        # Add reference model to the list
        round_models = [ref_model] + selected_models

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

    # Create a unique primary key for simulated questions.
    # A question with the same question_id (from the original dataset)
    # that is drawn in round R and R + 1 is treated as a different question
    # (i.e., will have different sim_question_id's). Also,
    # if the same question is drawn k > 1 times in the same round R,
    # it will also get different sim_question_id's.
    df_samples = df_samples.sort_values(
        ["model", "round_id", "question_id"], ascending=True
    ).reset_index(drop=True)
    df_samples["occ_question_id"] = (
        df_samples.groupby(["model", "round_id", "question_id"]).cumcount() + 1
    )

    # Create unique sim_question_id for within-round duplicates
    df_samples["sim_question_id"] = (
        df_samples["question_id"]
        + "-R"
        + df_samples["round_id"].astype(str)
        + "-"
        + df_samples["occ_question_id"].astype(str)
    )

    # Merge with original data to get forecasts and outcomes
    # round_id ensures uniqueness during merge
    df_results = df_samples.merge(
        df[["model", "question_id", "forecast", "resolved_to", "question_type"]],
        on=["model", "question_id"],
        how="left",
    )

    # Clean up
    df_results["question_id"] = df_results["sim_question_id"]
    df_results = df_results.drop(["occ_question_id", "sim_question_id"], axis=1)
    df_results = df_results.reset_index(drop=True)

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

    for sim in tqdm(range(n_simulations)):
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


def temperature_lookup(temperature, round_id):
    if temperature is None:
        return np.nan
    return temperature(round_id) if callable(temperature) else float(temperature)


def softmax_weights(x, temp):
    if np.isnan(temp):
        return None
    w = np.exp(temp * x)
    return w / w.sum()
