import json
from collections.abc import Callable
from os.path import join
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, DatasetEncoding, SimpleEncoding

# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import SimpleModel, elastic_net_predictor


def evaluate_forecast(y_true, y_pred):
    diff = y_pred - y_true
    country_error = diff.abs().sum()
    portfolio_country_error = diff.sum()
    return country_error, abs(portfolio_country_error)


def cross_validate_forecaster(
    predictor: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    verbose=True,
    save_path=None,
    n_splits=5,
):
    """
    Perform time-series cross-validation.
    Returns:
      mean_abs_err, mean_port_err, mean_final_score,
      std_abs_err,  std_port_err,  std_final_score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_absolute_errors = []
    all_portfolio_errors = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y.index), start=1):
        train_dates = pd.to_datetime(y.index[train_idx])
        test_dates = pd.to_datetime(y.index[test_idx])

        if len(test_dates) == 0:
            continue

        y_train = y.loc[train_dates]
        y_test = y.loc[test_dates]

        X_train = X.loc[train_dates]
        X_test = X.loc[test_dates]

        y_hat = predictor(X_train, y_train, X_test)

        country_err, portfolio_err = evaluate_forecast(y_test, y_hat)

        # Sum of absolute errors for that fold
        mean_fold_abs = country_err
        mean_fold_port = portfolio_err

        # Example scoring formula (your logic may differ):
        final_fold_score = (
            1.0 * mean_fold_abs
            + 5.0 * mean_fold_abs
            + 10.0 * mean_fold_port
            + 50.0 * mean_fold_port
        )
        # => 6.0 * mean_fold_abs + 60.0 * mean_fold_port

        all_absolute_errors.append(mean_fold_abs)
        all_portfolio_errors.append(mean_fold_port)
        fold_scores.append(final_fold_score)

    # Final metrics across folds
    mean_abs_err = np.mean(all_absolute_errors)
    mean_port_err = np.mean(all_portfolio_errors)
    mean_final_score = np.mean(fold_scores)

    std_abs_err = np.std(all_absolute_errors)
    std_port_err = np.std(all_portfolio_errors)
    std_final_score = np.std(fold_scores)

    return (
        mean_abs_err,
        mean_port_err,
        mean_final_score,
        std_abs_err,
        std_port_err,
        std_final_score,
    )


def evaluate(X: pd.DataFrame, y: pd.Series, save_path: str, my_predictor):
    """
    Runs cross-validation with a given predictor.
    Returns:
      abs_err, port_err, score, abs_err_std, port_err_std, score_std
    """
    results = cross_validate_forecaster(
        predictor=my_predictor,
        X=X,
        y=y,
        verbose=True,
        save_path=save_path,  # not currently used to save anything, but left for clarity
    )

    (abs_err,
     port_err,
     score,
     abs_err_std,
     port_err_std,
     score_std) = results

    return abs_err, port_err, score, abs_err_std, port_err_std, score_std


def main(model_name: str, my_predictor, max_customers=10):
    """
    Train and evaluate the models for IT and ES,
    then store one final score (sum of ES and IT),
    and also store the std dev of all absolute errors across both zones.
    """

    # We'll accumulate each zone's results in memory
    zone_final_scores = {}
    zone_errors_dataframes = {}

    for zone in ["ES", "IT"]:
        # Inputs
        input_path = r"datasets2025"
        output_path = r"outputs"

        # Load Datasets
        loader = DataLoader(input_path)
        training_set, features, example_results = loader.load_data(zone)

        # Additional data
        rollout, holidays = loader.load_additional_data(zone)

        # Data Manipulation and Training
        end_training = training_set.index.max()
        start_forecast, end_forecast = example_results.index[0], example_results.index[-1]

        dataset_encoding = DatasetEncoding(
            training_set,
            features,
            rollout,
            holidays,
            end_training=end_training,
            start_forecast=start_forecast,
            end_forecast=end_forecast,
        )

        range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1H")
        forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)
        forecast_step = 1

        # We'll store one row per-customer
        errors = pd.DataFrame(
            columns=[
                "abs_err",
                "port_err",
                "abs_err_std",
                "port_err_std",
                # We'll add "raw_cv_score" and "raw_cv_score_std" if we want to keep them
                "cv_score",
                "cv_score_std",
                "country",
            ]
        )

        customers = training_set.columns.values[:max_customers]

        for i, costumer in enumerate(customers, start=1):
            customer_id = int(costumer.split("_")[-1])
            # Progress bar
            bar_length = 30
            progress = int(bar_length * i / max_customers)
            bar = "#" * progress + "-" * (bar_length - progress)
            print(f"\r[{bar}] {i}/{max_customers} customers processed", end="", flush=True)

            df = dataset_encoding.generate_dataset(
                customer_id,
                window_size=24 * 7,
                forecast_skip=1,
                forecast_horizon=1,
                additional_feats=["mean", "std", "skew", "kurtosis", "min", "max"],
            )

            # Evaluate
            X, y = dataset_encoding.get_train_data(
                df, customer_id, forecast_step=forecast_step, drop_nans_X=True
            )

            (
                abs_err,
                port_err,
                cv_score,
                abs_err_std,
                port_err_std,
                cv_score_std,
            ) = evaluate(X, y, f"{output_path}/{customer_id}.png", my_predictor)

            errors.loc[customer_id] = {
                "abs_err": abs_err,
                "port_err": port_err,
                "abs_err_std": abs_err_std,
                "port_err_std": port_err_std,
                "cv_score": cv_score,
                "cv_score_std": cv_score_std,
                "country": zone,
            }

        # Now compute final "score" per-customer with the zone-specific weighting
        # (Stays the same as before, purely example logic).
        errors["score"] = errors.apply(
            lambda x: (x["abs_err"] * 5 + x["port_err"] * 50)
            if x["country"] == "ES"
            else (x["abs_err"] * 1 + x["port_err"] * 10),
            axis=1,
        )

        # Save the CSV for this zone
        errors.to_csv(f"{output_path}/errors_{zone}.csv")

        # Final "average score" across all customers for THIS zone
        final_score_zone = float(errors["score"].mean())
        zone_final_scores[zone] = final_score_zone

        # Keep the DataFrame in memory for post-processing
        zone_errors_dataframes[zone] = errors

        print(f"\n\n===== ZONE: {zone} =====")
        print(f"Final average score: {final_score_zone:.2f}")
        print("========================\n")

    # -----------------------
    # After processing both zones, we compute:
    #  1) A single final score = sum of the ES and IT final scores
    #  2) The overall std of all absolute errors across *both* zones
    # -----------------------
    final_score = zone_final_scores["ES"] + zone_final_scores["IT"]
    combined_errors = pd.concat(zone_errors_dataframes.values())

    # "The standard deviation of all errors" typically means the std of abs_err across the entire dataset
    overall_std_abs_err = float(combined_errors["abs_err"].std(ddof=1) / np.sqrt(len(combined_errors)))

    # Build one single record for the JSON
    record = {
        "model": model_name,
        # single final score (sum of zone-specific final scores)
        "final_score": final_score,
        # standard deviation of all absolute errors across ES & IT
        "std_abs_err": overall_std_abs_err,
    }

    # You might also decide to store "std of score" or portfolio errors here.
    # For example:
    # record["std_score"] = float(combined_errors["score"].std())

    # Append to results.json (as one line)
    with open("results.json", "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")

    print("===== FINAL COMBINED =====")
    print(f"Model: {model_name}")
    print(f"Final Summed Score (ES+IT): {final_score:.2f}")
    print(f"Std of all absolute errors: {overall_std_abs_err:.2f}")
    print("========================\n")


if __name__ == "__main__":
    model_name = "elastic_net_predictor"
    max_columns=100
    predictor=elastic_net_predictor
    main(model_name, predictor, max_columns)
