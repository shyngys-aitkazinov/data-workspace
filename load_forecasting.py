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
      mean_absolute_error, mean_portfolio_error, mean_final_score
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

        mean_fold_abs = country_err  # sum of absolute errors for that fold
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

        if verbose:
            print(f"\n✅ Fold {fold} completed")
            print(f"   ├─ Mean Absolute Error (per customer):  {mean_fold_abs:.2f}")
            print(f"   ├─ Mean Portfolio Error (per customer): {mean_fold_port:.2f}")
            print(f"   └─ Final Fold Score: {np.round(final_fold_score, 2)}")

    # Final metrics across folds
    mean_absolute_error = np.mean(all_absolute_errors)
    mean_portfolio_error = np.mean(all_portfolio_errors)
    mean_final_score = np.mean(fold_scores)

    return mean_absolute_error, mean_portfolio_error, mean_final_score


def evaluate(X: pd.DataFrame, y: pd.Series, save_path: str):
    """
    Runs cross-validation with a given predictor and returns:
      abs_err, port_err, score
    """
    abs_err, port_err, score = cross_validate_forecaster(
        predictor=elastic_net_predictor,
        X=X,
        y=y,
        verbose=True,
        save_path=save_path,  # not currently used to save anything, but left for clarity
    )

    return abs_err, port_err, score


def main(model_name: str):
    """
    Train and evaluate the models for IT and ES,
    then append the final average score to a JSON file.
    """

    for zone in ['ES', 'IT']:
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

        errors = pd.DataFrame(columns=["abs_err", "port_err", "country"])

        for costumer in training_set.columns.values:
            customer_id = int(costumer.split("_")[-1])
            print(f"******************************************")
            print(f"Start {customer_id}")

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

            abs_err, port_err, cv_score = evaluate(X, y, f"{output_path}/{customer_id}.png")
            errors.loc[customer_id] = {"abs_err": abs_err, "port_err": port_err, "country": zone}

        # Compute final per-customer 'score' based on country
        errors['score'] = errors.apply(
            lambda x: (x['abs_err'] * 5 + x['port_err'] * 50)
                      if x['country'] == 'ES'
                      else (x['abs_err'] * 1 + x['port_err'] * 10),
            axis=1
        )

        # Save the errors
        errors.to_csv(f"{output_path}/errors_{zone}.csv")

        # Append final average score to a JSON file
        final_score = float(errors['score'].mean())  # average across all customers for this zone
        record = {
            "model": model_name,
            "zone": zone,
            "final_score": final_score
        }

        with open("results.json", "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")

        print(f"\n===== ZONE: {zone} =====")
        print(f"Final average score: {final_score:.2f}")
        print("========================\n")


if __name__ == "__main__":
    # Example usage: pass the model name on the command line or hardcode it
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "elastic_net_predictor"

    main(model_name)
