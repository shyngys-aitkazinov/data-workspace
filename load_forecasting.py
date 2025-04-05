import json
from collections.abc import Callable
from os.path import join
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# from data import DataLoader, DatasetEncoding, SimpleEncoding
# from forecast_models import SimpleModel, elastic_net_predictor

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series):
    """
    Returns:
      - country_error: sum of absolute differences (abs err)
      - portfolio_country_error: absolute sum of the *signed* differences (port err)
      - raw_errors: the vector of raw (y_pred - y_true) for stdev calculations
    """
    diff = y_pred - y_true
    country_error = diff.abs().sum()
    portfolio_country_error = diff.sum()  # signed sum
    return country_error, abs(portfolio_country_error), diff  # also return raw diffs


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
      mean_abs_err,
      mean_port_err,
      mean_final_score,
      all_fold_diffs (the concatenation of raw errors for each fold),
        so you can compute an overall stdev as needed
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_absolute_errors = []  # store fold-wise sum of absolute errors
    all_portfolio_errors = [] # store fold-wise sum of portfolio errors
    fold_scores = []
    
    # We'll also keep track of the raw differences from each fold
    all_fold_diffs = []  # (this will be a list of arrays/vectors)

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

        country_err, portfolio_err, raw_diffs = evaluate_forecast(y_test, y_hat)

        mean_fold_abs = country_err  # sum of absolute errors for that fold
        mean_fold_port = portfolio_err

        # Example scoring formula (your logic may differ):
        final_fold_score = 6.0 * mean_fold_abs + 60.0 * mean_fold_port

        # Save sums for stats
        all_absolute_errors.append(mean_fold_abs)
        all_portfolio_errors.append(mean_fold_port)
        fold_scores.append(final_fold_score)

        # Save raw diffs for stdev
        all_fold_diffs.append(raw_diffs.values)

        if verbose:
            print(f"\n✅ Fold {fold} completed")
            print(f"   ├─ Mean Absolute Error (this fold):  {mean_fold_abs:.2f}")
            print(f"   ├─ Mean Portfolio Error (this fold): {mean_fold_port:.2f}")
            print(f"   └─ Final Fold Score: {np.round(final_fold_score, 2)}")

    # Final metrics across folds
    mean_absolute_error = np.mean(all_absolute_errors)
    mean_portfolio_error = np.mean(all_portfolio_errors)
    mean_final_score = np.mean(fold_scores)

    # Concatenate the list of raw_diffs into one big array
    # so you can compute the stdev of all errors across folds
    if len(all_fold_diffs) > 0:
        all_fold_diffs = np.concatenate(all_fold_diffs)  # shape: (sum of test sizes,)
    else:
        # Just in case there's no data
        all_fold_diffs = np.array([])

    return mean_absolute_error, mean_portfolio_error, mean_final_score, all_fold_diffs


def evaluate(X: pd.DataFrame, y: pd.Series, save_path: str, predictor: Callable):
    """
    Runs cross-validation with a given predictor and returns:
      abs_err, port_err, score, all_diffs (raw differences)
    """
    abs_err, port_err, score, raw_diffs = cross_validate_forecaster(
        predictor=predictor,
        X=X,
        y=y,
        verbose=True,
        save_path=save_path,  # not currently used to save anything
    )
    return abs_err, port_err, score, raw_diffs


def main(model_name: str):
    """
    Train and evaluate the models for IT and ES.
    
    Instead of writing to JSON twice (once per zone),
    we will accumulate results and produce *one* final JSON record.
    
    The final JSON record will have:
      - "final_score": sum of the final scores from both zones
      - "std_errors": standard deviation across *all* errors from both zones
    """

    # For demonstration, let's just define a dummy predictor here
    # that always returns y_train.mean() as a constant forecast.
    # In your real code, you would import from forecast_models, e.g. `elastic_net_predictor`
    def dummy_predictor(X_train, y_train, X_test):
        return pd.Series(y_train.mean(), index=X_test.index)

    zones = ['ES', 'IT']
    
    # We'll accumulate the zone final scores, and *all* raw diffs across zones
    zone_scores = []
    global_diffs = []  # hold all raw error values from all folds/customers across zones
    
    for zone in zones:
        print(f"\n=== Processing zone: {zone} ===\n")

        # -------------------------------------------------
        # In your real code, replace these next lines with:
        # loader = DataLoader(input_path)
        # training_set, features, example_results = loader.load_data(zone)
        # rollout, holidays = loader.load_additional_data(zone)
        # ...
        # For a minimal example, let's define dummy data:
        dates = pd.date_range("2020-01-01", periods=200, freq="H")
        training_set = pd.DataFrame({
            f"consumption_{i}": np.random.rand(len(dates)) * 100
            for i in range(1, 6)  # 5 "customers" as example
        }, index=dates)
        # Just treat X = time features and y = consumption for simplicity:
        features = training_set.copy()  # dummy
        # -------------------------------------------------

        # Prepare an output DataFrame to track per-customer errors
        errors = pd.DataFrame(columns=["abs_err", "port_err", "std_err", "country", "score"])

        # Evaluate each "customer" consumption column
        for costumer in training_set.columns.values:
            customer_id = int(costumer.split("_")[-1])
            print(f"----------------------------------------")
            print(f"Start customer {customer_id} (zone={zone})")

            # Build X, y for cross-validation from your real pipeline
            # In your actual code, you'd do something like:
            #   df = dataset_encoding.generate_dataset(...)
            #   X, y = dataset_encoding.get_train_data(...)
            # For demonstration, let's define:
            y = training_set[costumer]
            X = features.drop(columns=[col for col in training_set.columns
                                       if col != costumer])  # just pretend

            # Run cross-validation
            abs_err, port_err, cv_score, raw_diffs = evaluate(
                X, y,
                save_path=f"outputs/{customer_id}.png",  # in real usage
                predictor=dummy_predictor  # in real usage: elastic_net_predictor
            )

            # If you want a single stdev across folds for this customer,
            # compute it from `raw_diffs`:
            customer_std = float(np.ste(raw_diffs)) if len(raw_diffs) > 0 else np.nan

            # Weighted scoring example (same logic as your code):
            # If zone=ES => (abs_err * 5 + port_err * 50)
            # If zone=IT => (abs_err * 1 + port_err * 10)
            if zone == 'ES':
                cust_score = abs_err * 5 + port_err * 50
            else:  # IT
                cust_score = abs_err * 1 + port_err * 10

            # Add row to our "errors" DataFrame
            errors.loc[customer_id] = {
                "abs_err": abs_err,
                "port_err": port_err,
                "std_err": customer_std,
                "country": zone,
                "score": cust_score
            }

            # We'll also add these raw differences to the global list
            # so that at the very end, we can compute overall stdev across
            # *all* predictions for both zones:
            global_diffs.append(raw_diffs)

        # Save the CSV for this zone
        errors.to_csv(f"errors_{zone}.csv", index=True)

        # We'll keep track of the *mean* final score for the zone (across customers):
        zone_final_score = float(errors['score'].mean())
        zone_scores.append(zone_final_score)

        print(f"\n===== ZONE: {zone} =====")
        print(f"Final average score (zone): {zone_final_score:.2f}")
        print("=============\n")

    # Now we have the final average scores from both zones. Sum them:
    total_score = sum(zone_scores)

    # Compute the global stdev across *all* raw diffs from both zones
    if len(global_diffs) > 0:
        global_diffs = np.concatenate(global_diffs)
        global_stdev = float(np.std(global_diffs))
    else:
        global_stdev = float('nan')

    print("\n==========================")
    print("✅ All zones processed!")
    print(f"   Single final_score = {total_score:.2f}")
    print(f"   Global stdev of all errors = {global_stdev:.4f}")
    print("==========================")

    # Write *one* record to JSON
    record = {
        "model": model_name,
        "final_score": total_score,   # sum of zone scores
        "std_errors": global_stdev,   # stdev across all diffs
    }

    with open("results.json", "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "elastic_net_predictor"

    main(model_name)
