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
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_absolute_errors = []
    all_portfolio_errors = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y.index), start=1):
        train_dates = pd.to_datetime(y.index[train_idx])
        test_dates = pd.to_datetime(y.index[test_idx])

        fold_abs_error = []
        fold_port_error = []

        if verbose:
            print(f"\n📦 Fold {fold}/{n_splits}")
            print(f"├─ Train range: {train_dates[0]} → {train_dates[-1]}  ({len(train_idx)} samples)")
            print(f"└─ Test  range: {test_dates[0]} → {test_dates[-1]}  ({len(test_idx)} samples)")

        y_train = y.loc[train_dates]
        y_test = y.loc[test_dates]

        X_train = X.loc[train_dates]
        X_test = X.loc[test_dates]

        if verbose:
            print("   ┌── Data shapes for first customer:")
            print(f"   │   X_train: {X_train.shape} | y_train: {y_train.shape}")
            print(f"   │   X_test : {X_test.shape} | y_test : {y_test.shape}")
            print("   └──────────────────────────────")

        if len(y_test) == 0:
            continue

        y_hat = predictor(X_train, y_train, X_test)

        country_err, portfolio_err = evaluate_forecast(y_test, y_hat)
        fold_abs_error.append(country_err)
        fold_port_error.append(portfolio_err)

        mean_fold_abs = np.mean(fold_abs_error)
        mean_fold_port = np.mean(fold_port_error)
        final_fold_score = 1.0 * mean_fold_abs + 5.0 * mean_fold_abs + 10.0 * mean_fold_port + 50.0 * mean_fold_port

        all_absolute_errors.append(mean_fold_abs)
        all_portfolio_errors.append(mean_fold_port)
        fold_scores.append(final_fold_score)

        if verbose:
            print(f"\n✅ Fold {fold} completed")
            print(f"   ├─ Mean Absolute Error (per customer):  {mean_fold_abs:.2f}")
            print(f"   ├─ Mean Portfolio Error (per customer): {mean_fold_port:.2f}")
            print(f"   └─ Final Fold Score: {np.round(final_fold_score, 2)}")

    # Final metrics
    mean_absolute_error = np.mean(all_absolute_errors)
    mean_portfolio_error = np.mean(all_portfolio_errors)
    final_score = np.mean(fold_scores)

    print("\n📊 Cross-Validation Summary")
    print(f"   ├─ Mean Absolute Error:  {mean_absolute_error:.2f}")
    print(f"   ├─ Mean Portfolio Error: {mean_portfolio_error:.2f}")
    print(f"   └─ Final CV Forecast Score: {np.round(final_score, 0)}")

    # print("FOLD_SCORES!!==========================")
    # print(fold_scores)

    # Create 1 row, 3 columns canvas
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Fold Scores
    axs[0].plot(range(1, len(fold_scores) + 1), fold_scores, marker='o', linestyle='-')
    axs[0].set_title('Fold Scores')
    axs[0].set_xlabel('Fold')
    axs[0].set_ylabel('Score')
    axs[0].grid(True)

    # Plot 2: Absolute Errors
    axs[1].plot(range(1, len(all_absolute_errors) + 1), all_absolute_errors, marker='o', linestyle='-')
    axs[1].set_title('Absolute Errors')
    axs[1].set_xlabel('Fold')
    axs[1].set_ylabel('Absolute Error')
    axs[1].grid(True)

    # Plot 3: Portfolio Errors
    axs[2].plot(range(1, len(all_portfolio_errors) + 1), all_portfolio_errors, marker='o', linestyle='-')
    axs[2].set_title('Portfolio Errors')
    axs[2].set_xlabel('Fold')
    axs[2].set_ylabel('Portfolio Error')
    axs[2].grid(True)

    plt.tight_layout()

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return mean_absolute_error, mean_portfolio_error, final_score


def evaluate(X: pd.DataFrame, y: pd.Series, save_path: str):
    # abs_err, port_err, score = cross_validate_forecaster(simple_predictor, training_set, features)
    abs_err, port_err, score = cross_validate_forecaster(
        predictor=elastic_net_predictor,
        X=X,
        y=y,
        verbose=True,
        save_path=save_path,  # or None to just show the plot
    )

    return abs_err, port_err, score


def autoregression():
    pass


def main(zone: str):
    """

    Train and evaluate the models for IT and ES

    """

    # Inputs
    input_path = r"datasets2025"
    output_path = r"outputs"

    # Load Datasets
    loader = DataLoader(input_path)
    # features are holidays and temperature
    training_set, features, example_results = loader.load_data(zone)

    """
    EVERYTHING STARTING FROM HERE CAN BE MODIFIED.
    """
    rollout, holidays = loader.load_additional_data(zone)
    # Add additional data to features

    team_name = "HANGUK_ML"
    # Data Manipulation and Training
    start_training = training_set.index.min()
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
    errors = pd.DataFrame(
        columns=["abs_err", "port_err", "score"],
    )

    models = dict()

    for costumer in training_set.columns.values:
        customer_id = int(costumer.split("_")[-1])
        print(f"******************************************")
        print(f"Start {customer_id}")

        df = dataset_encoding.generate_dataset(
            customer_id,
            window_size=24 * 7,
            forecast_skip=1,
            forecast_horizon=1,
            additional_feats=[
                "mean",
                "std",
                "skew",
                "kurtosis",
                "min",
                "max",
            ],
        )

        # evaluate
        X, y = dataset_encoding.get_train_data(df, customer_id, forecast_step=forecast_step, drop_nans_X=True)
        abs_err, port_err, score = evaluate(X, y, f"{output_path}/{customer_id}.png")
        errors.loc[customer_id] = [abs_err, port_err, score]
        print(f"errors: {errors.loc[customer_id]}")
        # consumption = training_set.loc[:, costumer]

        # feature_dummy = features["temp"].loc[start_training:]

        # encoding = SimpleEncoding(consumption, feature_dummy, end_training, start_forecast, end_forecast)

        # feature_past, feature_future, consumption_clean = encoding.meta_encoding()

        # # Train
        # model = SimpleModel()
        # model.train(feature_past, consumption_clean)

        # # Predict
        # output = model.predict(feature_future)
        # forecast[costumer] = output

    """
    END OF THE MODIFIABLE PART.
    """
    errors.to_csv(f"{output_path}/errors.csv")
    print(errors.mean())
    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - example_results).sum().sum()
    assert np.all(forecast.columns == example_results.columns), "Wrong header or header order."
    assert np.all(forecast.index == example_results.index), "Wrong index or index order."
    assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
    assert forecast.isna().sum().sum() == 0, "NaN in forecast."
    # Your solution will be evaluated using
    # forecast_error = np.abs(forecast - testing_set).sum().sum(),
    # and then doing a weighted sum the two portfolios:
    # score = forecast_error_IT + 5 * forecast_error_ES

    forecast.to_csv(join(output_path, "students_results_" + team_name + "_" + country + ".csv"))


if __name__ == "__main__":
    country = "IT"  # it can be ES or IT
    main(country)
