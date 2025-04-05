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


class AutoRegressor:
    def __init__(
        self,
        model,
        dataset_encoding: DatasetEncoding,
    ):
        self.model = model
        self.end_training = dataset_encoding.end_training
        self.start_forecast = dataset_encoding.start_forecast
        self.end_forecast = dataset_encoding.end_forecast
        self.start_training = dataset_encoding.start_training

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


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
        final_fold_score = 1.0 * mean_fold_abs + 5.0 * mean_fold_abs + 10.0 * mean_fold_port + 50.0 * mean_fold_port
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

    (abs_err, port_err, score, abs_err_std, port_err_std, score_std) = results

    return abs_err, port_err, score, abs_err_std, port_err_std, score_std


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
        start_training=start_training,
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
