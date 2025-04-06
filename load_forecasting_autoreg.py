from collections.abc import Callable
from os.path import join
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import TimeSeriesSplit

# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, DatasetEncoding, SimpleEncoding
# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import ELasticNetModel, LightGBMModel, SimpleModel, elastic_net_predictor

FORECAST_STEP = 1


class AutoRegressor:
    def __init__(
        self,
        dataset_encoding: DatasetEncoding,
        model,
        model_path: str | None = None,
        **kwargs,
    ):
        self.model = model
        self._requires_train = True
        if model_path is not None:
            self.model = np.load(model_path)
            self._requires_train = False
        self.end_training = dataset_encoding.end_training
        self.start_forecast = dataset_encoding.start_forecast
        self.end_forecast = dataset_encoding.end_forecast
        self.start_training = dataset_encoding.start_training

        self.dataset_encoding = dataset_encoding
        self.kwargs = kwargs
        self.window_size = kwargs.get("window_size", 24 * 7)
        self.additional_feats = kwargs.get("additional_feats", [])

    def setup_df(self, customer_id: int):
        """
        Generate the dataset for the given customer ID.
        This method is called during initialization and when updating features.
        """
        self.df = self.dataset_encoding.generate_dataset(
            customer_id,
            start_time=self.start_training,
            end_time=self.end_training,
            **self.kwargs,
        )

    def train(self, customer_id: int, forecast_step=FORECAST_STEP):
        X, y = self.dataset_encoding.get_train_data(
            self.df.loc[self.start_training : self.end_training],
            customer_id,
            forecast_step,
        )
        self.model.fit(X, y)

    def predict(self, customer_id: int) -> pd.Series:
        if self._requires_train:
            self.train(customer_id)
        return self.predict_autoregressive(customer_id)

    def update_features(
        self,
        ts: pd.Timestamp,
        customer_id: int,
    ) -> pd.DataFrame:
        """
        Update the lag features of the current feature vector.
        Assumes that lag features are named as f"{customer_id}_lag_{i}" for i=1,...,window_size.

        The update shifts existing lag values one step back and inserts y_pred as the new lag_1.
        """

        timeseries = self.df.loc[
            ts - pd.Timedelta(hours=self.window_size - 1) : ts + pd.Timedelta(hours=1), "consumption"
        ]

        # For lag_i (i from 2 to window_size), set new lag value equal to the old lag_{i-1} from time ts.
        for lag in range(1, self.window_size + 1):
            self.df.loc[ts, f"{customer_id}_lag_{lag}"] = self.df.loc[
                ts - pd.Timedelta(hours=lag), f"{customer_id}_lag_{lag}"
            ]

        for add_feat in self.additional_feats:
            if add_feat == "mean":
                # Require full window for a valid mean, else NaN.
                self.df.loc[ts, "rolling_mean"] = np.mean(timeseries)
            elif add_feat == "std":
                # Require full window for a valid std, else NaN.
                self.df.loc[ts, "rolling_std"] = np.std(timeseries)
            elif add_feat == "kurtosis":
                self.df.loc[ts, "rolling_kurtosis"] = timeseries.kurt()
            elif add_feat == "skew":
                # Require full window for a valid skew, else NaN.
                self.df.loc[ts, "rolling_skew"] = timeseries.skew()
            elif add_feat == "min":
                # Require full window for a valid min, else NaN.
                self.df.loc[ts, "rolling_min"] = np.min(timeseries)
            elif add_feat == "max":
                self.df.loc[ts, "rolling_max"] = np.max(timeseries)
            else:
                if add_feat not in self.df.columns:
                    print(f"Warning: '{add_feat}' not recognized as a stat or existing column.")

    def predict_autoregressive(self, customer_id: int) -> pd.Series:
        """
        Perform autoregressive forecasting over forecast_horizon steps.

        1. Generate an initial feature vector (using dataset_encoding) at the forecast start.
        2. Iteratively predict one time step ahead.
        3. Update the lag features in the feature vector with the new prediction.

        Args:
            customer_id (int): The customer ID to forecast.
            forecast_horizon (int): How many time steps to forecast.
            window_size (int): The number of lag features used by the model.
            kwargs: Additional parameters for generating the dataset.

        Returns:
            list: A list of forecasted values.
        """
        # Generate initial forecast features.
        # Here we assume generate_dataset returns a DataFrame covering the forecast period.

        for ts in pd.date_range(self.start_forecast, self.end_forecast, freq="1H"):
            # Predict one time step ahead.
            # Suppose ts is a pd.Timestamp
            ts_before = ts - pd.Timedelta(hours=1)
            X, _ = self.dataset_encoding.get_test_sample(self.df, ts_before, customer_id, forecast_step=FORECAST_STEP)
            y_pred = self.model.predict(X)[0]

            self.df.loc[ts, "consumption"] = y_pred
            # Update the feature vector with the new prediction.
            self.update_features(ts, customer_id)

        return self.df.loc[self.start_forecast : self.end_forecast, "consumption"]


def plot(gt: pd.Series, pred: pd.Series, save_path: str | None = None):
    # Create a wide figure (e.g., 30 inches wide by 10 inches tall)
    plt.figure(figsize=(30, 10), dpi=100)

    index = gt.index

    # Plot the two series with different colors and labels
    plt.plot(index, gt, label="Ground Truth", color="blue", linewidth=2)
    plt.plot(index, pred, label="Prediction", color="red", linewidth=2)

    # Add title and labels
    plt.title("Time Series Plot of Two Lines (720 Data Points)")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Enable grid and legend
    plt.grid(True)
    plt.legend()

    # Improve layout and export to PNG with high resolution
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def evaluate_forecast(y_true, y_pred):
    diff = y_pred - y_true
    country_error = diff.abs().sum()
    portfolio_country_error = diff.sum()
    return country_error, abs(portfolio_country_error)


# def evaluate(X: pd.DataFrame, y: pd.Series, save_path: str, my_predictor):
#     """
#     Runs cross-validation with a given predictor.
#     Returns:
#       abs_err, port_err, score, abs_err_std, port_err_std, score_std
#     """
#     results = cross_validate_forecaster(
#         predictor=my_predictor,
#         X=X,
#         y=y,
#         verbose=True,
#         save_path=save_path,  # not currently used to save anything, but left for clarity
#     )

#     (abs_err, port_err, score, abs_err_std, port_err_std, score_std) = results

#     return abs_err, port_err, score, abs_err_std, port_err_std, score_std


def main(zone: str):
    """

    Train and evaluate the models for IT and ES

    """

    # Inputs
    input_path = r"datasets2025"
    output_path = r"outputs"
    # models_path = r"models_path"
    models_path = None

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
    # start_training = training_set.index.min()
    # end_training = training_set.index.max()
    # start_forecast, end_forecast = example_results.index[0], example_results.index[-1]

    data_format = "%Y-%m-%d %H:%M:%S"
    start_training = training_set.index.min()

    end_training = pd.to_datetime("2024-06-30 23:00:00", format=data_format)
    start_forecast = pd.to_datetime("2024-06-30 23:00:00", format=data_format)
    end_forecast = pd.to_datetime("2024-07-31 23:00:00", format=data_format)

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

    kwargs = dict(
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
    range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1h")
    forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)

    ar = AutoRegressor(
        dataset_encoding=dataset_encoding,
        model=LightGBMModel(),
        model_path=None,
        **kwargs,
    )

    for costumer in training_set.columns.values:
        customer_id = int(costumer.split("_")[-1])
        print(f"******************************************")
        print(f"Start {customer_id}")
        ar.setup_df(customer_id)
        forecast[costumer] = ar.predict(customer_id)
        plot(
            training_set.loc[range_forecast, costumer],
            forecast[costumer],
            save_path=join(output_path, f"{team_name}_{zone}_{customer_id}.png"),
        )
        break

    """
    END OF THE MODIFIABLE PART.
    """
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
