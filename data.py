import pandas as pd
from os.path import join


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")
        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=date_format
        )
        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        return consumptions, features, example_solution

    def load_additional_data(self, country: str):
        """Load additional data for the given country.
        Args:
            country (str): The country for which to load additional data.
        Returns:
            tuple: A tuple containing the rollout data and holidays data.
        """
        
        # Define file paths
        date_format = "%Y-%m-%d %H:%M:%S"
        rollout_path = join(self.path, "rollout_data_" + country + ".csv")
        holidays_path = join(self.path, "holiday_" + country + ".xlsx")
        
        # Load datasets
        rollout = pd.read_csv(
            rollout_path, index_col=0, parse_dates=True, date_format=date_format
        )
        
        holidays = pd.read_excel(
            holidays_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        
        return rollout, holidays
        
        
    
    def load_example_solution(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        # Load datasets
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        return example_solution


# Encoding Part


class SimpleEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        end_training,
        start_forecast,
        end_forecast,
    ):
        self.consumption_mask = ~consumption.isna()
        self.consumption = consumption[self.consumption_mask]
        self.features = features
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast

    def meta_encoding(self):
        """
        This function returns the feature, split between past (for training) and future (for forecasting)),
        as well as the consumption, without missing values.
        :return: three numpy arrays

        """
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values.reshape(-1, 1)

        features_past = features_past[self.consumption_mask]

        return features_past, features_future, self.consumption


class DatasetEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        rollout: pd.Series,
        holiday: pd.Series,
        end_training: pd.Timestamp,
        start_forecast: pd.Timestamp,
        end_forecast: pd.Timestamp,
    ):
        """Initialize the DatasetEncoding class with time series data and forecast parameters.

        Args:
            consumption (pd.Series): Historical consumption/demand data
            features (pd.Series): Feature variables for prediction
            rollout (pd.Series): Rollout data for the specified region
            holiday (pd.Series): Holiday calendar data
            end_training (pd.Timestamp): End date for the training period
            start_forecast (pd.Timestamp): Start date for the forecast period
            end_forecast (pd.Timestamp): End date for the forecast period

        Note:
            All time series data should have datetime index and be properly aligned
        """
        self.features = features
        self.consumption = consumption
        self.rollout = rollout
        self.holiday = holiday
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast
