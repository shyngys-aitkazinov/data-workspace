import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


class naive:
    """
    Naive model without any ML. Predicts future values by averaging
    corresponding values from one year ago and one month ago.
    
    The prediction range now starts one hour after the last observation,
    rather than snapping to the next day’s midnight.
    """
    def __init__(self, prediction_window: int = 720):
        # prediction_window is the number of future hourly time steps to forecast.
        self.prediction_window = prediction_window

    def train(self, **kwargs):
        # No training is needed for a naive predictor.
        pass

    def predict(self, x: pd.Series) -> pd.Series:
        """
        Predict the next prediction_window timesteps immediately after
        the last timestamp in x.
        
        The procedure is:
          - Define a prediction range: hourly timestamps starting from one hour
            after the last observation.
          - Candidate one: lookup the corresponding period one year ago.
          - Candidate two: lookup the corresponding period one month ago.
          - For each timestep in the prediction range, if both candidate values are available,
            take their average; if only one is available, use that value; if both are missing,
            return 0.
        
        Parameters:
          x (pd.Series): Time series with a DatetimeIndex and a 'consumption' column.
        
        Returns:
          pd.Series: Forecasted values for the next prediction_window hours, with the future timestamps.
        """
        # Ensure x has a DatetimeIndex and the 'consumption' series is available.
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Input time series must have a DatetimeIndex.")

        # If x is a DataFrame, extract the "consumption" column.
        if isinstance(x, pd.DataFrame):
            x = x['consumption']

        # Determine the current observation (last timestamp)
        current_observation = x.index[-1]
        # Set prediction_start to one hour after the last observation
        prediction_start = current_observation + pd.Timedelta(hours=1)
        # Create the prediction range with hourly frequency
        prediction_range = pd.date_range(start=prediction_start, periods=self.prediction_window, freq='H')

        # Define candidate ranges by shifting the prediction range.
        candidate_range_year = prediction_range - DateOffset(years=1)
        candidate_range_month = prediction_range - DateOffset(months=1)

        # Look up candidate values in x.
        candidate_one = x.reindex(candidate_range_year)
        candidate_one.index = prediction_range  # Align with prediction_range

        candidate_two = x.reindex(candidate_range_month)
        candidate_two.index = prediction_range  # Align with prediction_range
        
        # Combine candidates into a DataFrame.
        df_candidates = pd.DataFrame({
            "candidate_one": candidate_one,
            "candidate_two": candidate_two
        }, index=prediction_range)

        # Compute the row-wise average of available candidate values.
        forecast = df_candidates.mean(axis=1, skipna=True)
        forecast = forecast.fillna(0)

        # Debug prints (optional)
        print("Current observation:", current_observation)
        print("Prediction range starts at:", prediction_range[0])
        print("Candidate range (1 year ago) starts at:", candidate_range_year[0])
        print("Candidate range (1 month ago) starts at:", candidate_range_month[0])

        return forecast



