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



class naive2:
    """
    Enhanced naive model without any ML. It predicts future hourly power consumption 
    by combining multiple historical candidates to capture strong cyclicality:
    
      - Candidate 1: Value from the same time one year ago.
      - Candidate 2: Value from the same time one month ago.
      - Candidate 3: Value from the same time one week ago.
      - Candidate 4: Baseline average computed for each (day-of-week, hour) pair over the training period.
      
    The final prediction for each future timestep is the average of the candidates that are available.
    The forecast is returned as a pandas Series indexed with the next prediction_window hourly timestamps,
    starting at the next day 00:00 after the last training time step.
    """
    def __init__(self, prediction_window: int = 720):
        # Number of future hourly time steps to forecast.
        self.prediction_window = prediction_window
        # This will hold baseline averages keyed by (dayofweek, hour)
        self.baseline = None

    def train(self, x: pd.DataFrame or pd.Series):
        """
        Computes baseline averages from historical data.
        Expects x to be a DataFrame with a "consumption" column or a pd.Series.
        The baseline is the mean consumption for each combination of (day-of-week, hour).
        """
        # Allow for x to be a DataFrame with a 'consumption' column or a Series.
        if isinstance(x, pd.DataFrame):
            if "consumption" not in x.columns:
                raise ValueError("DataFrame must contain a 'consumption' column.")
            x = x["consumption"]
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Input time series must have a DatetimeIndex.")
            
        # Compute the baseline average for each (dayofweek, hour)
        grouped = x.groupby([x.index.dayofweek, x.index.hour]).mean()
        self.baseline = grouped.to_dict()
        # For debugging
        print("Baseline computed for (dayofweek, hour) pairs.")

    def predict(self, x: pd.DataFrame or pd.Series) -> pd.Series:
        # If x is a DataFrame, use its "consumption" column.
        if isinstance(x, pd.DataFrame):
            if "consumption" not in x.columns:
                raise ValueError("DataFrame must contain a 'consumption' column.")
            x = x["consumption"]
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Input time series must have a DatetimeIndex.")

        # Remove duplicate indices
        if x.index.duplicated().any():
            print("Duplicate indices found. Aggregating by mean over duplicate timestamps.")
            x = x.groupby(x.index).mean()

        # ... continue with prediction as before ...


        # Ensure baseline has been computed
        if self.baseline is None:
            raise ValueError("Please call train() before predict() to compute baseline averages.")

        # Determine the current (last) observation timestamp.
        current_observation = x.index[-1]
        # Determine the next day’s midnight after the last observation.
        prediction_start = (current_observation + pd.Timedelta(days=1)).normalize()
        # Construct the prediction range (hourly frequency).
        prediction_range = pd.date_range(start=prediction_start, periods=self.prediction_window, freq='H')

        # Define candidate ranges by shifting the prediction range:
        candidate_range_year = prediction_range - DateOffset(years=1)
        candidate_range_month = prediction_range - DateOffset(months=1)
        candidate_range_week = prediction_range - DateOffset(weeks=1)

        # Look up candidate values in x.
        candidate_year = x.reindex(candidate_range_year)
        candidate_month = x.reindex(candidate_range_month)
        candidate_week = x.reindex(candidate_range_week)

        # Candidate 4: Baseline average for each (day-of-week, hour)
        candidate_baseline = []
        for ts in prediction_range:
            key = (ts.dayofweek, ts.hour)
            val = self.baseline.get(key, np.nan)
            candidate_baseline.append(val)
        candidate_baseline = pd.Series(candidate_baseline, index=prediction_range)

        # Combine candidates into a DataFrame.
        df_candidates = pd.DataFrame({
            "year": candidate_year,
            "month": candidate_month,
            "week": candidate_week,
            "baseline": candidate_baseline
        }, index=prediction_range)

        # Compute row-wise average of available candidates.
        forecast = df_candidates.mean(axis=1, skipna=True)
        # Fill any remaining NaNs (if all candidates are missing) with 0.
        forecast = forecast.fillna(0)

        # Debug prints (optional)
        print("Current observation:", current_observation)
        print("Prediction range starts at:", prediction_range[0])
        print("Candidate (year ago) range starts at:", candidate_range_year[0])
        print("Candidate (month ago) range starts at:", candidate_range_month[0])
        print("Candidate (week ago) range starts at:", candidate_range_week[0])

        return forecast


