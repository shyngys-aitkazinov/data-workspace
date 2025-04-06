from os.path import join

import numpy as np
import pandas as pd
import pytz


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")
        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        consumptions = pd.read_csv(consumptions_path, index_col=0, parse_dates=True, date_format=date_format)
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
        rollout = pd.read_csv(rollout_path, index_col=0, parse_dates=True, date_format=date_format)

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
        print(self.features.shape)
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        print(f"features_past shape: {features_past.shape}")
        features_future = self.features[self.start_forecast : self.end_forecast].values.reshape(-1, 1)
        print(f"features_past shape: {features_past.shape}, features_future shape: {features_future.shape}")

        features_past = features_past[self.consumption_mask]
        print(f"features_past shape after masking: {self.consumption_mask.shape}")

        print(features_past.shape)
        return features_past, features_future, self.consumption


class DatasetEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.DataFrame,
        features: pd.DataFrame,
        rollout: pd.DataFrame,
        holiday: pd.DataFrame,
        start_training: pd.Timestamp,
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
        self.start_training = start_training

        self.customer_id_map = {int(col.split("_")[-1]): col for col in consumption.columns}
        self.rollout_id_map = {int(col.split("_")[-1]): col for col in rollout.columns}

    def generate_time_series_features(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """
        Generate time series features at hourly granularity between 'start_time' and 'end_time'.
        Encodes:
            - Hour (sin, cos)
            - Day of the week (sin, cos)
            - Month (sin, cos)
            - Weekend (binary)
            - Holiday (binary)
            - Summer time / Winter time (binary)
        """

        # 1. Create an hourly date range (naive UTC, then convert to local) or directly local:
        dt_index = pd.date_range(start=start_time, end=end_time, freq="1h", tz="Europe/Berlin")

        # 2. Create a DataFrame with that index
        df = pd.DataFrame(index=dt_index)

        # 3. Convert to a timezone that observes DST
        #    e.g., Europe/Berlin. Change as needed for your region.

        # 4. is_summer_time feature
        #    x.dst() will be non-zero (e.g. 1:00:00) in summer time.
        df["is_summer_time"] = dt_index.map(lambda x: int(bool(x.dst())))

        # ---------------------------------------
        # Hour of day (0-23)
        df["hour"] = dt_index.hour
        # Day of week: Monday=0, Sunday=6
        df["day_of_week"] = dt_index.dayofweek
        # Month (1–12)
        df["month"] = dt_index.month

        for h in range(23):
            df[f"hour_{h}"] = (df["hour"] == h).astype(int)

        # Manual one-hot for day of week (0–6)
        for d in range(6):
            df[f"dow_{d}"] = (df["day_of_week"] == d).astype(int)

        # Hour as sin/cos
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week as sin/cos
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Month as sin/cos
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # Weekend (binary)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Example for 'holiday' detection
        # --------------------------------
        holiday_dates = self.holiday.index.normalize()  # e.g., 2025-01-01, etc.
        df["date_only"] = dt_index.normalize()
        df["is_holiday"] = df["date_only"].isin(holiday_dates).astype(int)

        # Drop helper columns for cleanliness
        df.drop(["hour", "day_of_week", "month", "date_only"], axis=1, inplace=True)
        df.index = df.index.tz_localize(None)

        return df

    def generate_dataset(
        self,
        customer_id: int,
        window_size: int = 24 * 7,
        forecast_skip: int = 1,
        forecast_horizon: int = 24,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
        include_time_features: bool = True,
        additional_feats: list[str] = [],
    ) -> pd.DataFrame:
        """
        Generate a supervised dataset for a single customer.

        Args:
            customer_id (int): Numeric ID to select which customer to prepare data for.
            window_size (int): Number of past hours used as features (lag features).
            forecast_skip (int): How many hours to skip before the forecast horizon starts.
            forecast_horizon (int): Number of hours in the forecast horizon.
            start_time (pd.Timestamp, optional): Start time for data slicing.
                                                 Defaults to entire range if None.
            end_time (pd.Timestamp, optional): End time for data slicing.
                                               Defaults to entire range if None.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a "time step" and contains:
                          - Time-based features
                          - Past consumption lags (window_size columns)
                          - Future consumption values (forecast_horizon columns)
        """

        # ------------------------------------------------------
        # 1) Identify the target series for the chosen customer
        # ------------------------------------------------------
        cust_col_name = self.customer_id_map[customer_id]
        y = self.consumption[cust_col_name].copy()

        # If no start/end time provided, fallback to entire range for that customer
        if start_time is None:
            start_time = y.index.min()
        if end_time is None:
            end_time = y.index.max()

        # Slice the consumption data
        y = y.loc[start_time:end_time].sort_index()

        # ------------------------------------------------------
        # 2) Generate time-based features over the same period (if requested)
        # ------------------------------------------------------
        if include_time_features:
            time_features = self.generate_time_series_features(start_time, end_time)
            # We'll merge them on the datetime index
            df = pd.DataFrame({"consumption": y})
            df = df.join(time_features, how="right")
        else:
            df = pd.DataFrame({"consumption": y})

        # ------------------------------------------------------
        # 3) Create lag features (window_size hours of history)
        # ------------------------------------------------------
        lag_cols = {}
        for lag in range(1, window_size + 1):
            lag_cols[f"{customer_id}_lag_{lag}"] = df["consumption"].shift(lag)
        lag_df = pd.DataFrame(lag_cols, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        # ------------------------------------------------------
        # 4) Create future consumption columns (our targets)
        # ------------------------------------------------------
        future_cols = {}
        for i in range(1, forecast_horizon + 1):
            target_col = f"{customer_id}_future_{i}"
            future_cols[target_col] = df["consumption"].shift(-1 * (forecast_skip + i - 1))
        future_df = pd.DataFrame(future_cols, index=df.index)
        df = pd.concat([df, future_df], axis=1)

        # ------------------------------------------------------
        # 5) Create future features from self.features and self.rollout
        # ------------------------------------------------------
        future_feats_df_list = []

        # For each column in self.features, shift backward for each forecast step
        for col in self.features.columns:
            for i in range(1, forecast_horizon + 1):
                shift_amount = -1 * (forecast_skip + i - 1)
                future_feat_col = f"f{i}_{col}"
                future_feats_df_list.append(
                    pd.DataFrame({future_feat_col: self.features[col].shift(shift_amount)}, index=self.features.index)
                )

        # For rollout features (using a mapping: self.rollout_id_map)
        for i in range(1, forecast_horizon + 1):
            shift_amount = -1 * (forecast_skip + i - 1)
            future_feat_col = f"f{i}_{self.rollout_id_map[customer_id]}"
            future_feats_df_list.append(
                pd.DataFrame(
                    {future_feat_col: self.rollout[self.rollout_id_map[customer_id]].shift(shift_amount)},
                    index=self.rollout.index,
                )
            )

        if future_feats_df_list:
            future_feats_big = pd.concat(future_feats_df_list, axis=1)
            # Restrict to our time range
            future_feats_big = future_feats_big.loc[start_time:end_time]
            df = df.join(future_feats_big, how="left")

        # ------------------------------------------------------
        # 6) Add additional rolling statistics if requested
        # ------------------------------------------------------
        add_dict = {}
        for add_feat in additional_feats:
            if add_feat == "mean":
                # Require full window for a valid mean, else NaN.
                add_dict["rolling_mean"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).mean()
                )
            elif add_feat == "std":
                add_dict["rolling_std"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).std()
                )
            elif add_feat == "kurtosis":
                add_dict["rolling_kurtosis"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).kurt()
                )
            elif add_feat == "skew":
                add_dict["rolling_skew"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).skew()
                )
            elif add_feat == "min":
                add_dict["rolling_min"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).min()
                )
            elif add_feat == "max":
                add_dict["rolling_max"] = (
                    df["consumption"].rolling(window=window_size, min_periods=window_size // 2).max()
                )
            # ---- EXAMPLES: Custom daily/weekly lags ----
            elif add_feat == "lag_24":
                add_dict["lag_24"] = df["consumption"].shift(24)
            elif add_feat == "lag_48":
                add_dict["lag_48"] = df["consumption"].shift(48)
            elif add_feat == "lag_72":
                add_dict["lag_72"] = df["consumption"].shift(72)
            elif add_feat == "lag_168":
                add_dict["lag_168"] = df["consumption"].shift(168)
            # ---- EXAMPLE: Simple FFT-based feature on last 'window_size' points ----
            elif add_feat == "fft":
                # We'll extract the largest frequency amplitude in the window
                # This is a simple demonstration – real use might require more nuanced approach
                def fft_max_amp(series):
                    clean = series.dropna().values
                    if len(clean) < 2:
                        return 0.0
                    freqs = np.fft.fft(clean)
                    magnitudes = np.abs(freqs)
                    # Skip DC component if you like (magnitudes[1:]) or keep it. We'll skip it here:
                    return magnitudes[1:].max() if len(magnitudes) > 1 else 0.0

                def fft_dominant_phase(x):
                    try:
                        # Ensure input is a NumPy array
                        x = np.asarray(x)
                        if x.size == 0:
                            return np.nan  # Return NaN for empty inputs

                        # Compute the FFT of the input
                        fft_result = np.fft.fft(x)
                        # Compute amplitudes of the FFT result
                        amplitude = np.abs(fft_result)

                        # If the amplitudes are nearly all zero, return NaN to avoid spurious results
                        if np.allclose(amplitude, 0):
                            return np.nan

                        # If there's more than one element, ignore the DC component (first element)
                        if amplitude.size > 1:
                            dominant_index = np.argmax(amplitude[1:]) + 1
                        else:
                            dominant_index = 0

                        # Extract and return the phase at the dominant frequency index
                        phase = np.angle(fft_result[dominant_index])
                        return phase

                    except Exception:
                        # Return NaN if any computational issues arise
                        return np.nan

                # 1) Using the main window_size
                add_dict["fft_max_amp"] = df["consumption"].rolling(window=window_size).apply(fft_max_amp, raw=False)
                add_dict["fft_dom_phase"] = (
                    df["consumption"].rolling(window=window_size).apply(fft_dominant_phase, raw=False)
                )

                # 2) Fixed windows: 1 day, 1 week, 1 month (~30d), 1 year (~365d)
                windows_map = {
                    "1d": 24,
                    "1w": 168,
                    "1m": 720,  # approx 30 days
                    "1y": 8760,  # approx 365 days
                }
                for label, wsize in windows_map.items():
                    add_dict[f"fft_{label}_max_amp"] = df["consumption"].rolling(wsize).apply(fft_max_amp, raw=False)
                    add_dict[f"fft_{label}_phase"] = (
                        df["consumption"].rolling(wsize).apply(fft_dominant_phase, raw=False)
                    )
            else:
                if add_feat not in df.columns:
                    print(f"Warning: '{add_feat}' not recognized as a stat or existing column.")

        if add_dict:
            add_stats_df = pd.DataFrame(add_dict, index=df.index)
            df = pd.concat([df, add_stats_df], axis=1)

        # ------------------------------------------------------
        # 7) Return the final DataFrame covering the full [start_time, end_time]
        # ------------------------------------------------------
        return df

    def get_train_data(
        self,
        df: pd.DataFrame,
        customer_id: int,
        forecast_step: int,
        interpolate_limit: int = 1,
        drop_nans_X: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        target_column = f"{customer_id}_future_{forecast_step}"

        # Perform linear interpolation on the "target" column
        df[target_column] = df[target_column].interpolate(
            method="time", limit=interpolate_limit, limit_direction="both", limit_area="inside"
        )

        if drop_nans_X:
            futures = [col for col in df.columns if col.startswith(f"{customer_id}_future_") and col != target_column]
            df = df.drop(columns=futures).dropna(axis=0)
            y = df[target_column]
            X = df.drop(columns=target_column)
        else:
            mask = df[target_column].notna()
            y = df[target_column][mask]

            to_drop = [col for col in df.columns if col.startswith(f"{customer_id}_future_")]
            X = df[mask].drop(columns=to_drop)

        X.drop(
            columns=[
                "consumption",
                f"{customer_id}_lag_1",
                f"{customer_id}_lag_2",
            ],
            inplace=True,
            errors="ignore",
        )
        return X, y

    def get_test_sample(
        self,
        df: pd.DataFrame,
        ts: pd.Timestamp,
        customer_id: int,
        forecast_step: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        target_column = f"{customer_id}_future_{forecast_step}"
        to_drop = [col for col in df.columns if col.startswith(f"{customer_id}_future_")]
        X = df.drop(columns=to_drop + ["consumption", f"{customer_id}_lag_1", f"{customer_id}_lag_2"]).loc[[ts]]
        y = df[target_column].loc[ts]
        return X, y
