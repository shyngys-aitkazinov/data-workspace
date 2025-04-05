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
        # 2) Generate time-based features over the same period
        # ------------------------------------------------------
        if include_time_features:
            time_features = self.generate_time_series_features(start_time, end_time)

            # We'll merge them on the datetime index
            df = pd.DataFrame({"consumption": y})
            df = df.join(time_features, how="inner")  # or outer if needed

        # ------------------------------------------------------
        # 3) Create lag features (window_size hours of history)
        #    lag_1 => consumption at t-1
        #    lag_2 => consumption at t-2
        #    ...
        #    lag_{window_size} => consumption at t-window_size
        # ------------------------------------------------------
        lag_cols = {}
        for lag in range(1, window_size + 1):
            lag_cols[f"{customer_id}_lag_{lag}"] = df["consumption"].shift(lag)

        # Convert dict to a DataFrame and concat once
        lag_df = pd.DataFrame(lag_cols, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        # ------------------------------------------------------
        # 4) Create future consumption columns (our targets)
        #    - forecast_skip: how many hours to skip
        #    - forecast_horizon: how many steps we want to forecast
        #
        #    Example:
        #       If forecast_skip=1 and horizon=24, then for each time t,
        #       the targets are consumption at t+1 ... t+24
        # ------------------------------------------------------
        # Start of the forecast period is (t + forecast_skip)
        # End of the forecast period is (t + forecast_skip + i - 1)
        future_cols = {}
        for i in range(1, forecast_horizon + 1):
            target_col = f"{customer_id}_future_{i}"
            future_cols[target_col] = df["consumption"].shift(-1 * (forecast_skip + i - 1))

        future_df = pd.DataFrame(future_cols, index=df.index)
        df = pd.concat([df, future_df], axis=1)
        # ------------------------------------------------------
        # 5) Drop rows where we don't have enough history or future
        #    - We lose the first 'window_size' rows to lag features
        #    - We lose the last 'forecast_skip + forecast_horizon - 1' rows
        #      to future features
        # ------------------------------------------------------

        future_feats_df_list = []

        # For each column in self.features, shift backward
        for col in self.features.columns:
            for i in range(1, forecast_horizon + 1):
                shift_amount = -1 * (forecast_skip + i - 1)
                future_feat_col = f"f{i}_{col}"
                future_feats_df_list.append(
                    pd.DataFrame({future_feat_col: self.features[col].shift(shift_amount)}, index=self.features.index)
                )

        for i in range(1, forecast_horizon + 1):
            shift_amount = -1 * (forecast_skip + i - 1)
            future_feat_col = f"f{i}_{self.rollout_id_map[customer_id]}"
            future_feats_df_list.append(
                pd.DataFrame(
                    {future_feat_col: self.rollout[self.rollout_id_map[customer_id]].shift(shift_amount)},
                    index=self.rollout.index,
                )
            )

        # Combine all the future feature DataFrames
        if future_feats_df_list:
            future_feats_big = pd.concat(future_feats_df_list, axis=1)
            # Now slice future_feats_big to the same [start_time, end_time] index
            future_feats_big = future_feats_big.loc[start_time:end_time]
            # Merge them into df
            df = df.join(future_feats_big, how="left")

        earliest_valid_row = window_size  # because lag_n needs that many back steps
        latest_valid_row = len(df) - (forecast_skip + forecast_horizon - 1)
        df = df.iloc[earliest_valid_row:latest_valid_row]

        add_dict = {}
        for add_feat in additional_feats:
            if add_feat == "mean":
                add_dict["rolling_mean"] = df["consumption"].rolling(window=window_size).mean()
            elif add_feat == "std":
                add_dict["rolling_std"] = df["consumption"].rolling(window=window_size).std()
            elif add_feat == "kurtosis":
                add_dict["rolling_kurtosis"] = df["consumption"].rolling(window=window_size).kurt()
            elif add_feat == "skew":
                add_dict["rolling_skew"] = df["consumption"].rolling(window=window_size).skew()
            elif add_feat == "min":
                add_dict["rolling_min"] = df["consumption"].rolling(window=window_size).min()
            elif add_feat == "max":
                add_dict["rolling_max"] = df["consumption"].rolling(window=window_size).max()
            else:
                # Possibly it's a direct column name in self.features (or some external field)
                # If it's in df already, do nothing. If it's in self.features, we can explicitly join here.
                # For simplicity, we assume it's already joined above if it's in self.features or rollout.
                # If you want to handle it differently, you can do so here.
                if add_feat not in df.columns:
                    print(f"Warning: '{add_feat}' not recognized as a stat or existing column.")
                    # Optionally do: add_dict[add_feat] = self.features[add_feat].loc[start_time:end_time]

        # Concat all newly created additional features at once
        if add_dict:
            add_stats_df = pd.DataFrame(add_dict, index=df.index)
            df = pd.concat([df, add_stats_df], axis=1)

        # ------------------------------------------------------
        # 6) Return the final DataFrame
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

        return X, y
