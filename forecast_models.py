import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import OneHotEncoder


class SimpleModel:
    """
    This is a simple example of a model structure

    """

    def __init__(self):
        self.linear_regression = LinearRegression()

    def train(self, x, y):
        self.linear_regression.fit(x, y)

    def fit(self, x, y):
        self.linear_regression.fit(x, y)

    def predict(self, x):
        return self.linear_regression.predict(x)


class LightGBMModel:
    """
    LightGBM model wrapper with customizable hyperparameters.
    """

    def __init__(self, **lgbm_params):
        """
        Initialize the LightGBM regressor with user-defined hyperparameters.

        Parameters:
        ----------
        lgbm_params : dict
            Keyword arguments to pass to LGBMRegressor (e.g., n_estimators, learning_rate, max_depth).
        """
        self.model = LGBMRegressor(**lgbm_params)

    def train(self, x, y):
        """
        Fit the model on training data.

        Parameters:
        ----------
        x : array-like
            Feature matrix.
        y : array-like
            Target values.
        """
        self.model.fit(x, y)

    def fit(self, x, y):
        """
        Fit the model on training data.

        Parameters:
        ----------
        x : array-like
            Feature matrix.
        y : array-like
            Target values.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained model.

        Parameters:
        ----------
        x : array-like
            Feature matrix for prediction.

        Returns:
        -------
        y_pred : array
            Predicted values.
        """
        return self.model.predict(x)

    def feature_importances(self, X: pd.DataFrame):
        """
        Print feature importances of the trained model.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix used for training.
        """
        print(sorted(list(zip(self.model.feature_importances_, X.columns)), reverse=True))
        return


class TimeOLSmodel:
    def __init__(self, prediction_window: int = 720):
        self.prediction_window = prediction_window
        self.model = LinearRegression()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _generate_time_features(self, index: pd.DatetimeIndex):
        """
        Generate one-hot encoded hour of day and day of week features from DatetimeIndex.
        """
        df = pd.DataFrame()
        df["hour"] = index.hour
        df["dayofweek"] = index.dayofweek
        return self.encoder.transform(df)

    def train(self, x: pd.DataFrame, y: np.ndarray):
        """
        x should have a datetime index and a 'consumption' column.
        """
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")

        df_time = pd.DataFrame()
        df_time["hour"] = x.index.hour
        df_time["dayofweek"] = x.index.dayofweek
        self.encoder.fit(df_time)

        time_features = self._generate_time_features(x.index)

        # Combine time features with past consumption
        features = np.concatenate([x["consumption"].values.reshape(-1, 1), time_features], axis=1)

        self.model.fit(features, y)

    def predict(self, x: pd.DataFrame):
        """
        x should have a datetime index and a 'consumption' column.
        """
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")

        time_features = self._generate_time_features(x.index)
        features = np.concatenate([x["consumption"].values.reshape(-1, 1), time_features], axis=1)

        return self.model.predict(features)


class TimeLGBMModel:
    def __init__(self, prediction_window: int = 720):
        self.prediction_window = prediction_window
        self.model = LGBMRegressor(verbosity=0)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _generate_time_features(self, index: pd.DatetimeIndex):
        """
        Generate one-hot encoded hour of day and day of week features from DatetimeIndex.
        """
        df = pd.DataFrame()
        df["hour"] = index.hour
        df["dayofweek"] = index.dayofweek
        return self.encoder.transform(df)

    def train(self, x: pd.DataFrame, y: np.ndarray):
        """
        x should have a datetime index and a 'consumption' column.
        """
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")

        df_time = pd.DataFrame()
        df_time["hour"] = x.index.hour
        df_time["dayofweek"] = x.index.dayofweek
        self.encoder.fit(df_time)

        time_features = self._generate_time_features(x.index)

        # Combine time features with past consumption
        features = np.concatenate([x["consumption"].values.reshape(-1, 1), time_features], axis=1)

        self.model.fit(features, y)

    def predict(self, x: pd.DataFrame):
        """
        x should have a datetime index and a 'consumption' column.
        """
        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex.")

        time_features = self._generate_time_features(x.index)
        features = np.concatenate([x["consumption"].values.reshape(-1, 1), time_features], axis=1)

        return self.model.predict(features)


# Initialize and run
def elastic_net_predictor(X_train, y_train, X_future):
    """Train ElasticNet model on X_train and y_train, return predictions on X_future."""
    # Initialize ElasticNet model with default parameters
    model = ElasticNet()

    # Train the model
    model.fit(X_train, y_train)

    return model.predict(X_future)


class ELasticNetModel:
    def __init__(self, **params):
        self.model = ElasticNet(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self, X: pd.DataFrame):
        print(sorted(list(zip(self.model.coef_, X.columns)), reverse=True))


def lgbm_predictor(X_train, y_train, X_future):
    model = LightGBMModel()

    model.train(X_train, y_train)

    return model.predict(X_future)


def ols_time_predictor(X_train, y_train, X_future):
    model = TimeOLSmodel()

    model.train(X_train, y_train)

    return model.predict(X_future)


def lgbm_time_predictor(X_train, y_train, X_future):
    model = TimeLGBMModel()

    model.train(X_train, y_train)

    return model.predict(X_future)
