from sklearn.linear_model import ElasticNet, LinearRegression


class SimpleModel:
    """
    This is a simple example of a model structure

    """

    def __init__(self):
        self.linear_regression = LinearRegression()

    def train(self, x, y):
        self.linear_regression.fit(x, y)

    def predict(self, x):
        return self.linear_regression.predict(x)


from lightgbm import LGBMRegressor


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
# Initialize and run
def elastic_net_predictor(X_train, y_train, X_future):
    """Train ElasticNet model on X_train and y_train, return predictions on X_future."""
    # Initialize ElasticNet model with default parameters
    model = ElasticNet()

    # Train the model
    model.fit(X_train, y_train)

    return model.predict(X_future)
