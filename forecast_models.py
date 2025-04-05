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


# Initialize and run
def elastic_net_predictor(X_train, y_train, X_future):
    """Train ElasticNet model on X_train and y_train, return predictions on X_future."""
    # Initialize ElasticNet model with default parameters
    model = ElasticNet()

    # Train the model
    model.fit(X_train, y_train)

    return model.predict(X_future)
