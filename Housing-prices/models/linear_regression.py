import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression(data_prepared, labels):
    lin_reg = LinearRegression()
    # we fit the model to processed data
    lin_reg.fit(data_prepared, labels)
    predictions = lin_reg.predict(data_prepared)
    mse = np.sqrt(mean_squared_error(labels, predictions))
    return predictions, mse
