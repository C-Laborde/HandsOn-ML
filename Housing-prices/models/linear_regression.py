import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def linear_regression(data_prepared, labels):
    lin_reg = LinearRegression()
    # we fit the model to processed data
    lin_reg.fit(data_prepared, labels)
    predictions = lin_reg.predict(data_prepared)
    mse = np.sqrt(mean_squared_error(labels, predictions))
    scores = cross_val_score(lin_reg, data_prepared, labels,
                             scoring="neg_mean_squared_error", cv=10)
    return predictions, mse, scores
