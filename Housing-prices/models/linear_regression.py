import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression(data, prepared_data, labels, pipeline):
    lin_reg = LinearRegression()
    # we fit the model to processed data
    lin_reg.fit(prepared_data, labels)

    # we test the model on raw data
    data_prepared = pipeline.fit_transform(data)
    predictions = lin_reg.predict(data_prepared)
    mse = np.sqrt(mean_squared_error(labels, predictions))
    return predictions, mse
