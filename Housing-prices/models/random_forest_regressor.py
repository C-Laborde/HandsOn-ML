import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def random_forest_regressor(data_prepared, labels):
    forest_reg = RandomForestRegressor()
    forest_reg.fit(data_prepared, labels)
    predictions = forest_reg.predict(data_prepared)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    scores = cross_val_score(forest_reg, data_prepared, labels,
                             scoring="neg_mean_squared_error", cv=10)
    return predictions, rmse, scores
