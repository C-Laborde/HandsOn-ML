import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def decision_tree_regressor(data_prepared, labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(data_prepared, labels)
    predictions = tree_reg.predict(data_prepared)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return predictions, rmse
