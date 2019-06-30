import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def decision_tree_regressor(data_prepared, labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(data_prepared, labels)
    predictions = tree_reg.predict(data_prepared)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    scores = cross_val_score(tree_reg, data_prepared, labels,
                             scoring="neg_mean_squared_error", cv=10)
    return predictions, rmse, scores
