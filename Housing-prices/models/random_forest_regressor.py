import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]


def random_forest_regressor(data_prepared, labels, save=False):
    forest_reg = RandomForestRegressor()
    forest_reg.fit(data_prepared, labels)
    predictions = forest_reg.predict(data_prepared)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    scores = cross_val_score(forest_reg, data_prepared, labels,
                             scoring="neg_mean_squared_error", cv=10)
    if save:
        # save model hyperparams, trained params, scores and predictors
        path = "model_results/"
        joblib.dump(forest_reg, path + "forest_reg.pkl")
    return predictions, rmse, scores


def random_forest_grid_search(data_prepared, labels):
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(data_prepared, labels)
    return grid_search
