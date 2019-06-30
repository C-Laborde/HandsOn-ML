import os
import numpy as np
from utils.load_data import load_data
# from describe_data import describe_data
# from utils.test_data import split_train_test_by_id
# from sklearn.model_selection import train_test_split
from utils.test_data import stratified_shuffle
# from utils.explore_data import explore_data
# from utils.prepare_data import impute_na, encode_cat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from utils.combined_attributes_adder import CombinedAttributesAdder
from sklearn.compose import ColumnTransformer
# from models.linear_regression import linear_regression
# from models.decision_tree_regressor import decision_tree_regressor
from models.random_forest_regressor import random_forest_regressor
from models.random_forest_regressor import random_forest_grid_search
from sklearn.metrics import mean_squared_error
from scipy import stats


HOUSING_PATH = os.path.join("datasets", "housing")

if __name__ == "__main__":
    # ### LOAD AND DESCRIBE DATA ####
    df = load_data(HOUSING_PATH)
    # describe_data(df)

    # ### SPLIT DATA ####
    # 1) in-house test train split function:
    # df = df.reset_index()
    # df["id"] = df["longitude"] * 1000 + df["latitude"]
    # train_set, test_set = split_train_test_by_id(df, 0.2, "id")

    # 2) sklearn function for random splitting
    # train_set, test_set = train_test_split(df, test_size=0.2,
    #                                        random_state=42)

    # 3) stratify data and split accordingly
    strat_train_set, strat_test_set = stratified_shuffle(df)

    # ### EXPLORE DATA ####
    # explore_data(df)

    # ### CLEAN DATA THROUGH PIPELINE ####
    housing = strat_train_set.copy()
    housing = housing.drop("median_house_value", axis=1)
    # housing = impute_na(strat_train_set)
    housing_labels = strat_train_set["median_house_value"].copy()

    # categorical attributes
    #   housing_cat = encode_cat(housing[["ocean_proximity"]])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    # underfitting model with cross validation
    # predictions, rmse, scores = linear_regression(housing_prepared,
    #                                               housing_labels)

    # model with cross validation to improve overfitting, still too bad
    # predictions, rmse, scores = decision_tree_regressor(housing_prepared,
    #                                                     housing_labels)

    # random forest model
    predictions, rmse, scores = random_forest_regressor(housing_prepared,
                                                        housing_labels,
                                                        save=False)
    print("Model results")
    print("Predictions: ", predictions[:5])
    print("Labels: ", list(housing_labels[:5]))
    print("Errors: ", rmse)

    tree_rmse_scores = np.sqrt(-scores)
    print("Cross validation results:")
    print("Scores: ", tree_rmse_scores)
    print("Mean: ", tree_rmse_scores.mean())
    print("Standard deviation: ", tree_rmse_scores.std())

    # grid search
    grid_search = random_forest_grid_search(housing_prepared, housing_labels)
    print("\n")
    print("Grid search results: ")
    print("Best Params: ", grid_search.best_params_)
    print("Best estimator: ", grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # evaluate the final model on the test set
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    # confidence
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                   loc=squared_errors.mean(),
                                   scale=stats.sem(squared_errors))))
