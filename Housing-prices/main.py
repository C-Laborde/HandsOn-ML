import os
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
from models.linear_regression import linear_regression
from sklearn.linear_model import LinearRegression


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
    print("Model results")
    linear_regression(housing, housing_prepared, housing_labels, full_pipeline)
