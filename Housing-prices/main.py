# ### basic imports
import os
import pandas as pd
import numpy as np
# ### advanced imports
# from utils.test_data import split_train_test_by_id
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
# ### in-house imports
from utils.load_data import load_data
# from describe_data import describe_data

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
    # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    # 3) stratify data and split accordingly
    df["income_cat"] = pd.cut(df["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
