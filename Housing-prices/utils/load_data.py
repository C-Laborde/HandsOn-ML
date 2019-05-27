import os
import pandas as pd

HOUSING_PATH = os.path.join("datasets", "housing")


def load_data(path=HOUSING_PATH):
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)
