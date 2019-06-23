import pandas as pd
from sklearn.impute import SimpleImputer


def impute_na(train_set):
    housing = train_set.drop("median_house_value", axis=1)
    # Missing data is imputed with the median value:
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

    return housing_tr
