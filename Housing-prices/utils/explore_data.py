import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def explore_data(data):
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=data["population"]/100, label="population", figsize=(10, 7),
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()

    print(data.corr())

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(data[attributes], figsize=(12, 8))

    data.plot(kind="scatter", x="median_income", y="median_house_value",
              alpha=0.1)
