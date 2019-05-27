import matplotlib.pyplot as plt
from utils.load_data import load_data

df = load_data()


def describe_data(df):
    print("Header:\n")
    print(df.head())
    print("Description:\n")
    print(df.describe())
    print("Info:\n")
    print(df.info())
    df.hist()
    plt.show()


if __name__ == "__main__":
    describe_data(df)
