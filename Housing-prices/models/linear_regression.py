from sklearn.linear_model import LinearRegression


def linear_regression(data, prepared_data, labels, pipeline):
    lin_reg = LinearRegression()
    # we fit the model to processed data
    lin_reg.fit(prepared_data, labels)

    some_data = data.iloc[:5]
    some_labels = labels.iloc[:5]
    # we test the model on raw data
    some_data_prepared = pipeline.transform(some_data)
    print("Predictions: ", lin_reg.predict(some_data_prepared))
    print("Labels: ", list(some_labels))
