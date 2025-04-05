#     range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1H")
#     forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)
#     for costumer in training_set.columns.values:
#         print(costumer)
#         consumption = training_set.loc[:, costumer]

#         feature_dummy = features["temp"].loc[start_training:]

#         encoding = SimpleEncoding(consumption, feature_dummy, end_training, start_forecast, end_forecast)

#         feature_past, feature_future, consumption_clean = encoding.meta_encoding()

#         # Train
#         model = LightGBMModel()
#         model.train(feature_past, consumption_clean)

#         # Predict
#         output = model.predict(feature_future)
#         forecast[costumer] = output

#     """
#     END OF THE MODIFIABLE PART.
#     """

#     # test to make sure that the output has the expected shape.
#     dummy_error = np.abs(forecast - example_results).sum().sum()
#     assert np.all(forecast.columns == example_results.columns), "Wrong header or header order."
#     assert np.all(forecast.index == example_results.index), "Wrong index or index order."
#     assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
#     assert forecast.isna().sum().sum() == 0, "NaN in forecast."
#     # Your solution will be evaluated using
#     # forecast_error = np.abs(forecast - testing_set).sum().sum(),
#     # and then doing a weighted sum the two portfolios:
#     # score = forecast_error_IT + 5 * forecast_error_ES

#     forecast.to_csv(join(output_path, "students_results_" + team_name + "_" + country + ".csv"))


# if __name__ == "__main__":
#     country = "IT"  # it can be ES or IT
#     main(country)
