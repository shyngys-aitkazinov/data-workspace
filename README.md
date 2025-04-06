# HANGUK ML (HANGUK_ML for scoring)

This repository features the following shared files:

- **`load_forecasting_autoreg.py`**  
  Extended load forecasting which supports autoregressive k-timesteps into the future prediction (given a model predicting one step ahead). This includes iteratively extending the feature dataset to allow for that.

- **`eval_parallel.ipynb`**  
  Contains time-series 5-fold crossvalidation pipeline that we used to experiment with features and models.

- **`data.py`**  
  Code to handle dataset preparation (loading, aggregation, cleaning, feature engineering).

- **`baseline_models.py`**  
  First baseline model we compared against, which didn't use any ML (merely naively extending last month's and one year ago consumption into the next month).

- **`forecast_models.py`**  
  Includes models that we tried (although the final version uses LGBM).
