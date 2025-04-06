# HANGUK ML team: ⚡ Load Forecasting with Autoregressive Modeling 

This repository focuses on forecasting load on the energy grid using autoregressive modeling. It includes all components from data preparation and baseline modeling to training machine learning models and evaluating them using time-series cross-validation.

## 📦 Installation

To install the dependencies using [Poetry](https://python-poetry.org/):

```bash
pip install poetry
poetry install
```

## 📁 File Structure

```bash
.
├── outputs/                                 # Generated outputs (predictions)
│   ├── students_results_HANGUK_ML_ES.csv    # Prediction results for Spain region (CSV format)
│   └── students_results_HANGUK_ML_IT.csv    # Prediction results for Italy region (CSV format)
├── __init__.py                              # Marks this directory as a Python package
├── .gitignore                               # Git ignore rules
├── .python-version                          # Python version for the environment
├── baseline_models.py                       # Naive baselines (e.g., last year/month copy)
├── data.py                                  # Data loading, cleaning, aggregation, and feature engineering
├── environmentAlpiqDatathon.yml             # Conda environment (from Alpiq Datathon)
├── eval_parallel.ipynb                      # 5-fold time-series cross-validation notebook
├── forecast_models.py                       # Forecasting models (mainly LightGBM)
├── LICENSE                                  # Repository license
├── load_forecasting_autoreg.py              # Multi-step autoregressive energy grid load forecasting
├── poetry.toml                              # Poetry configuration
├── pyproject.toml                           # Dependency and build configuration
├── README.md                                # You're here!
├── results.json                             # Evaluation metrics and model predictions
└── scoring_script.py                        # Script score the final predictions
```

## 🧩 Key Files
`load_forecasting_autoreg.py`
Supports forecasting multiple timesteps ahead using an autoregressive loop with a 1-step-ahead model. Dynamically extends features to account for each predicted step.

`eval_parallel.ipynb`
Contains the full cross-validation pipeline (5-fold, time-series aware) for model comparison and feature experimentation.

`data.py`
Prepares the dataset: loading, aggregating, cleaning, and feature engineering.

`baseline_models.py`
Implements naive baselines using historical consumption (e.g., last month, same time last year).

`forecast_models.py`
Implements various forecasting models. The final version uses LightGBM for its performance and interpretability.

`scoring_script.py`
Produces a score from the final model predictions.

`results.json`
Stores evaluations for model and hyperparameter selection.

`environmentAlpiqDatathon.yml`
Conda environment definition used during the Alpiq Datathonpy.

## Important note!

Please use 'HANGUK_ML' for scoring; this is the name we use in our output files