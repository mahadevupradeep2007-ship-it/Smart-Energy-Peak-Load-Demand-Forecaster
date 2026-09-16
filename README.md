# Smart Energy Peak Load Demand Forecaster

IBM Case Study project: **Smart Energy Peak Load Demand Forecaster**

This project implements an end-to-end machine-learning workflow for hourly electricity-demand forecasting.

## Project workflow

1. Load hourly energy-consumption data
2. Clean and validate timestamps
3. Sort observations chronologically
4. Create calendar features
5. Create historical demand features
6. Split the time series chronologically
7. Train a Random Forest regression model
8. Evaluate using MAE, RMSE and R²
9. Identify historical peak demand
10. Forecast the next 24 hours
11. Save the trained model and results
12. Display results through an optional Streamlit dashboard

## Features

- Hour
- Day
- Day of week
- Month
- Weekend flag
- Previous-hour load
- Previous-day load
- Previous-week load
- 24-hour rolling average

## Target

Hourly electricity load / consumption.

## Expected CSV

The input CSV must contain:

```text
Datetime,Load
2025-01-01 00:00:00,123.45
2025-01-01 01:00:00,118.20
...
```

The code also recognizes common load-column names such as `PJME_MW` and `Consumption`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train_model.py --data data/PJME_hourly.csv
```

## Forecast

```bash
python src/predict.py --model models/energy_load_model.joblib --data data/PJME_hourly.csv
```

## Streamlit Dashboard

The project includes a complete interactive Streamlit dashboard.

```bash
streamlit run src/app.py
```

The dashboard provides demand KPIs, hourly trends, model metrics, actual-vs-predicted charts, feature importance, next-24-hour forecasting, and CSV download.

## Generated outputs

```text
outputs/
├── metrics.csv
├── test_predictions.csv
├── next_24h_forecast.csv
├── actual_vs_predicted.png
└── feature_importance.png
```

## Model

RandomForestRegressor is used as the baseline regression model.

The split is chronological rather than random because this is a time-series forecasting problem.

## Important

This is an educational/case-study prototype. It should not be treated as a production grid-control system without additional validation, real-time data, weather/contextual variables, monitoring and operational testing.
