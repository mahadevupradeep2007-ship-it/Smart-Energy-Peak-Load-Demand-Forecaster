# Streamlit Dashboard

The project includes a complete Streamlit dashboard for the Smart Energy Peak Load Demand Forecaster.

## Start the dashboard

From the project root:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Streamlit will open the dashboard in your browser.

## Dashboard sections

- Demand Overview
- Historical hourly load chart
- Hourly load profile
- ML performance: MAE, RMSE and R²
- Actual vs Predicted chart
- Feature importance
- Next 24-hour forecast
- Forecast peak time
- Downloadable forecast CSV

## Input

Upload an hourly CSV from the sidebar.

Expected minimum columns:

```text
Datetime
Load
```

The app also recognizes common alternatives such as:

```text
PJME_MW
Consumption
Energy
Load_MW
```

## Model

The dashboard loads:

```text
models/energy_load_model.joblib
```

Train the model first if this file does not exist:

```bash
python src/train_model.py --data data/PJME_hourly.csv
```
