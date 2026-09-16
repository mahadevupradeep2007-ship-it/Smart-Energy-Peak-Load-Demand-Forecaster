import numpy as np
import pandas as pd

FEATURES = [
    "hour",
    "day",
    "dayofweek",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]


def detect_load_column(df):
    candidates = {
        "load", "consumption", "pjme_mw", "energy",
        "consumption_mw", "load_mw"
    }

    for col in df.columns:
        if col.lower() in candidates:
            return col

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric) == 1:
        return numeric[0]

    raise ValueError(
        "Could not detect the load column. Rename the column to "
        "'Load', 'Consumption', or 'PJME_MW'."
    )


def load_data(path):
    df = pd.read_csv(path)

    if "Datetime" not in df.columns:
        raise ValueError("CSV must contain a 'Datetime' column.")

    load_col = detect_load_column(df)

    df = df[["Datetime", load_col]].copy()
    df.columns = ["Datetime", "Load"]

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Load"] = pd.to_numeric(df["Load"], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates(subset="Datetime")
    df = df.sort_values("Datetime")
    df = df.set_index("Datetime")

    df = df.resample("h").mean()
    df["Load"] = df["Load"].interpolate(method="time").ffill().bfill()

    return df


def make_features(df):
    x = df.copy()

    x["hour"] = x.index.hour
    x["day"] = x.index.day
    x["dayofweek"] = x.index.dayofweek
    x["month"] = x.index.month
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)

    x["lag_1"] = x["Load"].shift(1)
    x["lag_24"] = x["Load"].shift(24)
    x["lag_168"] = x["Load"].shift(168)
    x["rolling_mean_24"] = x["Load"].shift(1).rolling(24).mean()

    return x.dropna()
