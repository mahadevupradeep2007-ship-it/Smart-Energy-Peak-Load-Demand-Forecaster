from pathlib import Path
import argparse

import joblib
import pandas as pd

from features import FEATURES, load_data


def forecast_next_24_hours(model, history):
    """
    Generate recursive forecasts for the next 24 hours.

    The model is temporarily configured to use one worker
    during forecasting to avoid unnecessary joblib/scikit-learn
    parallelization warnings.
    """

    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    work = history.copy()
    rows = []

    for _ in range(24):

        next_time = (
            work.index[-1]
            + pd.Timedelta(hours=1)
        )

        row = pd.DataFrame(
            [{
                "hour": next_time.hour,
                "day": next_time.day,
                "dayofweek": next_time.dayofweek,
                "month": next_time.month,
                "is_weekend": int(
                    next_time.dayofweek >= 5
                ),
                "lag_1": work["Load"].iloc[-1],
                "lag_24": work["Load"].iloc[-24],
                "lag_168": work["Load"].iloc[-168],
                "rolling_mean_24": (
                    work["Load"]
                    .iloc[-24:]
                    .mean()
                ),
            }],
            index=[next_time],
        )

        prediction = float(
            model.predict(
                row[FEATURES]
            )[0]
        )

        work.loc[
            next_time,
            "Load"
        ] = prediction

        rows.append(
            {
                "Datetime": next_time,
                "Forecast_Load": prediction,
            }
        )

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--output",
        default="../outputs/next_24h_forecast.csv"
    )

    args = parser.parse_args()

    package = joblib.load(
        args.model
    )

    model = package["model"]

    history = load_data(
        args.data
    )

    forecast = forecast_next_24_hours(
        model,
        history
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast.to_csv(
        output,
        index=False
    )

    peak = forecast.loc[
        forecast["Forecast_Load"].idxmax()
    ]

    print(
        "\n=== NEXT 24 HOURS ==="
    )

    print(
        forecast.to_string(
            index=False
        )
    )

    print(
        "\n=== FORECAST PEAK ==="
    )

    print(
        f"Load: {peak['Forecast_Load']:.4f}"
    )

    print(
        f"Time: {peak['Datetime']}"
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()