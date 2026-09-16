from pathlib import Path
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from features import FEATURES, load_data, make_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def main():


    parser = argparse.ArgumentParser(
        description="Train Smart Energy Peak Load Demand Forecaster"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to hourly energy CSV file",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_DIR),
        help="Model directory",
    )

    args = parser.parse_args()


    output_dir = Path(args.output)
    model_dir = Path(args.model)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("Loading data...")

    df = load_data(
        args.data
    )

    print(
        f"Records: {len(df):,}"
    )

    print(
        f"Range: "
        f"{df.index.min()} -> "
        f"{df.index.max()}"
    )


    data = make_features(
        df
    )


    split = int(
        len(data) * 0.80
    )

    train = data.iloc[:split]

    test = data.iloc[split:]


    X_train = train[FEATURES]

    y_train = train["Load"]

    X_test = test[FEATURES]

    y_test = test["Load"]


    print(
        f"Training records: {len(train):,}"
    )

    print(
        f"Testing records: {len(test):,}"
    )



    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=20,

        min_samples_leaf=2,

        random_state=42,

        # One worker avoids unnecessary sklearn/joblib
        # parallelization warnings during this project.
        n_jobs=1,
    )


    print(
        "Training Random Forest..."
    )


    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    metrics = pd.DataFrame(
        [{
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }]
    )


    metrics.to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "Datetime": test.index,
            "Actual_Load": y_test.values,
            "Predicted_Load": predictions,
        }
    )


    predictions_df.to_csv(
        output_dir /
        "test_predictions.csv",
        index=False,
    )


    peak_idx = df["Load"].idxmax()

    peak_value = df["Load"].max()

    average_value = df["Load"].mean()

    minimum_value = df["Load"].min()


    demand_summary = pd.DataFrame(
        [{
            "Peak_Load": peak_value,
            "Peak_Datetime": peak_idx,
            "Average_Load": average_value,
            "Minimum_Load": minimum_value,
        }]
    )


    demand_summary.to_csv(
        output_dir /
        "demand_summary.csv",
        index=False,
    )

    plot_df = predictions_df.tail(
        24 * 7
    )


    plt.figure(
        figsize=(14, 5)
    )


    plt.plot(
        plot_df["Datetime"],
        plot_df["Actual_Load"],
        label="Actual Load",
        linewidth=2,
    )


    plt.plot(
        plot_df["Datetime"],
        plot_df["Predicted_Load"],
        label="Predicted Load",
        linewidth=2,
    )


    plt.title(
        "Actual vs Predicted Energy Load"
    )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "Load"
    )


    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()


    plt.savefig(
        output_dir /
        "actual_vs_predicted.png",
        dpi=180,
    )

    plt.close()


    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )


    importance.to_csv(
        output_dir /
        "feature_importance.csv",
        index=False,
    )


    plt.figure(
        figsize=(10, 5)
    )


    plt.barh(
        importance["Feature"][::-1],
        importance["Importance"][::-1],
    )


    plt.title(
        "Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )


    plt.tight_layout()


    plt.savefig(
        output_dir /
        "feature_importance.png",
        dpi=180,
    )


    plt.close()



    model_path = (
        model_dir /
        "energy_load_model.joblib"
    )


    joblib.dump(
        {
            "model": model,

            "features": FEATURES,

            "sklearn_version":
                sklearn.__version__,
        },
        model_path,
    )


    print(
        "\n=== MODEL PERFORMANCE ==="
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


    print(
        "\n=== DEMAND ==="
    )

    print(
        f"Peak Load    : "
        f"{peak_value:.4f}"
    )

    print(
        f"Peak Datetime: "
        f"{peak_idx}"
    )

    print(
        f"Average Load : "
        f"{average_value:.4f}"
    )

    print(
        f"Minimum Load : "
        f"{minimum_value:.4f}"
    )


    print(
        "\n=== MODEL FILE ==="
    )

    print(
        f"Saved to: "
        f"{model_path}"
    )

    print(
        f"Scikit-learn version: "
        f"{sklearn.__version__}"
    )


    print(
        "\n=== OUTPUT DIRECTORY ==="
    )

    print(
        output_dir
    )

    print(
        "\nModel and outputs saved successfully."
    )


if __name__ == "__main__":
    main()