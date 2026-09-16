import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from features import load_data
from predict import forecast_next_24_hours


st.set_page_config(
    page_title="Smart Energy Peak Load Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(
    """
    <style>


    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }



    .hero {
        background: linear-gradient(
            135deg,
            #111111 0%,
            #1f1f1f 72%,
            #0f62fe 72%
        );

        padding: 28px 32px;
        border-radius: 14px;

        color: white;

        margin-bottom: 24px;

        border-left: 6px solid #0f62fe;

        box-shadow:
            0 4px 15px rgba(0, 0, 0, 0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff !important;
    }

    .hero p {
        margin-top: 8px;
        color: #e8e8e8 !important;
        font-size: 1rem;
    }


    .section-title {
        border-left: 5px solid #0f62fe;

        padding-left: 10px;

        margin-top: 28px;
        margin-bottom: 14px;

        color: #ffffff !important;
        font-weight: 700;
    }



    [data-testid="stMetric"] {

        background-color: #ffffff !important;

        border: 1px solid #dde3ea;

        border-radius: 10px;

        padding: 16px;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.05);
    }



    [data-testid="stMetricLabel"] {
        color: #161616 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #161616 !important;
        font-weight: 600 !important;
    }



    [data-testid="stMetricValue"] {
        color: #161616 !important;
    }

    [data-testid="stMetricValue"] div {
        color: #161616 !important;
        font-weight: 700 !important;
    }


    [data-testid="stMetricDelta"] {
        color: #525252 !important;
    }


    [data-testid="stDataFrame"] {
        color: #161616 !important;
    }



    [data-testid="stSidebar"] {
        background-color: #252631;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }


    .stCaption {
        color: #d0d0d0 !important;
    }


    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
    }



    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }



    hr {
        border-color: #3a3a3a;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">

        <h1>
            ⚡ Smart Energy Peak Load Demand Forecaster
        </h1>

        <p>
            IBM Case Study • Machine Learning + Streamlit Dashboard
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)



st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload hourly energy CSV",
    type=["csv"],
    help=(
        "CSV must contain a Datetime column and "
        "a load/consumption column."
    ),
)

default_model = (
    ROOT /
    "models" /
    "energy_load_model.joblib"
)


st.sidebar.markdown("---")

st.sidebar.subheader("Expected data")

st.sidebar.code(
    "Datetime,Load\n"
    "2025-01-01 00:00,500.2"
)


if uploaded_file is None:

    st.info(
        "Upload your hourly energy-consumption CSV "
        "from the sidebar to start the forecasting dashboard."
    )

    st.markdown(
        """
        ### Dashboard capabilities

        #### 📊 Demand Monitoring

        - Historical peak load
        - Average load
        - Minimum load
        - Hourly demand trend

        #### 🔮 Forecasting

        - Next 24-hour load forecast
        - Predicted peak period

        #### 🤖 ML Evaluation

        - MAE
        - RMSE
        - R² Score

        #### 🔍 Model Insights

        - Feature importance
        - Actual vs predicted demand
        """
    )

    st.stop()

temp_path = (
    ROOT /
    "data" /
    "_streamlit_upload.csv"
)

temp_path.write_bytes(
    uploaded_file.getvalue()
)



try:

    df = load_data(
        str(temp_path)
    )

except Exception as exc:

    st.error(
        f"Could not process the uploaded file: {exc}"
    )

    st.stop()



st.markdown(
    '<h2 class="section-title">'
    '📊 Demand Overview'
    '</h2>',
    unsafe_allow_html=True,
)



peak_time = df["Load"].idxmax()

peak_load = float(
    df["Load"].max()
)

average_load = float(
    df["Load"].mean()
)

minimum_load = float(
    df["Load"].min()
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Peak Load",
    f"{peak_load:,.2f}"
)

c2.metric(
    "Average Load",
    f"{average_load:,.2f}"
)

c3.metric(
    "Minimum Load",
    f"{minimum_load:,.2f}"
)

c4.metric(
    "Observations",
    f"{len(df):,}"
)


st.caption(
    f"Data range: {df.index.min()} → "
    f"{df.index.max()} | "
    f"Historical peak: {peak_time}"
)


st.markdown(
    '<h2 class="section-title">'
    '📈 Historical Energy Demand'
    '</h2>',
    unsafe_allow_html=True,
)

st.line_chart(
    df["Load"],
    height=360
)


st.markdown(
    '<h2 class="section-title">'
    '🕒 Hourly Load Profile'
    '</h2>',
    unsafe_allow_html=True,
)


hourly_profile = (
    df
    .assign(
        Hour=df.index.hour
    )
    .groupby("Hour")["Load"]
    .mean()
)


st.line_chart(
    hourly_profile,
    height=300
)


if not default_model.exists():

    st.warning(
        "No trained model was found in models/. "
        "Run train_model.py first, then refresh the dashboard."
    )

    st.stop()


try:

    package = joblib.load(
        default_model
    )

    model = package["model"]

except Exception as exc:

    st.error(
        f"Could not load the trained model: {exc}"
    )

    st.stop()


metrics_path = (
    ROOT /
    "outputs" /
    "metrics.csv"
)


if metrics_path.exists():

    metrics = pd.read_csv(
        metrics_path
    ).iloc[0]


    st.markdown(
        '<h2 class="section-title">'
        '🤖 Model Performance'
        '</h2>',
        unsafe_allow_html=True,
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "MAE",
        f"{metrics['MAE']:.4f}"
    )


    m2.metric(
        "RMSE",
        f"{metrics['RMSE']:.4f}"
    )


    m3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )


predictions_path = (
    ROOT /
    "outputs" /
    "test_predictions.csv"
)


if predictions_path.exists():

    predictions = pd.read_csv(
        predictions_path
    )


    predictions["Datetime"] = pd.to_datetime(
        predictions["Datetime"]
    )


    st.markdown(
        '<h2 class="section-title">'
        '📈 Actual vs Predicted'
        '</h2>',
        unsafe_allow_html=True,
    )


    chart_df = (
        predictions
        .tail(24 * 7)
        .set_index("Datetime")
        [
            [
                "Actual_Load",
                "Predicted_Load"
            ]
        ]
    )


    st.line_chart(
        chart_df,
        height=360
    )



importance_path = (
    ROOT /
    "outputs" /
    "feature_importance.csv"
)


if importance_path.exists():

    importance = pd.read_csv(
        importance_path
    ).sort_values(
        "Importance",
        ascending=True
    )


    st.markdown(
        '<h2 class="section-title">'
        '🔍 Model Feature Importance'
        '</h2>',
        unsafe_allow_html=True,
    )


    st.bar_chart(
        importance.set_index(
            "Feature"
        )["Importance"],
        height=320
    )


st.markdown(
    '<h2 class="section-title">'
    '🔮 Next 24-Hour Forecast'
    '</h2>',
    unsafe_allow_html=True,
)


try:

    forecast = forecast_next_24_hours(
        model,
        df
    )

    forecast_peak = forecast.loc[
        forecast["Forecast_Load"].idxmax()
    ]


    f1, f2 = st.columns(2)


    f1.metric(
        "Forecast Peak",
        f"{forecast_peak['Forecast_Load']:,.2f}"
    )


    f2.metric(
        "Forecast Peak Time",
        str(
            forecast_peak["Datetime"]
        )
    )


    forecast_chart = (
        forecast
        .set_index("Datetime")
        [
            ["Forecast_Load"]
        ]
    )


    st.line_chart(
        forecast_chart,
        height=360
    )


    st.subheader(
        "📋 Forecast Details"
    )


    st.dataframe(
        forecast,
        width="stretch",
        hide_index=True,
    )


    csv_data = forecast.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇️ Download 24-Hour Forecast",
        data=csv_data,
        file_name="next_24h_forecast.csv",
        mime="text/csv",
        width="stretch",
    )


except Exception as exc:

    st.error(
        "The model could not generate "
        "a 24-hour forecast. "
        "Make sure the dataset contains "
        "at least 168 hourly observations. "
        f"Details: {exc}"
    )


st.markdown("---")

st.caption(
    "Smart Energy Peak Load Demand Forecaster "
    "• Educational AIML Prototype"
)