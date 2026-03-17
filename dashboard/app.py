"""
Streamlit dashboard for AutoML Studio.
Provides dataset upload, pipeline training, results visualization, and prediction UI.
"""

import sys
from pathlib import Path

# Fix Python path for src imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.pipeline.train_pipeline import TrainingPipeline
from api.predictor import Predictor


# ---------------------------------------------------
# App Setup
# ---------------------------------------------------

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML Studio Dashboard")


# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------

defaults = {
    "experiment_id": None,
    "dataset_path": None,
    "target_column": None,
    "task_type": None,
    "best_model_name": None,
    "best_metrics": None,
    "leaderboard": None,
    "config": None,
    "predictor": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Upload & Configure",
        "Train Pipeline",
        "Results & Leaderboard",
        "Predict",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("AutoML Studio v1.0")


# ---------------------------------------------------
# Page 1 — Upload & Configure
# ---------------------------------------------------

if page == "Upload & Configure":

    st.header("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        # Save dataset to data/raw
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = data_dir / uploaded_file.name
        df.to_csv(file_path, index=False)

        st.session_state.dataset_path = str(file_path)

        st.subheader("Configuration")

        target_column = st.selectbox(
            "Select Target Column",
            df.columns,
        )

        st.session_state.target_column = target_column

        # Detect task type
        if df[target_column].nunique() <= 10:
            task_type = "classification"
        else:
            task_type = "regression"

        st.session_state.task_type = task_type

        st.info(f"Detected Task Type: **{task_type}**")

        st.subheader("Training Settings")

        test_size = st.slider(
            "Test Size",
            0.1,
            0.4,
            0.2,
        )

        n_trials = st.number_input(
            "Optuna Trials",
            1,
            50,
            5,
        )

        if st.button("Save Configuration"):

            config = load_config()

            config.data.dataset_path = st.session_state.dataset_path
            config.data.target_column = target_column

            config.data.test_size = test_size
            config.models.n_trials = int(n_trials)

            st.session_state.config = config

            st.success("Configuration saved successfully!")

    else:
        st.info("Upload a CSV file to begin.")


# ---------------------------------------------------
# Page 2 — Train Pipeline
# ---------------------------------------------------

elif page == "Train Pipeline":

    st.header("Train AutoML Pipeline")

    if st.session_state.config is None:
        st.warning("Please upload and configure a dataset first.")
        st.stop()

    st.subheader("Training Configuration")

    col1, col2, col3 = st.columns(3)
    col1.info(f"Dataset: `{st.session_state.dataset_path}`")
    col2.info(f"Target: `{st.session_state.target_column}`")
    col3.info(f"Task: `{st.session_state.task_type}`")

    if st.button("Start Training"):

        with st.spinner("Running AutoML pipeline... This may take a few minutes."):

            try:

                pipeline = TrainingPipeline(
                    config=st.session_state.config
                )

                best_metrics = pipeline.run()

                st.session_state.experiment_id = pipeline.experiment_id
                st.session_state.best_metrics = best_metrics
                st.session_state.best_model_name = best_metrics.get("model_name")
                st.session_state.predictor = None  # reset predictor on retrain

                st.success("Training completed successfully!")

                st.subheader("Best Model Metrics")
                st.json(best_metrics)

            except Exception as e:

                st.error("Pipeline execution failed.")
                st.exception(e)


# ---------------------------------------------------
# Page 3 — Results & Leaderboard
# ---------------------------------------------------

elif page == "Results & Leaderboard":

    st.header("Results & Leaderboard")

    if st.session_state.experiment_id is None:
        st.warning("Run the training pipeline first.")
        st.stop()

    experiment_id = st.session_state.experiment_id

    st.caption(f"Experiment ID: `{experiment_id}`")

    leaderboard_path = (
        Path("artifacts/experiments")
        / experiment_id
        / "leaderboard.csv"
    )

    if not leaderboard_path.exists():
        st.error("Leaderboard file not found.")
        st.stop()

    leaderboard = pd.read_csv(leaderboard_path)
    st.session_state.leaderboard = leaderboard

    st.subheader("Model Leaderboard")
    st.dataframe(leaderboard)

    # Detect primary metric automatically from leaderboard columns
    metric_columns = [
        c for c in leaderboard.columns if c != "model_name"
    ]
    primary_metric = metric_columns[0]

    st.subheader(f"Model Performance — {primary_metric}")
    chart_data = leaderboard.set_index("model_name")[primary_metric]
    st.bar_chart(chart_data)

    st.subheader("Best Model Metrics")

    best_metrics = st.session_state.best_metrics

    if best_metrics:

        filtered_metrics = {
            k: v for k, v in best_metrics.items()
            if k != "model_name"
        }

        cols = st.columns(len(filtered_metrics))

        for i, (metric, value) in enumerate(filtered_metrics.items()):
            cols[i].metric(metric, round(value, 4))


# ---------------------------------------------------
# Page 4 — Predict
# ---------------------------------------------------

elif page == "Predict":

    st.header("Run Prediction")

    if st.session_state.experiment_id is None:
        st.warning("Train a model first.")
        st.stop()

    config = st.session_state.config
    logger = get_logger("dashboard_predict", config)

    # Load predictor once into session state
    if st.session_state.predictor is None:
        predictor = Predictor(config, logger)
        predictor.load_artifacts(st.session_state.experiment_id)
        st.session_state.predictor = predictor

    predictor = st.session_state.predictor

    dataset_path = st.session_state.dataset_path
    df = pd.read_csv(dataset_path)
    target_column = st.session_state.target_column
    feature_columns = [c for c in df.columns if c != target_column]

    st.subheader("Input Features")

    feature_inputs = {}

    for col in feature_columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_inputs[col] = st.number_input(
                col,
                value=float(df[col].median()),
            )
        else:
            options = df[col].dropna().unique().tolist()
            feature_inputs[col] = st.selectbox(col, options)

    if st.button("Predict"):

        result = predictor.predict(feature_inputs)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", result["prediction"])

        if result.get("probability") is not None:
            col2.metric("Probability", round(result["probability"], 4))

        st.caption(f"Model: {result['model_name']}")
        st.caption(f"Experiment ID: {result['experiment_id']}")