
from pathlib import Path

import pandas as pd
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MLflow Experiments",
    page_icon="🧪",
    layout="wide",
)

EXPERIMENT_NAME = "Real_Estate_Investment_Advisor"


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🧪 MLflow Experiment Tracking")

st.markdown(
    """
    Explore machine-learning experiments, model parameters,
    evaluation metrics and logged artifacts.
    """
)


# --------------------------------------------------
# CONNECT TO MLFLOW
# --------------------------------------------------

st.subheader("Experiment Connection")

try:
    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        st.error(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

        st.info(
            "Make sure the training script has completed and "
            "the application is using the same MLflow tracking store."
        )

        st.stop()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100,
        order_by=["attributes.start_time DESC"],
    )

except Exception as error:
    st.error(f"Unable to connect to MLflow: {error}")

    st.info(
        """
        Check that MLflow is installed and that the application
        is using the same tracking URI as the training script.
        """
    )

    st.stop()


# --------------------------------------------------
# EXPERIMENT INFORMATION
# --------------------------------------------------

st.success("Successfully connected to the MLflow experiment.")

info_col1, info_col2, info_col3 = st.columns(3)

info_col1.metric(
    "Experiment ID",
    str(experiment.experiment_id),
)

info_col2.metric(
    "Total Runs",
    str(len(runs)),
)

info_col3.metric(
    "Experiment Status",
    experiment.lifecycle_stage.capitalize(),
)

st.write("**Experiment Name:**", experiment.name)

st.write("**Tracking URI:**", mlflow.get_tracking_uri())

st.divider()


# --------------------------------------------------
# PREPARE RUN DATA
# --------------------------------------------------

run_records = []

for run in runs:

    run_records.append({
        "Run ID": run.info.run_id,
        "Run Name": run.data.tags.get("mlflow.runName", "Unnamed"),
        "Status": run.info.status,
        "Start Time": (
            pd.to_datetime(
                run.info.start_time,
                unit="ms",
                errors="coerce",
            )
            if run.info.start_time
            else None
        ),
        "Accuracy": run.data.metrics.get("accuracy"),
        "Precision": run.data.metrics.get("precision"),
        "Recall": run.data.metrics.get("recall"),
        "F1 Score": run.data.metrics.get("f1_score"),
        "MAE": run.data.metrics.get("mae"),
        "RMSE": run.data.metrics.get("rmse"),
        "R2": run.data.metrics.get("r2"),
    })

runs_df = pd.DataFrame(run_records)


# --------------------------------------------------
# RUN SUMMARY
# --------------------------------------------------

st.header("1. Experiment Runs")

if runs_df.empty:

    st.warning("No runs were found for this experiment.")

else:

    st.dataframe(
        runs_df,
        use_container_width=True,
        hide_index=True,
    )

    successful_runs = sum(
        1 for run in runs if run.info.status == "FINISHED"
    )

    failed_runs = sum(
        1 for run in runs if run.info.status == "FAILED"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Successful Runs",
        successful_runs,
    )

    col2.metric(
        "Failed Runs",
        failed_runs,
    )

    col3.metric(
        "Other Runs",
        len(runs) - successful_runs - failed_runs,
    )


st.divider()


# --------------------------------------------------
# SELECT A RUN
# --------------------------------------------------

st.header("2. Run Details")

run_options = {
    (
        f"{run.data.tags.get('mlflow.runName', 'Unnamed')} "
        f"({run.info.run_id[:8]})"
    ): run.info.run_id
    for run in runs
}

if run_options:

    selected_label = st.selectbox(
        "Select an experiment run",
        list(run_options.keys()),
    )

    selected_run_id = run_options[selected_label]

    selected_run = client.get_run(selected_run_id)

    tab1, tab2, tab3 = st.tabs([
        "Metrics",
        "Parameters and Tags",
        "Artifacts",
    ])

    with tab1:

        st.subheader("Logged Metrics")

        if selected_run.data.metrics:

            metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": name,
                        "Value": value,
                    }
                    for name, value in selected_run.data.metrics.items()
                ]
            )

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No metrics were logged for this run.")


    with tab2:

        st.subheader("Parameters")

        if selected_run.data.params:

            params_df = pd.DataFrame(
                [
                    {
                        "Parameter": name,
                        "Value": value,
                    }
                    for name, value in selected_run.data.params.items()
                ]
            )

            st.dataframe(
                params_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No parameters were logged.")

        st.subheader("Tags")

        if selected_run.data.tags:

            tags_df = pd.DataFrame(
                [
                    {
                        "Tag": name,
                        "Value": value,
                    }
                    for name, value in selected_run.data.tags.items()
                ]
            )

            st.dataframe(
                tags_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No tags were logged.")


    with tab3:

        st.subheader("Logged Artifacts")

        try:

            artifacts = client.list_artifacts(
                selected_run_id
            )

            if artifacts:

                artifact_records = [
                    {
                        "Artifact": artifact.path,
                        "Is Directory": artifact.is_dir,
                        "Size (Bytes)": artifact.file_size,
                    }
                    for artifact in artifacts
                ]

                st.dataframe(
                    pd.DataFrame(artifact_records),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info("No artifacts were found for this run.")

        except Exception as error:

            st.warning(
                f"Unable to list artifacts: {error}"
            )


st.divider()


# --------------------------------------------------
# MLFLOW UI INFORMATION
# --------------------------------------------------

st.header("3. MLflow Tracking Interface")

st.write(
    """
    MLflow provides a separate interface for reviewing experiment
    history, comparing runs, and inspecting logged model artifacts.
    """
)

st.code(
    "python -m mlflow ui --host 127.0.0.1 --port 5000",
    language="powershell",
)

st.markdown(
    "When the MLflow UI is running, open "
    "[http://127.0.0.1:5000](http://127.0.0.1:5000) "
    "in your browser."
)

st.info(
    """
    The Streamlit page reads experiment data from the MLflow
    tracking store. The separate MLflow UI is an additional
    interface, not a requirement for the dashboard to display runs.
    """
)
