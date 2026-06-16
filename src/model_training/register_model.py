import mlflow
import os

def register(model_name: str = "tdtoptw_dqn"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    experiment = mlflow.get_experiment_by_name("tdtoptw_dqn")
    if experiment is None:
        raise ValueError("Experiment 'tdtoptw_dqn' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("No runs found in experiment 'tdtoptw_dqn'")

    run_id = runs.iloc[0]["run_id"]
    print(f"Using latest run_id: {run_id}")

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model version: {result.version}")
    return result.version

if __name__ == "__main__":
    register()
