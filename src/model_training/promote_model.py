import os
from mlflow import MlflowClient
import mlflow

def promote(model_name="tdtoptw_dqn", version=1):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Model {model_name} v{version} promoted to Production")

if __name__ == "__main__":
    promote()
