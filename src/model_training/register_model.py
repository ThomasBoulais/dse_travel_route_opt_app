import mlflow
import sys

def register(run_id: str, model_name: str = "tdtoptw_dqn"):
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Registered model version: {result.version}")

if __name__ == "__main__":
    run_id = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tdtoptw_dqn"
    register(run_id, model_name)
