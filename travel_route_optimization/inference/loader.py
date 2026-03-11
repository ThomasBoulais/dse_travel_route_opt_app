import mlflow
import mlflow.pytorch
from typing import Optional


def load_model(model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
    """
    Load a trained DQN model from the MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        Name of the registered model (e.g., "tdtoptw_dqn").
    version : str, optional
        Specific version number to load (e.g., "3").
        Mutually exclusive with `stage`.
    stage : str, optional
        Model stage to load ("Staging", "Production").
        Mutually exclusive with `version`.

    Returns
    -------
    torch.nn.Module
        The loaded PyTorch model ready for inference.

    Raises
    ------
    ValueError
        If both `version` and `stage` are provided.
    """

    if version and stage:
        raise ValueError("Specify either a version OR a stage, not both.")

    # Build MLflow URI
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Default: load the latest Production model
        model_uri = f"models:/{model_name}/Production"

    # Load the model
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    return model
