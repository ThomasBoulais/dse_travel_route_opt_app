# loader.py

import mlflow
import mlflow.pytorch
from typing import Optional


def load_model(model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
    """
    Charge un DQN trained model depuis le Model Registry de MLFlow 

    Parameters
    ----------
    model_name : str
        Nom du registered model (par ex. "tdtoptw_dqn")
    version : str, optional
        Num de version spécifique (par ex. "3")
        Mutuellement exclusif avec `stage`
    stage : str, optional
        Statut du modèle à charger ("Staging", "Production")
        Mutuellement exclusif avec `version`

    Returns
    -------
    torch.nn.Module
        La version chargée du PyTorch model pour l'inférence

    Raises
    ------
    ValueError
        Si `version` et `stage` sont toutes les 2 renseignées
    """

    if version and stage:
        raise ValueError("Specify either a version OR a stage, not both.")

    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/Production"

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model
