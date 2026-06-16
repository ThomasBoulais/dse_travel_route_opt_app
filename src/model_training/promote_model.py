import os
from mlflow import MlflowClient
import mlflow

def promote(model_name="tdtoptw_dqn"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    client = MlflowClient()
    
    try:
        all_versions = client.search_model_versions(f'name="{model_name}"')
        
        if not all_versions:
            print(f"ERROR: No versions found for model '{model_name}'")
            return False
        
        highest_version = max(all_versions, key=lambda v: int(v.version))
        
        print(f"Found {len(all_versions)} total version(s)")
        print(f"Highest version: v{highest_version.version}, current stage: {highest_version.current_stage}")
        
        client.transition_model_version_stage(
            name=model_name,
            version=highest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✓ Model {model_name} v{highest_version.version} promoted to Production")
        
        updated = client.get_model_version(model_name, highest_version.version)
        print(f"✓ Verification: v{updated.version} is now in stage: {updated.current_stage}")
        return True
        
    except Exception as e:
        print(f"ERROR during promotion: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    promote()
