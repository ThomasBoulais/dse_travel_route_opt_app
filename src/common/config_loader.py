from omegaconf import OmegaConf
from pathlib import Path

def load_config():
    config_path = Path(__file__).parents[2] / "configs/config.yaml"
    return OmegaConf.load(config_path)


