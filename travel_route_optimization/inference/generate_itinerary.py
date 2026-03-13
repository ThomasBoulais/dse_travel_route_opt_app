from pathlib import Path 
import yaml
import torch
import mlflow
import numpy as np
import geopandas as gpd

from dataclasses import dataclass
from travel_route_optimization.model_training.train_dqn import load_env_train, QNet
from travel_route_optimization.inference.env_v1 import TDTOPTWEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Data class for a single itinerary step
# ---------------------------------------------------------
@dataclass
class RouteStep:
    poi_idx: int
    poi_name: str
    day: int
    arrival_minute: int
    travel_time: float
    visit_duration: float
    departure_day: int
    departure_minute: int
    category: str
    is_accommodation: bool

    def to_dict(self):
        return {
            "poi_idx": int(self.poi_idx),
            "poi_name": self.poi_name,
            "day": int(self.day),
            "arrival_minute": int(self.arrival_minute),
            "travel_time": float(self.travel_time),
            "visit_duration": float(self.visit_duration),
            "departure_day": int(self.departure_day),
            "departure_minute": int(self.departure_minute),
            "category": self.category,
            "is_accommodation": bool(self.is_accommodation),
        }


# ---------------------------------------------------------
# Load POI metadata (names, categories, accommodation flag)
# ---------------------------------------------------------
def load_pois_and_metadata(cfg):
    pois = gpd.read_parquet(cfg["data"]["pois_geoparquet"]).reset_index(drop=True)
    main_categories = pois["main_category"].to_numpy()
    is_accommodation = pois["categories"].apply(
        lambda s: "accomodation" in s if isinstance(s, str) else False
    ).to_numpy()
    return pois, main_categories, is_accommodation


# ---------------------------------------------------------
# Greedy route generation (same logic as eval_route.py)
# ---------------------------------------------------------
def generate_route(env: TDTOPTWEnv, qnet: QNet, pois, main_categories, is_accommodation, max_steps=150):
    state = env.reset()
    route_steps = []

    for _ in range(max_steps):
        feas = env._feasible_actions_mask()
        if not feas.any():
            break

        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = qnet(s_t).squeeze(0).cpu().numpy()

        q_vals[~feas] = -1e9
        action_idx = int(np.argmax(q_vals))

        neighbors = env.knn_neighbors[env.current_poi]
        if action_idx >= len(neighbors):
            break

        current_poi = env.current_poi
        next_poi = neighbors[action_idx]

        travel_t = env.travel_time[current_poi, next_poi]
        arrival_day, arrival_minute = env._time_to_indices(travel_t)
        visit_dur = env.visit_durations[next_poi]

        next_state, reward, done, info = env.step(action_idx)
        departure_day = env.current_day
        departure_minute = env.current_minute

        route_steps.append(
            RouteStep(
                poi_idx=next_poi,
                poi_name=pois.iloc[next_poi]["name"],
                day=arrival_day,
                arrival_minute=arrival_minute,
                travel_time=float(travel_t),
                visit_duration=float(visit_dur),
                departure_day=departure_day,
                departure_minute=departure_minute,
                category=main_categories[next_poi],
                is_accommodation=bool(is_accommodation[next_poi]),
            )
        )

        state = next_state
        if done:
            break

    return route_steps


# ---------------------------------------------------------
# Main function used by FastAPI
# ---------------------------------------------------------
def generate_itinerary(run_id: str, start_poi: int, start_day: int, num_days: int, config_path: str):
    config_path = Path(config_path)

    # If the path is not absolute, resolve it relative to this file
    if not config_path.is_absolute():
        base_dir = Path(__file__).parent
        config_path = (base_dir / ".." / "model_training" / config_path).resolve()
    
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load environment
    env = load_env_train(config_path)
    env.start_poi_idx = start_poi
    env.start_day = start_day
    env.num_days = num_days

    # Load POI metadata
    pois, main_categories, is_accommodation = load_pois_and_metadata(cfg)

    # Build model architecture
    state_dim = env._get_state().shape[0]
    n_actions = env.max_actions
    qnet = QNet(state_dim, n_actions).to(device)

    # Load model weights from MLflow
    model_uri = f"runs:/{run_id}/tdtoptw_dqn.pt"
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    state_dict = torch.load(local_path, map_location=device)
    qnet.load_state_dict(state_dict)
    qnet.eval()

    # Generate route
    route = generate_route(env, qnet, pois, main_categories, is_accommodation)

    # Return JSON-serializable itinerary
    return [step.to_dict() for step in route]
