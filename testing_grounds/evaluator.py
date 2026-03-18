from typing import Dict, Any
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd

from ..travel_route_optimization.inference.loader import load_model
from ..travel_route_optimization.inference.env_v1 import TDTOPTWEnv
from ..travel_route_optimization.inference.validators import validate_route
from ..travel_route_optimization.inference.scoring import score_route
from ..travel_route_optimization.inference.route_step import RouteStep


# ---------------------------------------------------------
# Load environment from config
# ---------------------------------------------------------
def load_env(config_path: str) -> TDTOPTWEnv:
    """
    Build the frozen environment (env_v1) using the same config
    that was used during training.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data paths from config
    pois_geoparquet = cfg["data"]["pois_geoparquet"]
    knn_graph_path = cfg["data"]["knn_drive_time_graph_df"]

    # Load data
    pois = gpd.read_parquet(pois_geoparquet).reset_index(drop=True)
    knn_df = pd.read_csv(knn_graph_path)

    # Build travel_time matrix
    N = len(pois)
    travel_time = np.full((N, N), np.inf, dtype=np.float32)
    for _, row in knn_df.iterrows():
        i = int(row["poi_from"])
        j = int(row["poi_to"])
        travel_time[i, j] = float(row["drive_time"]) / 60.0  # minutes

    # Coordinates
    coords = np.vstack(
        [pois.geometry.y.values, pois.geometry.x.values]
    ).T.astype(np.float32)

    visit_durations = pois["visit_duration"].to_numpy().astype(np.float32)
    poi_scores = pois["interest_score"].to_numpy().astype(np.float32)
    main_categories = pois["main_category"].to_numpy()
    is_accommodation = pois["categories"].apply(
        lambda s: "accomodation" in s if isinstance(s, str) else False
    ).to_numpy()

    # Opening mask
    opening_mask = np.array(
        [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
        dtype=np.uint8,
    )

    # Neighbor index matrix from KNN graph
    neighbors = [[] for _ in range(N)]
    for _, row in knn_df.iterrows():
        i = int(row["poi_from"])
        j = int(row["poi_to"])
        neighbors[i].append(j)

    max_neighbors = max(len(n) for n in neighbors)
    neighbor_idx = np.full((N, max_neighbors), -1, dtype=np.int32)
    for i in range(N):
        if neighbors[i]:
            neighbor_idx[i, : len(neighbors[i])] = neighbors[i]

    # POI features (lat, lon, visit_duration)
    poi_features = np.hstack(
        [coords, visit_durations.reshape(-1, 1)]
    ).astype(np.float32)

    # Build environment
    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]

    env = TDTOPTWEnv(
        poi_features=poi_features,
        poi_scores=poi_scores,
        opening_mask=opening_mask,
        travel_time=travel_time,
        neighbor_idx=neighbor_idx,
        main_categories=main_categories,
        is_accommodation=is_accommodation,
        visit_durations=visit_durations,
        start_poi_idx=env_cfg["start_poi_idx"],
        start_day=env_cfg["start_day"],
        day_start_minute=env_cfg["day_start_minute"],
        day_end_minute=env_cfg["day_end_minute"],
        num_days=env_cfg["num_days"],
        max_steps=env_cfg["max_steps"],
        reward_cfg=reward_cfg,
    )

    return env


# ---------------------------------------------------------
# Run the trained policy
# ---------------------------------------------------------
def run_policy(env: TDTOPTWEnv, model) -> list:
    """
    Runs the trained DQN policy in the environment until termination.
    Returns a list of RouteStep objects.
    """
    import torch

    state = env.reset()
    route = []
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(state_t).squeeze(0).cpu().numpy()

        feas = env._feasible_actions_mask()
        if not feas.any():
            break

        q_vals[~feas] = -1e9
        action_idx = int(np.argmax(q_vals))

        prev_poi = env.current_poi
        prev_day = env.current_day
        prev_minute = env.current_minute

        next_state, reward, done, info = env.step(action_idx)

        next_poi = env.current_poi
        tt = env.travel_time[prev_poi, next_poi]

        # Build route step (for the POI we just arrived at)
        step = RouteStep(
            poi_idx=next_poi,
            poi_name="",  # fill if needed
            day=env.current_day,
            arrival_minute=int(env.current_minute - env.visit_durations[next_poi]),
            departure_minute=int(env.current_minute),
            category=env.main_categories[next_poi],
            is_accommodation=bool(env.is_accommodation[next_poi]),
            travel_time=float(tt),
            visit_duration=float(env.visit_durations[next_poi]),
            interest_score=float(env.poi_scores[next_poi]),
        )

        route.append(step)
        state = next_state

    return route


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------
def generate_itinerary(
    model_name: str,
    start_poi: int,
    start_day: int,
    num_days: int,
    config_path: str,
) -> Dict[str, Any]:
    """
    High-level function that:
    - loads the model
    - loads the environment
    - runs the policy
    - validates the route
    - scores the route
    """
    model = load_model(model_name)
    env = load_env(config_path)

    # Override dynamic parameters
    env.start_poi = start_poi
    env.start_day = start_day
    env.num_days = num_days

    route = run_policy(env, model)
    validation = validate_route(
        route,
        env.opening_mask,
        env.travel_time,
    )

    score = score_route(route)

    return {
        "route": route,
        "validation": validation,
        "score": score,
    }
