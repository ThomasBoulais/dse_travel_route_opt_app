import os
import json
import yaml
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import sys

from dataclasses import dataclass
from travel_route_optimization.model_training.env_tdtoptw import TDTOPTWEnv
from travel_route_optimization.model_training.qnet import QNet
from travel_route_optimization.model_training.train_dqn import (
    load_config,
    build_data,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://localhost:5000")


def to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj


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



def format_time(day: int, minute: int) -> str:
    h = minute // 60
    m = minute % 60
    return f"Day {day} – {h:02d}:{m:02d}"


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

        current_poi = env.current_poi
        next_poi = env.neighbor_idx[current_poi, action_idx]

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


def main(run_id: str):
    cfg = load_config(
        os.path.join(os.path.dirname(__file__), "config.yaml")
    )

    (
        pois,
        poi_features,
        poi_scores,
        opening_mask,
        travel_time,
        neighbor_idx,
        main_categories,
        is_accommodation,
        visit_durations,
    ) = build_data(cfg)

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

    state_dim = env._get_state().shape[0]
    n_actions = env.max_actions

    model_uri = f"runs:/{run_id}/model"
    qnet = mlflow.pytorch.load_model(model_uri).to(device)
    qnet.eval()

    route = generate_route(env, qnet, pois, main_categories, is_accommodation)

    route_dict = [step.to_dict() for step in route]
    
    with open("route.json", "w") as f:
        json.dump(route_dict, f, indent=2)

    with mlflow.start_run(run_name=f"eval_{run_id}"):
        mlflow.log_param("evaluated_run", run_id)
        mlflow.log_artifact("route.json")



    for step in route:
        print(
            f"POI: {step.poi_name} (#{step.poi_idx}) [{step.category}] "
            f"{'(ACCOM)' if step.is_accommodation else ''}"
        )
        print(f"  Arrive:   {format_time(step.day, step.arrival_minute)}")
        print(f"  Travel:   {step.travel_time:.1f} min")
        print(f"  Visit:    {step.visit_duration:.1f} min")
        print(f"  Depart:   {format_time(step.departure_day, step.departure_minute)}")
        print()


if __name__ == "__main__":
    # replace with a real run_id from MLflow UI => `python -m travel_route_optimization_training.eval_route <RUN ID>`
    example_run_id = sys.argv[1]
    main(example_run_id)
