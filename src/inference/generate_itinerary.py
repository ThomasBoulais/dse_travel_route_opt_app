# src/inference/generate_itinerary.py
from pathlib import Path
import torch
import numpy as np
import geopandas as gpd
from dataclasses import dataclass

from src.model_training.qnet import QNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_route(env, qnet: QNet, pois, main_categories, is_accommodation, max_steps=150):
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
        try:
            travel_t = int(travel_t) + 1
        except OverflowError:
            continue

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
