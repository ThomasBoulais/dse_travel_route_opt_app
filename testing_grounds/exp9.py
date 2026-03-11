import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
import random

from typing import List, Dict, Set
from collections import deque
from dataclasses import dataclass

# -------------------------------------------------------------------
# 0. Config & data loading
# -------------------------------------------------------------------
from travel_route_optimization.data_pipeline.utils.config import (
    GOLD_POIS_GEOPARQUET,
    KNN_DRIVE_TIME_GRAPH_DF,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pois: gpd.GeoDataFrame = gpd.read_parquet(GOLD_POIS_GEOPARQUET)

# -------------------------------------------------------------------
# 1. Interest score & category preprocessing
# -------------------------------------------------------------------
INTEREST_SCORE = {
    "leisure & entertainment": 150,
    "cultural, historical & religious events or sites": 200,
    "parks, garden & nature": 140,
    "sportive": 50,
    "restauration": 80,
    "accomodation": 60,
    "transport & mobility": -100,
    "utilitaries": -100,
    "": -1000,
}


def extract_categories(cat_str: str) -> List[str]:
    if not isinstance(cat_str, str) or not cat_str.strip():
        return [""]
    return [c.strip() for c in cat_str.split("|") if c.strip()]


def add_interest_score(poi_row: gpd.GeoSeries) -> int:
    cats = extract_categories(poi_row["categories"])
    best = -1000
    for cat in cats:
        best = max(best, INTEREST_SCORE.get(cat, -1000))
    return best


pois.index = pois.index.astype(int)
poi_ids = pois.index.to_numpy()
N = len(poi_ids)

pois["interest_score"] = pois.apply(add_interest_score, axis=1)
pois["main_category"] = pois["categories"].apply(
    lambda s: extract_categories(s)[0] if extract_categories(s) else ""
)

is_accommodation = pois["categories"].apply(
    lambda s: any(c.strip() == "accomodation" for c in extract_categories(s))
).to_numpy()

# -------------------------------------------------------------------
# 2. Core RL inputs: features, scores, opening masks, durations
# -------------------------------------------------------------------
coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T  # (N, 2)
visit_durations = pois["visit_duration"].to_numpy().astype(np.float32)  # (N,)
durations_feat = visit_durations.reshape(-1, 1)

poi_scores = pois["interest_score"].to_numpy().astype(np.float32)
main_categories = pois["main_category"].to_numpy()

opening_mask = np.array(
    [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
    dtype=np.uint8,
)  # (N, 7, 1440)

poi_features = np.hstack([coords, durations_feat]).astype(np.float32)  # (N, d)
d = poi_features.shape[1]

# -------------------------------------------------------------------
# 3. Neighbors & travel_time from KNN drive-time graph
# -------------------------------------------------------------------
knn_df = pd.read_csv(KNN_DRIVE_TIME_GRAPH_DF)

knn_df["poi_from"] = knn_df["poi_from"].astype(int)
knn_df["poi_to"] = knn_df["poi_to"].astype(int)

poi_id_to_idx = {pid: i for i, pid in enumerate(poi_ids)}
knn_df["i_from"] = knn_df["poi_from"].map(poi_id_to_idx)
knn_df["i_to"] = knn_df["poi_to"].map(poi_id_to_idx)

knn_df = knn_df.dropna(subset=["i_from", "i_to"])
knn_df["i_from"] = knn_df["i_from"].astype(int)
knn_df["i_to"] = knn_df["i_to"].astype(int)

neighbors: List[List[int]] = [[] for _ in range(N)]
travel_time = np.full((N, N), np.inf, dtype=np.float32)  # minutes

for _, row in knn_df.iterrows():
    i = int(row["i_from"])
    j = int(row["i_to"])
    tt_min = float(row["drive_time"]) / 60.0
    neighbors[i].append(j)
    travel_time[i, j] = tt_min

# -------------------------------------------------------------------
# 4. Environment: multi-day TDTOPTW with night accommodation
# -------------------------------------------------------------------
class TDTOPTWEnv:
    """
    Multi-day Time-Dependent Orienteering with Time Windows:
    - Multiple days horizon
    - Daily active window [day_start_minute, day_end_minute]
    - Must be at an accommodation POI if ending the day after day_end_minute
    """

    def __init__(
        self,
        poi_features: np.ndarray,
        poi_scores: np.ndarray,
        opening_mask: np.ndarray,   # (N, 7, 1440)
        travel_time: np.ndarray,    # (N, N) in minutes
        neighbors: List[List[int]],
        main_categories: np.ndarray,
        is_accommodation: np.ndarray,
        visit_durations: np.ndarray,
        start_poi_idx: int,
        start_day: int = 0,
        day_start_minute: int = 9 * 60,   # 09:00
        day_end_minute: int = 21 * 60,    # 21:00
        num_days: int = 3,
        max_steps: int = 150,
    ):
        self.poi_features = poi_features
        self.poi_scores = poi_scores
        self.opening_mask = opening_mask
        self.travel_time = travel_time
        self.neighbors = neighbors
        self.main_categories = main_categories
        self.is_accommodation = is_accommodation.astype(bool)
        self.visit_durations = visit_durations.astype(np.float32)

        self.N = poi_features.shape[0]
        self.d = poi_features.shape[1]

        self.start_poi = start_poi_idx
        self.start_day = start_day
        self.day_start_minute = day_start_minute
        self.day_end_minute = day_end_minute
        self.num_days = num_days
        self.max_steps = max_steps

        self.total_time_budget = num_days * (day_end_minute - day_start_minute)

        self.reset()

    # ----------------- Time helpers -----------------
    def _time_to_indices(self, delta_min: float):
        total = self.current_day * 1440 + self.current_minute + int(delta_min)
        day = (total // 1440) % 7
        minute = total % 1440
        return day, minute

    def _is_open(self, poi_idx: int, day: int, minute: int) -> bool:
        return self.opening_mask[poi_idx, day, minute] == 1

    def _day_index_from_start(self, day: int) -> int:
        return (day - self.start_day) % 7

    # ----------------- RL API -----------------
    def reset(self):
        self.current_poi = self.start_poi
        self.current_day = self.start_day
        self.current_minute = self.day_start_minute
        self.steps = 0

        self.visited = np.zeros(self.N, dtype=bool)
        self.visited[self.current_poi] = True

        self.used_time = 0.0

        self.visited_categories: Set[str] = set()
        self.visited_categories.add(self.main_categories[self.current_poi])

        return self._get_state()

    def _get_state(self):
        feat = self.poi_features[self.current_poi]
        time_ratio = np.array(
            [self.used_time / max(self.total_time_budget, 1e-6)], dtype=np.float32
        )
        day_norm = np.array([self.current_day / 6.0], dtype=np.float32)
        minute_norm = np.array([self.current_minute / 1440.0], dtype=np.float32)
        visited_ratio = np.array([self.visited.sum() / self.N], dtype=np.float32)
        diversity_ratio = np.array(
            [len(self.visited_categories) / max(self.N, 1)], dtype=np.float32
        )

        state = np.concatenate(
            [feat, time_ratio, day_norm, minute_norm, visited_ratio, diversity_ratio]
        )
        return state.astype(np.float32)

    def _feasible_actions_mask(self):
        cand = self.neighbors[self.current_poi]
        mask = []
        for j in cand:
            if self.visited[j]:
                mask.append(False)
                continue

            tt = self.travel_time[self.current_poi, j]
            visit_dur = self.visit_durations[j]
            total = tt + visit_dur

            day_arr, min_arr = self._time_to_indices(tt)
            day_dep, min_dep = self._time_to_indices(total)
            day_idx = self._day_index_from_start(day_dep)

            if day_idx >= self.num_days:
                mask.append(False)
                continue

            open_ok = True
            for m in range(int(visit_dur)):
                d, mm = self._time_to_indices(tt + m)
                if not self._is_open(j, d, mm):
                    open_ok = False
                    break
            if not open_ok:
                mask.append(False)
                continue

            if min_dep > self.day_end_minute and not self.is_accommodation[j]:
                mask.append(False)
                continue

            mask.append(True)

        return np.array(mask, dtype=bool)

    def step(self, action_idx: int):
        cand = self.neighbors[self.current_poi]
        if action_idx is None or action_idx < 0 or action_idx >= len(cand):
            return self._get_state(), -50.0, True, {"reason": "invalid_action"}

        feas_mask = self._feasible_actions_mask()
        if not feas_mask[action_idx]:
            return self._get_state(), -50.0, True, {"reason": "infeasible"}

        prev_poi = self.current_poi
        next_poi = cand[action_idx]
        tt = self.travel_time[prev_poi, next_poi]
        visit_dur = self.visit_durations[next_poi]
        total = tt + visit_dur

        day_dep, min_dep = self._time_to_indices(total)
        day_idx = self._day_index_from_start(day_dep)

        self.used_time += total
        self.current_day, self.current_minute = day_dep, min_dep
        self.current_poi = next_poi
        self.visited[next_poi] = True
        self.steps += 1

        cat = self.main_categories[next_poi]
        new_category = cat not in self.visited_categories
        if new_category:
            self.visited_categories.add(cat)

        interest = float(self.poi_scores[next_poi])

        travel_penalty = -0.05 * tt

        prev_coord = self.poi_features[prev_poi][0:2]
        next_coord = self.poi_features[next_poi][0:2]
        dist = float(np.linalg.norm(prev_coord - next_coord))
        distance_penalty = -2.0 if dist < 0.001 else 0.0  # tune threshold

        diversity_bonus = 10.0 if new_category else 0.0

        time_usage_bonus = 0.5 * total

        step_bonus = +1

        reward = (
            interest
            + travel_penalty
            + distance_penalty
            + diversity_bonus
            + time_usage_bonus
            + step_bonus
        )

        done = False
        info: Dict = {}

        if day_idx >= self.num_days:
            done = True
            info["reason"] = "horizon_exceeded"

        if self.current_minute > self.day_end_minute and not self.is_accommodation[self.current_poi]:
            reward -= 50.0
            done = True
            info["reason"] = "night_without_accommodation"

        if self.steps >= self.max_steps:
            done = True
            info.setdefault("reason", "max_steps")

        return self._get_state(), reward, done, info


# -------------------------------------------------------------------
# 5. Q-network (state-value approximation)
# -------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_batch):
        return self.net(state_batch).squeeze(-1)


# -------------------------------------------------------------------
# 6. Env instantiation
# -------------------------------------------------------------------
env = TDTOPTWEnv(
    poi_features=poi_features,
    poi_scores=poi_scores,
    opening_mask=opening_mask,
    travel_time=travel_time,
    neighbors=neighbors,
    main_categories=main_categories,
    is_accommodation=is_accommodation,
    visit_durations=visit_durations,
    start_poi_idx=0,
    start_day=0,
    day_start_minute=9 * 60,
    day_end_minute=21 * 60,
    num_days=3,
    max_steps=150,
)

state_dim = env._get_state().shape[0]
qnet = QNet(state_dim).to(device)
target_qnet = QNet(state_dim).to(device)
target_qnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
gamma = 0.99
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 3000
replay = deque(maxlen=100_000)
batch_size = 64
global_step = 0

# -------------------------------------------------------------------
# 7. Action selection (epsilon-greedy over neighbors)
# -------------------------------------------------------------------
def select_action(env: TDTOPTWEnv, state: np.ndarray, epsilon: float):
    cand = env.neighbors[env.current_poi]
    feas = env._feasible_actions_mask()
    if len(cand) == 0 or not feas.any():
        return None

    if random.random() < epsilon:
        feasible_indices = np.where(feas)[0]
        return int(random.choice(feasible_indices))

    q_values = []
    for a_idx, j in enumerate(cand):
        if not feas[a_idx]:
            q_values.append(-1e9)
            continue

        tt = env.travel_time[env.current_poi, j]
        visit_dur = env.visit_durations[j]
        total = tt + visit_dur
        day2, min2 = env._time_to_indices(total)

        visited2 = env.visited.copy()
        visited2[j] = True
        visited_categories2 = set(env.visited_categories)
        cat_j = env.main_categories[j]
        visited_categories2.add(cat_j)

        used_time2 = env.used_time + total
        time_ratio2 = used_time2 / max(env.total_time_budget, 1e-6)
        day_norm2 = day2 / 6.0
        minute_norm2 = min2 / 1440.0
        visited_ratio2 = visited2.sum() / env.N
        diversity_ratio2 = len(visited_categories2) / max(env.N, 1)

        feat_j = env.poi_features[j]
        s2 = np.concatenate(
            [
                feat_j,
                np.array([time_ratio2], dtype=np.float32),
                np.array([day_norm2], dtype=np.float32),
                np.array([minute_norm2], dtype=np.float32),
                np.array([visited_ratio2], dtype=np.float32),
                np.array([diversity_ratio2], dtype=np.float32),
            ]
        )

        s2_t = torch.tensor(s2, dtype=torch.float32, device=device).unsqueeze(0)
        q_values.append(qnet(s2_t).item())

    return int(np.argmax(q_values))


# -------------------------------------------------------------------
# 8. Training loop (DQN-style on state values)
# -------------------------------------------------------------------
num_episodes = 8000
loss = torch.tensor(0.0, device=device)

for ep in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
            -global_step / epsilon_decay
        )

        action_idx = select_action(env, state, epsilon)
        if action_idx is None:
            break

        next_state, reward, done, info = env.step(action_idx)
        replay.append((state, action_idx, reward, next_state, done))
        state = next_state
        global_step += 1

        if len(replay) < batch_size:
            continue

        batch = random.sample(replay, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_qs = []
            for ns in next_states:
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0)
                next_qs.append(target_qnet(ns_t).item())
            next_qs_t = torch.tensor(next_qs, dtype=torch.float32, device=device)

        targets = rewards_t + gamma * (1 - dones_t) * next_qs_t
        q_pred = qnet(states_t)
        loss = ((q_pred - targets) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % 20 == 0:
        target_qnet.load_state_dict(qnet.state_dict())
        print(f"Episode {ep} - epsilon={epsilon:.3f} - last_loss={loss.item():.4f}")


# -------------------------------------------------------------------
# 9. Route generation (greedy policy, multi-day)
# -------------------------------------------------------------------
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


def format_time(day: int, minute: int) -> str:
    h = minute // 60
    m = minute % 60
    return f"Day {day} – {h:02d}:{m:02d}"


def generate_detailed_route(env: TDTOPTWEnv, qnet: QNet, max_steps=150):
    state = env.reset()
    route_steps: List[RouteStep] = []

    for _ in range(max_steps):
        action_idx = select_action(env, state, epsilon=0.0)
        if action_idx is None:
            break

        current_poi = env.current_poi
        next_poi = env.neighbors[current_poi][action_idx]

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
                travel_time=travel_t,
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


route = generate_detailed_route(env, qnet)

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
