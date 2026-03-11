import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim

from typing import List

from travel_route_optimization.data_pipeline.utils.config import (
    GOLD_POIS_GEOPARQUET,
    KNN_DRIVE_TIME_GRAPH_DF,
)

# -------------------------------------------------------------------
# 1. Load POIs and add interest_score
# -------------------------------------------------------------------
pois: gpd.GeoDataFrame = gpd.read_parquet(GOLD_POIS_GEOPARQUET)

INTEREST_SCORE = {
    "leisure & entertainment": 15,
    "cultural, historical & religious events or sites": 20,
    "parks, garden & nature": 14,
    "sportive": 5,
    "restauration": 8,
    "accomodation": 6,
    "transport & mobility": -10,
    "utilitaries": -10,
    "": -1000,
}


def add_interest_score(poi_row: gpd.GeoSeries) -> int:
    best = -1000
    for cat in poi_row["categories"].split("|"):
        best = max(best, INTEREST_SCORE.get(cat, -1000))
    return best


pois["interest_score"] = pois.apply(add_interest_score, axis=1)

# -------------------------------------------------------------------
# 2. Build core RL inputs: features, scores, opening masks
# -------------------------------------------------------------------
poi_ids = pois.index.to_numpy()
N = len(poi_ids)

coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T  # (N, 2)
durations = pois["visit_duration"].to_numpy().reshape(-1, 1)            # (N, 1)
poi_scores = pois["interest_score"].to_numpy().astype(np.float32)       # (N,)

opening_mask = np.array(
    [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
    dtype=np.uint8,
)  # (N, 7, 1440)

poi_features = np.hstack([coords, durations]).astype(np.float32)        # (N, d)
d = poi_features.shape[1]

# -------------------------------------------------------------------
# 3. Build neighbors and travel_time from KNN drive-time graph
# -------------------------------------------------------------------
knn_df = pd.read_csv(KNN_DRIVE_TIME_GRAPH_DF)

# Ensure POI ids are ints and mapping is consistent
pois.index = pois.index.astype(int)
knn_df["poi_from"] = knn_df["poi_from"].astype(int)
knn_df["poi_to"] = knn_df["poi_to"].astype(int)

poi_ids = pois.index.to_numpy()
poi_id_to_idx = {pid: i for i, pid in enumerate(poi_ids)}

knn_df["i_from"] = knn_df["poi_from"].map(poi_id_to_idx)
knn_df["i_to"] = knn_df["poi_to"].map(poi_id_to_idx)

# Drop rows where mapping failed (NaN) and cast to int
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
# 4. Environment with time windows (TDOPTW / TOPTW)
# -------------------------------------------------------------------
class TDOPTWEnv:
    def __init__(
        self,
        poi_features: np.ndarray,
        poi_scores: np.ndarray,
        opening_mask: np.ndarray,   # (N, 7, 1440)
        travel_time: np.ndarray,    # (N, N) in minutes
        neighbors: List[List[int]],
        start_poi_idx: int,
        start_day: int = 0,
        start_minute: int = 9 * 60,  # 09:00
        time_budget_min: int = 11 * 60,
        visit_duration_min: int = 30,
        max_steps: int = 50,
    ):
        self.poi_features = poi_features
        self.poi_scores = poi_scores
        self.opening_mask = opening_mask
        self.travel_time = travel_time
        self.neighbors = neighbors

        self.N = poi_features.shape[0]
        self.d = poi_features.shape[1]

        self.start_poi = start_poi_idx
        self.start_day = start_day
        self.start_minute = start_minute
        self.time_budget_init = time_budget_min
        self.visit_duration = visit_duration_min
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        self.current_poi = self.start_poi
        self.current_day = self.start_day
        self.current_minute = self.start_minute
        self.time_remaining = self.time_budget_init
        self.visited = np.zeros(self.N, dtype=bool)
        self.visited[self.current_poi] = True
        self.steps = 0
        return self._get_state()

    def _time_to_indices(self, delta_min: float):
        total = self.current_day * 1440 + self.current_minute + int(delta_min)
        day = (total // 1440) % 7
        minute = total % 1440
        return day, minute

    def _is_open(self, poi_idx: int, day: int, minute: int) -> bool:
        return self.opening_mask[poi_idx, day, minute] == 1

    def _get_state(self):
        feat = self.poi_features[self.current_poi]
        t_rem = np.array([self.time_remaining / self.time_budget_init], dtype=np.float32)
        day_norm = np.array([self.current_day / 6.0], dtype=np.float32)
        minute_norm = np.array([self.current_minute / 1440.0], dtype=np.float32)
        visited_ratio = np.array([self.visited.sum() / self.N], dtype=np.float32)
        state = np.concatenate([feat, t_rem, day_norm, minute_norm, visited_ratio])
        return state.astype(np.float32)

    def _feasible_actions_mask(self):
        cand = self.neighbors[self.current_poi]
        mask = []
        for j in cand:
            if self.visited[j]:
                mask.append(False)
                continue

            tt = self.travel_time[self.current_poi, j]
            total = tt + self.visit_duration
            if total > self.time_remaining:
                mask.append(False)
                continue

            open_ok = True
            for m in range(self.visit_duration):
                d, mm = self._time_to_indices(tt + m)
                if not self._is_open(j, d, mm):
                    open_ok = False
                    break

            mask.append(open_ok)

        return np.array(mask, dtype=bool)

    def step(self, action_idx: int):
        cand = self.neighbors[self.current_poi]
        if action_idx < 0 or action_idx >= len(cand):
            return self._get_state(), -10.0, True, {"reason": "invalid_action"}

        next_poi = cand[action_idx]
        tt = self.travel_time[self.current_poi, next_poi]
        total = tt + self.visit_duration

        feas_mask = self._feasible_actions_mask()
        if not feas_mask[action_idx]:
            return self._get_state(), -10.0, True, {"reason": "infeasible"}

        self.time_remaining -= total
        self.current_day, self.current_minute = self._time_to_indices(total)
        self.current_poi = next_poi
        self.visited[next_poi] = True
        self.steps += 1

        interest            = self.poi_scores[next_poi]
        # travel_penalty      = -0.1 * tt
        # open_bonus          = 2.0 if self._is_open(next_poi, self.current_day, self.current_minute) else -2.0
        # diversity_bonus     = 1.0 if self._category_changed(next_poi) else 0.0
        # distance_penalty    = -0.1 * self._euclidean_distance(self.current_poi, next_poi)

        reward = interest # + travel_penalty + open_bonus + diversity_bonus + distance_penalty

        done = (self.time_remaining <= 0) or (self.steps >= self.max_steps)
        return self._get_state(), reward, done, {}


# -------------------------------------------------------------------
# 6. Q-network
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
# 7. Setup RL components
# -------------------------------------------------------------------
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TDOPTWEnv(
    poi_features=poi_features,
    poi_scores=poi_scores,
    opening_mask=opening_mask,
    travel_time=travel_time,
    neighbors=neighbors,
    start_poi_idx=0,
    time_budget_min=8 * 60,
    visit_duration_min=30,
    max_steps=30,
)

state_dim = env._get_state().shape[0]
qnet = QNet(state_dim).to(device)
target_qnet = QNet(state_dim).to(device)
target_qnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
gamma = 0.99
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 10_000
replay = deque(maxlen=50_000)
batch_size = 64
global_step = 0


# -------------------------------------------------------------------
# 8. Action selection (epsilon-greedy over neighbors)
# -------------------------------------------------------------------
def select_action(env: TDOPTWEnv, state: np.ndarray, epsilon: float):
    cand = env.neighbors[env.current_poi]
    feas = env._feasible_actions_mask()
    if not feas.any():
        return None

    if random.random() < epsilon:
        return int(random.choice(np.where(feas)[0]))

    q_values = []
    for a_idx, j in enumerate(cand):
        if not feas[a_idx]:
            q_values.append(-1e9)
            continue

        tt = env.travel_time[env.current_poi, j]
        total = tt + env.visit_duration
        day2, min2 = env._time_to_indices(total)
        visited2 = env.visited.copy()
        visited2[j] = True

        feat = env.poi_features[j]
        t_rem2 = env.time_remaining - total
        s2 = np.concatenate(
            [
                feat,
                np.array([t_rem2 / env.time_budget_init], dtype=np.float32),
                np.array([day2 / 6.0], dtype=np.float32),
                np.array([min2 / 1440.0], dtype=np.float32),
                np.array([visited2.sum() / env.N], dtype=np.float32),
            ]
        )

        s2_t = torch.tensor(s2, dtype=torch.float32, device=device).unsqueeze(0)
        q_values.append(qnet(s2_t).item())

    return int(np.argmax(q_values))


# -------------------------------------------------------------------
# 9. Training loop (minimal DQN-style)
# -------------------------------------------------------------------
num_episodes = 500

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

        next_state, reward, done, _ = env.step(action_idx)
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

    if ep % 10 == 0:
        target_qnet.load_state_dict(qnet.state_dict())


# -------------------------------------------------------------------
# 10. Route generation (greedy policy)
# -------------------------------------------------------------------
def generate_route(env: TDOPTWEnv, qnet: QNet, max_steps: int = 30):
    state = env.reset()
    route = [env.current_poi]

    for _ in range(max_steps):
        action_idx = select_action(env, state, epsilon=0.0)
        if action_idx is None:
            break
        next_state, reward, done, _ = env.step(action_idx)
        route.append(env.current_poi)
        state = next_state
        if done:
            break

    return route

# route_idx = generate_route(env, qnet)
# route_pois = pois.iloc[route_idx]["name"].tolist()


# print(route_idx)

# for idx in route_idx:
#     print(pois.iloc[idx])


# --------

from dataclasses import dataclass

@dataclass
class RouteStep:
    poi_idx: int
    poi_name: str
    arrival_day: int
    arrival_minute: int
    travel_time: float
    visit_duration: float
    departure_day: int
    departure_minute: int


def format_time(day: int, minute: int) -> str:
    h = minute // 60
    m = minute % 60
    return f"Day {day} – {h:02d}:{m:02d}"


def generate_detailed_route(env: TDOPTWEnv, qnet: QNet, max_steps=30):
    state = env.reset()
    route_steps = []

    for _ in range(max_steps):
        action_idx = select_action(env, state, epsilon=0.0)
        if action_idx is None:
            break

        # Identify next POI
        current_poi = env.current_poi
        next_poi = env.neighbors[current_poi][action_idx]

        # Travel time and arrival time
        travel_t = env.travel_time[current_poi, next_poi]
        arrival_day, arrival_minute = env._time_to_indices(travel_t)

        # Visit duration
        visit_dur = env.visit_duration

        # Apply transition
        next_state, _, done, _ = env.step(action_idx)

        # Departure time (updated by env.step)
        departure_day = env.current_day
        departure_minute = env.current_minute

        # Log the step
        route_steps.append(
            RouteStep(
                poi_idx=next_poi,
                poi_name=pois.iloc[next_poi]["name"],
                arrival_day=arrival_day,
                arrival_minute=arrival_minute,
                travel_time=travel_t,
                visit_duration=visit_dur,
                departure_day=departure_day,
                departure_minute=departure_minute,
            )
        )

        state = next_state
        if done:
            break

    return route_steps



route = generate_detailed_route(env, qnet)

for step in route:
    print(f"POI: {step.poi_name} (#{step.poi_idx})")
    print(f"  Arrive:   {format_time(step.arrival_day, step.arrival_minute)}")
    print(f"  Travel:   {step.travel_time:.1f} min")
    print(f"  Visit:    {step.visit_duration:.1f} min")
    print(f"  Depart:   {format_time(step.departure_day, step.departure_minute)}")
    print()
