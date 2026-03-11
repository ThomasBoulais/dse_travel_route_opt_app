# OBJECTIF : affecter interest_score, category, visit_duration pour chaque POI

# 1. category
# already done :)

# 2. visit_duration
# Fixer la durée des visites à sommer aux temps de trajet en secondes pour obtenir le poids de chaque arrête 
# already done :)

# 3. interest_score
# A partir des category, définir un score d'intérêt (aka profit) à maximiser lors de l'entrainement
# - pouvoir donner des notes générales par categorie
# - comment gérer les multiples categories de plusieurs types ? prendre la note de la meilleure
# (- pouvoir donner du détail sur les sous-catégories)

# 4. véirifer que toutes les données sont prises en compte et sous le bon format pour lancer le training:
# - pois_feature (np.array) (N,d) -> pois[['opening_mask', 'categories']]
# - pois_score (np.array) (N,) -> pois['interest_score']
# - travel_time (sparse matrix) -> knn_drive_time_graph.csv
# t

import pandas as pd
import geopandas as gpd
from pprint import pprint
import re

from travel_route_optimization.data_pipeline.utils.config import GOLD_POIS_GEOPARQUET, KNN_DRIVE_TIME_GRAPH_DF

pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)

# 2. interest_score
dict_interest_score = {
    'leisure & entertainment'                           : 8,
    'cultural, historical & religious events or sites'  : 10,
    'parks, garden & nature'                            : 7,
    'sportive'                                          : 5,
    'restauration'                                      : 6,
    'accomodation'                                      : 4,
    'transport & mobility'                              : 0,
    'utilitaries'                                       : 0,
    ''                                                  : 0
}


def add_interest_score_score(poi: gpd.GeoSeries, dict_interest_score: dict) -> int:
    """Ajoute la valeur la plus élevée d'intérêt par catégorie pour chaque POI"""
    m_interest_score = 0
    for cat in poi['categories'].split("|"):
        if dict_interest_score[cat] > m_interest_score:
            m_interest_score = dict_interest_score[cat]
    return m_interest_score


pois['interest_score'] = pois.apply(add_interest_score_score, args=(dict_interest_score,), axis=1)

# print(pois.loc[0])

knn_drive_time_graph_df = pd.read_csv(KNN_DRIVE_TIME_GRAPH_DF)

print(knn_drive_time_graph_df.head())

# ------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from typing import List, Dict


# Basic numeric features: visit_duration (in minutes), maybe lat/lon
poi_ids = pois.index.to_numpy()
N = len(poi_ids)

coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T  # (N, 2)
durations = pois["visit_duration"].to_numpy().reshape(-1, 1)            # (N, 1)

# Simple score: you can define your own (e.g. popularity, rating, etc.)
poi_scores = np.ones(N, dtype=np.float32)  # placeholder

# Opening mask: (N, 7, 1440)
opening_mask = np.array(
    [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
    dtype=np.uint8
)

# Final feature vector per POI (you can enrich later)
poi_features = np.hstack([coords, durations]).astype(np.float32)  # shape (N, d)
d = poi_features.shape[1]


knn_df = knn_drive_time_graph_df.copy()

# Map POI ids to index 0..N-1
poi_id_to_idx = {pid: i for i, pid in enumerate(poi_ids)}

knn_df["i_from"] = knn_df["poi_from"].map(poi_id_to_idx)
knn_df["i_to"]   = knn_df["poi_to"].map(poi_id_to_idx)

# Build neighbors list and travel_time matrix (in minutes)
neighbors: List[List[int]] = [[] for _ in range(N)]
travel_time = np.full((N, N), np.inf, dtype=np.float32)

for _, row in knn_df.iterrows():
    i = row["i_from"]
    j = row["i_to"]
    tt_min = row["drive_time"] / 60.0
    neighbors[i].append(j)
    travel_time[i, j] = tt_min

class TDOPTWEnv:
    def __init__(self,
                 poi_features: np.ndarray,
                 poi_scores: np.ndarray,
                 opening_mask: np.ndarray,   # (N, 7, 1440)
                 travel_time: np.ndarray,    # (N, N) in minutes
                 neighbors: List[List[int]],
                 start_poi_idx: int,
                 start_day: int = 0,
                 start_minute: int = 9*60,   # 09:00
                 time_budget_min: int = 8*60,
                 visit_duration_min: int = 30,
                 max_steps: int = 50):

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
        """Advance time by delta_min, return new (day, minute)."""
        total = self.current_day * 1440 + self.current_minute + int(delta_min)
        day = (total // 1440) % 7
        minute = total % 1440
        return day, minute

    def _is_open(self, poi_idx: int, day: int, minute: int) -> bool:
        return self.opening_mask[poi_idx, day, minute] == 1

    def _get_state(self):
        feat = self.poi_features[self.current_poi]  # (d,)
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

            # Check opening at arrival + during visit
            day_arr, min_arr = self._time_to_indices(tt)
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

        # Apply transition
        self.time_remaining -= total
        self.current_day, self.current_minute = self._time_to_indices(total)
        self.current_poi = next_poi
        self.visited[next_poi] = True
        self.steps += 1

        reward = float(self.poi_scores[next_poi])
        done = (self.time_remaining <= 0) or (self.steps >= self.max_steps)
        return self._get_state(), reward, done, {}


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-value for one action
        )

    def forward(self, state_batch, action_mask_batch):
        """
        state_batch: (B, state_dim)
        action_mask_batch: list of boolean masks per sample (variable K)
        We’ll handle actions outside the net (per-action evaluation).
        """
        return self.net(state_batch).squeeze(-1)

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
    time_budget_min=8*60,
    visit_duration_min=30,
    max_steps=30
)

state_dim = env._get_state().shape[0]
qnet = QNet(state_dim).to(device)
target_qnet = QNet(state_dim).to(device)
target_qnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
gamma = 0.99
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 10000
replay = deque(maxlen=50000)
batch_size = 64
global_step = 0


def select_action(env, state, epsilon):
    cand = env.neighbors[env.current_poi]
    feas = env._feasible_actions_mask()
    if not feas.any():
        return None  # no feasible move

    # epsilon-greedy
    if random.random() < epsilon:
        idxs = np.where(feas)[0]
        return int(random.choice(idxs))

    # greedy: evaluate Q(next_state) for each feasible action
    q_values = []
    for a_idx, j in enumerate(cand):
        if not feas[a_idx]:
            q_values.append(-1e9)
            continue
        # simulate transition (without modifying env)
        tt = env.travel_time[env.current_poi, j]
        total = tt + env.visit_duration
        day2, min2 = env._time_to_indices(total)
        visited2 = env.visited.copy()
        visited2[j] = True

        feat = env.poi_features[j]
        t_rem2 = env.time_remaining - total
        t_rem_norm = np.array([t_rem2 / env.time_budget_init], dtype=np.float32)
        day_norm = np.array([day2 / 6.0], dtype=np.float32)
        min_norm = np.array([min2 / 1440.0], dtype=np.float32)
        vis_ratio = np.array([visited2.sum() / env.N], dtype=np.float32)
        s2 = np.concatenate([feat, t_rem_norm, day_norm, min_norm, vis_ratio])

        s2_t = torch.tensor(s2, dtype=torch.float32, device=device).unsqueeze(0)
        q = qnet(s2_t, None).item()
        q_values.append(q)

    return int(np.argmax(q_values))


num_episodes = 500

for ep in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1.0 * global_step / epsilon_decay)

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

        # For simplicity, approximate target as max Q(next_state) over a random feasible action
        with torch.no_grad():
            next_qs = []
            for ns in next_states:
                # crude: just evaluate Q(ns) as a state value
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0)
                qv = target_qnet(ns_t, None).item()
                next_qs.append(qv)
            next_qs_t = torch.tensor(next_qs, dtype=torch.float32, device=device)

        targets = rewards_t + gamma * (1 - dones_t) * next_qs_t

        q_pred = qnet(states_t, None)
        loss = ((q_pred - targets) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Periodically update target network
    if ep % 10 == 0:
        target_qnet.load_state_dict(qnet.state_dict())


def generate_route(env: TDOPTWEnv, qnet: QNet, max_steps=30):
    state = env.reset()
    route = [env.current_poi]

    for _ in range(max_steps):
        action_idx = select_action(env, state, epsilon=0.0)  # greedy
        if action_idx is None:
            break
        next_state, reward, done, info = env.step(action_idx)
        route.append(env.current_poi)
        state = next_state
        if done:
            break

    return route

route_idx = generate_route(env, qnet)
route_pois = pois.iloc[route_idx]["name"].tolist()
print(route_pois)
