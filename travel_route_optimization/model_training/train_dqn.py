import os
import random
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import mlflow
import mlflow.pytorch

from travel_route_optimization.data_pipeline.utils.config import (
    GOLD_POIS_GEOPARQUET,
    KNN_DRIVE_TIME_GRAPH_DF,
)
from travel_route_optimization.model_training.env_tdtoptw import TDTOPTWEnv
from travel_route_optimization.model_training.qnet import QNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


INTEREST_SCORE = {
    "leisure & entertainment": 0.6,
    "cultural, historical & religious events or sites": 0.8,
    "parks, garden & nature": 0.5,
    "sportive": 0.2,
    "restauration": 0.3,
    "accomodation": 0.25,
    "transport & mobility": -0.4,
    "utilitaries": -0.4,
    "": -1.0,
}


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_categories(cat_str: str):
    if not isinstance(cat_str, str) or not cat_str.strip():
        return [""]
    return [c.strip() for c in cat_str.split("|") if c.strip()]


def build_data(cfg):
    pois: gpd.GeoDataFrame = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    pois.index = pois.index.astype(int)
    poi_ids = pois.index.to_numpy()
    N = len(poi_ids)

    def add_interest_score(row):
        cats = extract_categories(row["categories"])
        best = -1.0
        for c in cats:
            best = max(best, INTEREST_SCORE.get(c, -1.0))
        return best

    pois["interest_score"] = pois.apply(add_interest_score, axis=1)
    pois["main_category"] = pois["categories"].apply(
        lambda s: extract_categories(s)[0] if extract_categories(s) else ""
    )
    is_accommodation = pois["categories"].apply(
        lambda s: any(c.strip() == "accomodation" for c in extract_categories(s))
    ).to_numpy()

    coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T.astype(
        np.float32
    )
    visit_durations = pois["visit_duration"].to_numpy().astype(np.float32)
    durations_feat = visit_durations.reshape(-1, 1)

    poi_scores = pois["interest_score"].to_numpy().astype(np.float32)
    main_categories = pois["main_category"].to_numpy()

    opening_mask = np.array(
        [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
        dtype=np.uint8,
    )

    poi_features = np.hstack([coords, durations_feat]).astype(np.float32)

    knn_df = pd.read_csv(KNN_DRIVE_TIME_GRAPH_DF)
    knn_df["poi_from"] = knn_df["poi_from"].astype(int)
    knn_df["poi_to"] = knn_df["poi_to"].astype(int)

    poi_id_to_idx = {pid: i for i, pid in enumerate(poi_ids)}
    knn_df["i_from"] = knn_df["poi_from"].map(poi_id_to_idx)
    knn_df["i_to"] = knn_df["poi_to"].map(poi_id_to_idx)
    knn_df = knn_df.dropna(subset=["i_from", "i_to"])
    knn_df["i_from"] = knn_df["i_from"].astype(int)
    knn_df["i_to"] = knn_df["i_to"].astype(int)

    neighbors = [[] for _ in range(N)]
    travel_time = np.full((N, N), np.inf, dtype=np.float32)
    for _, row in knn_df.iterrows():
        i = int(row["i_from"])
        j = int(row["i_to"])
        tt_min = float(row["drive_time"]) / 60.0
        neighbors[i].append(j)
        travel_time[i, j] = tt_min

    max_neighbors = max(len(n) for n in neighbors)
    neighbor_idx = np.full((N, max_neighbors), -1, dtype=np.int32)
    for i in range(N):
        neigh = neighbors[i]
        neighbor_idx[i, : len(neigh)] = np.array(neigh, dtype=np.int32)

    return (
        pois,
        poi_features,
        poi_scores,
        opening_mask,
        travel_time,
        neighbor_idx,
        main_categories,
        is_accommodation,
        visit_durations,
    )


def train():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    # Optional: point to your MLflow server or local folder
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params(
            {
                **cfg["env"],
                **cfg["dqn"],
                **{f"reward_{k}": v for k, v in cfg["reward"].items()},
            }
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

        # Scale reward coefficients down for stability
        reward_cfg_stable = {
            "travel_penalty": reward_cfg.get("travel_penalty", -0.02) * 0.1,
            "distance_penalty": reward_cfg.get("distance_penalty", -0.8) * 0.1,
            "diversity_bonus": reward_cfg.get("diversity_bonus", 5.0) * 0.1,
            "time_usage_bonus": reward_cfg.get("time_usage_bonus", 0.1) * 0.1,
            "step_bonus": reward_cfg.get("step_bonus", 0.8) * 0.1,
            "invalid_penalty": reward_cfg.get("invalid_penalty", -5.0) * 0.1,
            "night_penalty": reward_cfg.get("night_penalty", -5.0) * 0.1,
        }

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
            reward_cfg=reward_cfg_stable,
        )

        state_dim = env._get_state().shape[0]
        n_actions = env.max_actions

        dqn_cfg = cfg["dqn"]
        qnet = QNet(state_dim, n_actions, hidden_dim=dqn_cfg["hidden_dim"]).to(device)
        target_qnet = QNet(state_dim, n_actions, hidden_dim=dqn_cfg["hidden_dim"]).to(
            device
        )
        target_qnet.load_state_dict(qnet.state_dict())

        optimizer = optim.Adam(qnet.parameters(), lr=dqn_cfg["lr"])
        gamma = dqn_cfg["gamma"]
        epsilon_start = dqn_cfg["epsilon_start"]   # 1.0
        epsilon_end = dqn_cfg["epsilon_end"]       # 0.05
        epsilon_decay_episodes = dqn_cfg.get("epsilon_decay_episodes", 2500)
        # slightly faster decay
        epsilon_decay = dqn_cfg.get("epsilon_decay", 3000) * 0.7
        batch_size = dqn_cfg["batch_size"]
        replay_size = dqn_cfg["replay_size"]
        num_episodes = dqn_cfg["num_episodes"]

        replay = deque(maxlen=replay_size)
        global_step = 0
        loss = torch.tensor(0.0, device=device)

        def get_epsilon(ep):
            # Linear decay over episodes, not global steps
            frac = min(ep / epsilon_decay_episodes, 1.0)
            return epsilon_start + frac * (epsilon_end - epsilon_start)

        def select_action(state, epsilon):
            feas = env._feasible_actions_mask()
            if not feas.any():
                return None

            if random.random() < epsilon:
                # Prefer actions leading to unvisited POIs
                candidate_actions = np.where(feas)[0]
                unvisited = [
                    a for a in candidate_actions
                    if not env.visited[env.neighbor_idx[env.current_poi, a]]
                ]
                if unvisited:
                    return int(random.choice(unvisited))
                return int(random.choice(candidate_actions))

            s_t = torch.from_numpy(state).float().to(device).unsqueeze(0)
            with torch.no_grad():
                q_vals = qnet(s_t).squeeze(0).cpu().numpy()
            q_vals[~feas] = -1e9
            return int(np.argmax(q_vals))


        target_update_every = 10  # episodes

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0

            epsilon = get_epsilon(ep)

            while not done:
                action_idx = select_action(state, epsilon)
                if action_idx is None:
                    break

                next_state, reward, done, info = env.step(action_idx)
                feas_next = info.get(
                    "feasible_mask_next", np.ones(n_actions, dtype=bool)
                )

                replay.append((state, action_idx, reward, next_state, done, feas_next))
                state = next_state
                global_step += 1
                episode_reward += reward
                episode_steps += 1

                if len(replay) < batch_size:
                    continue

                batch = random.sample(replay, batch_size)
                states, actions, rewards, next_states, dones, feas_nexts = zip(*batch)

                states_np = np.stack(states, axis=0).astype(np.float32)
                next_states_np = np.stack(next_states, axis=0).astype(np.float32)
                feas_next_np = np.stack(feas_nexts, axis=0)

                states_t = torch.from_numpy(states_np).to(device)
                next_states_t = torch.from_numpy(next_states_np).to(device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
                feas_next_t = torch.tensor(feas_next_np, dtype=torch.bool, device=device)

                q_all = qnet(states_t)
                q_pred = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_all = target_qnet(next_states_t)
                    next_q_all[~feas_next_t] = -1e9
                    max_next_q = next_q_all.max(dim=1).values

                targets = rewards_t + gamma * (1 - dones_t) * max_next_q
                targets = torch.clamp(targets, -5.0, 5.0)
                q_pred = torch.clamp(q_pred, -5.0, 5.0)

                loss = nn.functional.mse_loss(q_pred, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
                optimizer.step()

                mlflow.log_metric("loss", loss.item(), step=global_step)
                mlflow.log_metric("epsilon", float(epsilon), step=global_step)

            mlflow.log_metric("episode_reward", episode_reward, step=ep)
            mlflow.log_metric("episode_steps", episode_steps, step=ep)

            if ep % target_update_every == 0:
                target_qnet.load_state_dict(qnet.state_dict())
                print(
                    f"Episode {ep} - eps={epsilon:.3f} "
                    f"reward={episode_reward:.1f} steps={episode_steps} "
                    f"loss={loss.item():.4f}"
                )

        mlflow.pytorch.log_model(qnet, "model")


if __name__ == "__main__":
    train()
