import os
import random
from collections import deque
from typing import Deque, Tuple

import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.model_training.env_tdtoptw import TDTOPTWEnv
from src.model_training.qnet import QNet


# ---------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------
# Load environment for training
# ---------------------------------------------------------
def load_env_train(config_path: str) -> TDTOPTWEnv:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pois_geoparquet = cfg["data"]["pois_geoparquet"]
    knn_graph_path = cfg["data"]["knn_drive_time_graph_df"]

    pois = gpd.read_parquet(pois_geoparquet).reset_index(drop=True)
    knn_df = pd.read_csv(knn_graph_path)

    N = len(pois)
    travel_time = np.full((N, N), np.inf, dtype=np.float32)
    for _, row in knn_df.iterrows():
        i = int(row["poi_from"])
        j = int(row["poi_to"])
        travel_time[i, j] = float(row["drive_time"]) / 60.0

    coords = np.vstack(
        [pois.geometry.y.values, pois.geometry.x.values]
    ).T.astype(np.float32)

    visit_durations = pois["visit_duration"].to_numpy().astype(np.float32)
    poi_scores = pois["interest_score"].to_numpy().astype(np.float32)
    main_categories = pois["main_category"].to_numpy()
    is_accommodation = pois["categories"].apply(
        lambda s: "accomodation" in s if isinstance(s, str) else False
    ).to_numpy()

    opening_mask = np.array(
        [np.array(m).reshape(7, 1440) for m in pois["opening_mask_flat"].values],
        dtype=np.uint8,
    )

    knn_neighbors = [[] for _ in range(N)]
    for _, row in knn_df.iterrows():
        i = int(row["poi_from"])
        j = int(row["poi_to"])
        knn_neighbors[i].append(j)

    poi_features = np.hstack(
        [coords, visit_durations.reshape(-1, 1)]
    ).astype(np.float32)

    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]

    env = TDTOPTWEnv(
        poi_features=poi_features,
        poi_scores=poi_scores,
        opening_mask=opening_mask,
        travel_time=travel_time,
        knn_neighbors=knn_neighbors,
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
# ε-greedy action selection
# ---------------------------------------------------------
def select_action(env: TDTOPTWEnv, qnet: QNet, state: np.ndarray, epsilon: float, device):
    feas = env._feasible_actions_mask()
    if not feas.any():
        return None

    if random.random() < epsilon:
        candidate_actions = np.where(feas)[0]
        unvisited = []
        neighbors = env.knn_neighbors[env.current_poi]
        for a in candidate_actions:
            if a < len(neighbors):
                j = neighbors[a]
                if not env.visited[j]:
                    unvisited.append(a)
        if unvisited:
            return int(random.choice(unvisited))
        return int(random.choice(candidate_actions))

    s_t = torch.from_numpy(state).float().to(device).unsqueeze(0)
    with torch.no_grad():
        q_vals = qnet(s_t).squeeze(0).cpu().numpy()
    q_vals[~feas] = -1e9
    return int(np.argmax(q_vals))


# ---------------------------------------------------------
# Training loop with MLflow
# ---------------------------------------------------------
def train(config_path: str = "configs/training.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg["experiment_name"]
    run_name = cfg["run_name"]

    # Use tracking URI from env if set, otherwise default local store
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log config once at the beginning
        mlflow.log_artifact(config_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = load_env_train(config_path)
        state_dim = env._get_state().shape[0]
        action_dim = env.max_actions

        dqn_cfg = cfg["dqn"]

        hidden_dim = dqn_cfg["hidden_dim"]
        gamma = dqn_cfg["gamma"]
        lr = dqn_cfg["lr"]
        batch_size = dqn_cfg["batch_size"]
        replay_size = dqn_cfg["replay_size"]
        num_episodes = dqn_cfg["num_episodes"]
        epsilon_start = dqn_cfg["epsilon_start"]
        epsilon_end = dqn_cfg["epsilon_end"]
        epsilon_decay_episodes = dqn_cfg["epsilon_decay_episodes"]

        mlflow.log_params(dqn_cfg)

        qnet = QNet(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        target_qnet = QNet(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        target_qnet.load_state_dict(qnet.state_dict())
        target_qnet.eval()

        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        buffer = ReplayBuffer(replay_size)

        epsilon = epsilon_start
        global_step = 0
        grad_clip = 5.0
        target_update_steps = 500
        reward_scale = 10.0
        q_clip = 200.0
        loss_fn = nn.SmoothL1Loss()

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            ep_reward = 0.0
            episode_steps = 0
            last_loss = None

            while not done:
                action = select_action(env, qnet, state, epsilon, device)
                if action is None:
                    done = True
                    break

                next_state, reward, done, info = env.step(action)
                reward = reward / reward_scale

                buffer.push(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward
                global_step += 1
                episode_steps += 1

                if len(buffer) >= batch_size:
                    s, a, r, s2, d = buffer.sample(batch_size)

                    s_t = torch.from_numpy(s).float().to(device)
                    a_t = torch.from_numpy(a).long().to(device)
                    r_t = torch.from_numpy(r).float().to(device)
                    s2_t = torch.from_numpy(s2).float().to(device)
                    d_t = torch.from_numpy(d.astype(np.float32)).float().to(device)

                    q_vals_all = qnet(s_t)
                    q_vals = q_vals_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q_all = target_qnet(s2_t)
                        next_q, _ = next_q_all.max(1)
                        target = r_t + gamma * next_q * (1.0 - d_t)
                        target = torch.clamp(target, -q_clip, q_clip)

                    loss = loss_fn(q_vals, target)
                    last_loss = float(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(qnet.parameters(), grad_clip)
                    optimizer.step()

                    if global_step % target_update_steps == 0:
                        target_qnet.load_state_dict(qnet.state_dict())

            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * (ep + 1) / epsilon_decay_episodes,
            )

            mlflow.log_metric("reward", ep_reward, step=ep)
            mlflow.log_metric("loss", last_loss if last_loss is not None else 0.0, step=ep)
            mlflow.log_metric("steps", episode_steps, step=ep)
            mlflow.log_metric("epsilon", epsilon, step=ep)

            if (ep + 1) % 10 == 0:
                print(
                    f"Episode {ep+1}/{num_episodes} - "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Reward: {ep_reward:.2f} | "
                    f"Steps: {episode_steps} | "
                    f"Loss: {last_loss if last_loss is not None else 'N/A'}"
                )

        os.makedirs("models", exist_ok=True)
        model_path = "tdtoptw_dqn.pt"
        torch.save(qnet.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        mlflow.pytorch.log_model(
            pytorch_model=qnet,
            artifact_path="model"
        )

        print("\nTraining terminé.")
        print("Poids bruts (raw weights) sauvegardés sous tdtoptw_dqn.pt")
        print("MLflow model loggué sous artifacts/model/")
        print("La run peut être enregistrée dans le MLflow Model Registry.")


if __name__ == "__main__":
    train()
