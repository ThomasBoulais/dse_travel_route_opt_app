# **README — Training and Evaluating the TDTOPTW DQN Agent**

This module contains all components required to train, evaluate, and version a Deep Q‑Network (DQN) agent for the **Time‑Dependent Tourist Orienteering Problem with Time Windows (TDTOPTW)**.  
It integrates:

- a custom multi‑day RL environment with opening hours and accommodation constraints,
- a Q(s,a) neural network,
- a full DQN training loop,
- MLflow experiment tracking,
- route generation for evaluation.

---

## **1. Project Structure**

```
travel_route_optimization/
└── model_training/
    ├── config.yaml          # hyperparameters and environment settings
    ├── env_tdtoptw.py       # RL environment (multi-day, time windows)
    ├── qnet.py              # Q(s,a) neural network
    ├── train_dqn.py         # training script with MLflow logging
    └── eval_route.py        # route generation and evaluation
```

The training code relies on data prepared in `data_pipeline/` (POIs, opening masks, drive‑time graph).

---

## **2. Installation and Requirements**

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure MLflow is installed:

```bash
pip install mlflow
```

Start a local MLflow UI (optional but recommended):

```bash
mlflow ui
```

This opens the tracking dashboard at:

```
http://localhost:5000
```

---

## **3. Configuration (`config.yaml`)**

All hyperparameters and environment settings are centralized in `config.yaml`.  
You can adjust:

- environment parameters (start POI, day start/end, number of days),
- DQN hyperparameters (learning rate, epsilon schedule, batch size),
- reward shaping coefficients,
- MLflow experiment/run names.

Example:

```yaml
experiment_name: "tdtoptw_dqn"
run_name: "qsa_multiday_v1"

env:
  start_poi_idx: 0
  start_day: 0
  day_start_minute: 540
  day_end_minute: 1260
  num_days: 3
  max_steps: 150

dqn:
  hidden_dim: 256
  gamma: 0.99
  lr: 0.001
  batch_size: 64
  replay_size: 100000
  num_episodes: 4000
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 3000

reward:
  travel_penalty: -0.02
  distance_penalty: -1.0
  diversity_bonus: 5.0
  time_usage_bonus: 0.05
  step_bonus: 0.5
  invalid_penalty: -5.0
  night_penalty: -5.0
```

---

## **4. Training the Agent**

Run:

```bash
python -m travel_route_optimization.model_training.train_dqn
```

During training, MLflow logs:

- loss curves,
- epsilon schedule,
- episode rewards,
- episode lengths,
- all hyperparameters,
- the trained PyTorch model.

Every 50 episodes, the target network is updated and progress is printed.

You can monitor training live in MLflow UI.

---

## **5. Evaluating a Trained Model**

Once training is complete, copy the **run ID** from MLflow UI.

Then run:

```bash
python -m travel_route_optimization.model_training.eval_route --run_id <RUN_ID>
```

This script:

- loads the trained model from MLflow,
- rebuilds the environment,
- generates a greedy route using Q(s,a),
- prints the detailed itinerary,
- saves `route.json`,
- logs the route as an MLflow artifact.

Example output:

```
POI: Musée Fabre (#123) [cultural]
  Arrive:   Day 0 – 09:20
  Travel:   12.3 min
  Visit:    90.0 min
  Depart:   Day 0 – 10:50
```

---

## **6. Understanding the Environment**

The environment (`env_tdtoptw.py`) models:

- **time‑dependent opening hours** (minute‑level),
- **multi‑day planning**,
- **mandatory accommodation at night**,
- **per‑POI visit durations**,
- **travel times from a KNN drive‑time graph**,
- **reward shaping** for interest, diversity, travel cost, and time usage.

Actions correspond to **neighbor slots** in a fixed‑size action space.

---

## **7. Understanding the Model**

`qnet.py` defines a feed‑forward network:

```
state → hidden → hidden → Q-values for all actions
```

The DQN implementation includes:

- replay buffer,
- target network,
- epsilon‑greedy exploration,
- masking of infeasible actions,
- Q(s,a) updates with `max_a' Q(s',a')`.

---

## **8. MLflow Integration**

The training script logs:

- all hyperparameters,
- training metrics,
- the trained model,
- evaluation artifacts (via `eval_route.py`).

You can compare runs, visualize learning curves, and restore any model version.

---

## **9. Typical Workflow**

1. Adjust hyperparameters in `config.yaml`.
2. Train the agent:

   ```bash
   python -m travel_route_optimization.model_training.train_dqn
   ```

3. Inspect results in MLflow UI.
4. Evaluate a specific run:

   ```bash
   python -m travel_route_optimization.model_training.eval_route --run_id <RUN_ID>
   ```

5. Compare routes and iterate on reward shaping or environment constraints.
