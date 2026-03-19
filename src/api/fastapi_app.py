# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import mlflow
import torch
from pathlib import Path
import yaml
import geopandas as gpd

from src.model_training.train_dqn import load_env_train
from src.model_training.qnet import QNet
from src.inference.generate_itinerary import generate_route, RouteStep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ItineraryRequest(BaseModel):
    start_poi: int
    start_day: int
    num_days: int
    model_name: str = "tdtoptw_dqn"
    config_path: str = "training.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      - connect to MLflow
      - load config
      - load env
      - load POIs
      - load QNet from MLflow
      - cache everything in app.state
    """
    # 1) MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # 2) Resolve config path
    config_path = Path("configs") / "training.yaml"
    config_path = config_path.resolve()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3) Load environment ONCE
    env = load_env_train(str(config_path))

    # 4) Load POIs + metadata ONCE
    pois = gpd.read_parquet(cfg["data"]["pois_geoparquet"]).reset_index(drop=True)
    main_categories = pois["main_category"].to_numpy()
    is_accommodation = pois["categories"].apply(
        lambda s: "accomodation" in s if isinstance(s, str) else False
    ).to_numpy()

    # 5) Build QNet and load weights from MLflow ONCE
    state_dim = env._get_state().shape[0]
    n_actions = env.max_actions
    qnet = QNet(state_dim, n_actions).to(device)

    model_name = "tdtoptw_dqn"
    model_uri = f"models:/{model_name}/Production"
    mlflow_model = mlflow.pytorch.load_model(model_uri)
    qnet.load_state_dict(mlflow_model.state_dict())
    qnet.eval()

    # 6) Cache everything
    app.state.env = env
    app.state.pois = pois
    app.state.main_categories = main_categories
    app.state.is_accommodation = is_accommodation
    app.state.qnet = qnet
    app.state.cfg = cfg

    yield

    # (Optional) teardown logic here
    # e.g., close DB connections, etc.


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/itinerary")
def itinerary(req: ItineraryRequest):
    try:
        # Get cached objects
        env_template = app.state.env
        pois = app.state.pois
        main_categories = app.state.main_categories
        is_accommodation = app.state.is_accommodation
        qnet = app.state.qnet

        # IMPORTANT:
        # env is stateful; we need a fresh copy per request.
        # If your Env class has a proper clone/copy method, use it.
        # Otherwise, reload a LIGHT version here.
        env = env_template.copy() if hasattr(env_template, "copy") else load_env_train(
            str(Path("configs") / "training.yaml")
        )

        # Configure env for this request
        env.start_poi_idx = req.start_poi
        env.start_poi = req.start_poi
        env.start_day = req.start_day
        env.num_days = req.num_days
        env.total_time_budget = req.num_days * (
            env.day_end_minute - env.day_start_minute
        )

        # Run greedy route generation (fast)
        route_steps = generate_route(
            env=env,
            qnet=qnet,
            pois=pois,
            main_categories=main_categories,
            is_accommodation=is_accommodation,
        )

        return [step.to_dict() for step in route_steps]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
