from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.generate_itinerary import generate_itinerary
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI(
    title="Travel Route Optimization API",
    description="Génération d'itinéraires mulit-jours optimisés avec un DQN model entraîné.",
    version="1.0.0"
)

class ItineraryRequest(BaseModel):
    start_poi: int
    start_day: int
    num_days: int
    model_name: str = "tdtoptw_dqn"
    config_path: str = "training.yaml"

@app.post("/itinerary")
def itinerary(request: ItineraryRequest):
    result = generate_itinerary(
        model_name=request.model_name,
        start_poi=request.start_poi,
        start_day=request.start_day,
        num_days=request.num_days,
        config_path=request.config_path,
    )
    return result
