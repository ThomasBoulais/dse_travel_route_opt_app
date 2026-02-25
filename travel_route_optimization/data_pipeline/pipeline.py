
import os
from travel_route_optimization.data_pipeline.utils.config import BRONZE_DIR, SILVER_DIR, GOLD_DIR


os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(GOLD_DIR, exist_ok=True)

# tout à faire une fois le pipeline split dans les bons dossiers pour OSM & DATATourisme