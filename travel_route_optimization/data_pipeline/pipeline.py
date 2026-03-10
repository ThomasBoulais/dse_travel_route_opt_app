
import os
from pathlib import Path
import logging
import sys

import travel_route_optimization.data_pipeline.gold.gold as gold

# sys.path.append(f"{Path(__file__).parents[1] / "utils"}")
sys.path.append(f"{Path(__file__).parents[1] / "data_pipeline"}")


import travel_route_optimization.data_pipeline.bronze.datatourisme as dt_bronze
import travel_route_optimization.data_pipeline.silver.datatourisme as dt_silver
import travel_route_optimization.data_pipeline.bronze.osm as osm_bronze
import travel_route_optimization.data_pipeline.silver.osm as osm_silver 
from travel_route_optimization.data_pipeline.utils.config import BRONZE_DIR, DATA_DIR, DT_BRONZE_DIR, OSM_BRONZE_DIR, SILVER_DIR, GOLD_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(OSM_BRONZE_DIR, exist_ok=True)
os.makedirs(DT_BRONZE_DIR, exist_ok=True)
os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(GOLD_DIR, exist_ok=True)


def main():

    log.info("=== Source => Bronze (Ajout & lecture des fichiers bruts) ======================")
    
    log.info("Source => Bronze (OSM) : Démarrage pipeline")
    pois_gdf    = osm_bronze.get_pois()
    G_drive     = osm_bronze.get_road_networks("drive")
    G_walk      = osm_bronze.get_road_networks("walk")
    osm_bronze.ingest_bronze(pois_gdf, G_drive, G_walk)
    log.info("Source => Bronze (OSM) : Fin pipeline")

    log.info("Source => Bronze (DATATOURISME) : Démarrage pipeline")
    dt_bronze.get_dump()
    dt_bronze.extract_dump()
    index       = dt_bronze.load_index()
    raw_entries = dt_bronze.ingest_bronze(index)
    log.info("Source => Bronze (DATATOURISME) : Fin pipeline")


    log.info("=== Bronze => Silver (transformation & nettoyage) ==============================")
    
    log.info("Bronze => Silver (OSM) : Démarrage pipeline")
    slim_pois_gdf = osm_silver.transform_silver(pois_gdf)
    osm_silver.export_silver(slim_pois_gdf, G_drive, G_walk)
    log.info("Bronze => Silver (OSM) : Fin pipeline")

    log.info("Bronze => Silver (DATATOURISME) : Démarrage pipeline")
    gdf = dt_silver.transform_silver(raw_entries)
    dt_silver.export_silver(gdf)
    log.info("Bronze => Silver (DATATOURISME) : Fin pipeline")
    

    log.info("=== Silver => Gold (transformation finale pour utilisation) ====================")
    dt_gdf      = gold.dt_transform_gold()
    osm_gdf     = gold.osm_transform_gold()

    log.info("Silver => Gold (MERGE) : Démarrage du pipeline ")
    gold_gdf    = gold.merge_gold(dt_gdf, osm_gdf)
    
    gold.export_gold(gold_gdf, G_drive, G_walk)
    log.info("Silver => Gold (MERGE) : Fin du pipeline")

if __name__ == "__main__":
    main()