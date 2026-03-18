
import os
from pathlib import Path
import logging
import sys

import src.data_pipeline.gold.gold as gold

# sys.path.append(f"{Path(__file__).parents[1] / "utils"}")
sys.path.append(f"{Path(__file__).parents[1] / "data_pipeline"}")


import src.data_pipeline.bronze.datatourisme as dt_bronze
import src.data_pipeline.silver.datatourisme as dt_silver
import src.data_pipeline.bronze.osm as osm_bronze
import src.data_pipeline.silver.osm as osm_silver 
# from src.utils.config import bbox_bottom, bbox_left, bbox_right, bbox_top, BRONZE_DIR, DATA_DIR, DT_BRONZE_DIR, OSM_BRONZE_DIR, SILVER_DIR, GOLD_DIR
from src.common.config_loader import load_config

cfg = load_config()


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def compute_bbox(cfg):
    lat = cfg.bbox.lat_center
    lon = cfg.bbox.lon_center
    dlat = cfg.bbox.lat_delta
    dlon = cfg.bbox.lon_delta

    return [
        lat - dlat,
        lon - dlon,
        lat + dlat,
        lon + dlon,
    ]


os.makedirs(cfg.paths.data_dir, exist_ok=True)
os.makedirs(cfg.paths.bronze_dir, exist_ok=True)
# os.makedirs(OSM_BRONZE_DIR, exist_ok=True)
# os.makedirs(DT_BRONZE_DIR, exist_ok=True)
os.makedirs(cfg.paths.silver_dir, exist_ok=True)
os.makedirs(cfg.paths.gold_dir, exist_ok=True)


def main():

    log.info("=== Source => Bronze (Ajout & lecture des fichiers bruts) ======================")
    log.info("Source => Bronze (OSM) : Démarrage pipeline")
    bbox_left, bbox_top, bbox_right, bbox_bottom = compute_bbox(cfg)
    osm_pois_gdf    = osm_bronze.get_pois(bbox_left, bbox_right, bbox_bottom, bbox_top)
    # G_drive         = osm_bronze.get_road_networks("drive") # sans boundary box
    G_drive         = osm_bronze.get_road_networks("drive", bbox_left, bbox_right, bbox_bottom, bbox_top)
    G_walk          = osm_bronze.get_road_networks("walk", bbox_left, bbox_right, bbox_bottom, bbox_top)
    osm_bronze.ingest_bronze(osm_pois_gdf, G_drive, G_walk)
    log.info("Source => Bronze (OSM) : Fin pipeline")

    log.info("Source => Bronze (DATATOURISME) : Démarrage pipeline")
    dt_bronze.get_dump()
    dt_bronze.extract_dump()
    index       = dt_bronze.load_index()
    dt_raw_entries = dt_bronze.ingest_bronze(index)
    log.info("Source => Bronze (DATATOURISME) : Fin pipeline")


    log.info("=== Bronze => Silver (transformation & nettoyage) ==============================")
    log.info("Bronze => Silver (OSM) : Démarrage pipeline")
    osm_pois_gdf = osm_silver.transform_silver(osm_pois_gdf)
    osm_silver.export_silver(osm_pois_gdf, G_drive, G_walk)
    log.info("Bronze => Silver (OSM) : Fin pipeline")

    log.info("Bronze => Silver (DATATOURISME) : Démarrage pipeline")
    dt_pois_gdf = dt_silver.transform_silver(dt_raw_entries, bbox_left, bbox_right, bbox_bottom, bbox_top)
    dt_silver.export_silver(dt_pois_gdf)
    log.info("Bronze => Silver (DATATOURISME) : Fin pipeline")
    

    log.info("=== Silver => Gold (transformation finale pour utilisation) ====================")
    dt_gdf      = gold.dt_transform_gold()
    osm_gdf     = gold.osm_transform_gold()

    log.info("Silver => Gold (MERGE) : Démarrage du pipeline ")
    gold_gdf    = gold.merge_gold(dt_gdf, osm_gdf)
    
    edges_df = gold.create_knn_drive_graph(G_drive, gold_gdf)

    gold.export_gold(gold_gdf, G_drive, edges_df)
    log.info("Silver => Gold (MERGE) : Fin du pipeline")

if __name__ == "__main__":
    main()