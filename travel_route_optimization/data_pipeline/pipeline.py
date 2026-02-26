
import os
from pathlib import Path
import logging
import sys

# sys.path.append(f"{Path(__file__).parents[1] / "utils"}")
sys.path.append(f"{Path(__file__).parents[1] / "data_pipeline"}")


import data_pipeline.bronze.datatourisme as dt_bronze
import data_pipeline.silver.datatourisme as dt_silver
import data_pipeline.bronze.osm as osm_bronze
import data_pipeline.silver.osm as osm_silver 
from data_pipeline.utils.config import BRONZE_DIR, DATA_DIR, DT_BRONZE_DIR, OSM_BRONZE_DIR, SILVER_DIR, GOLD_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(OSM_BRONZE_DIR, exist_ok=True)
os.makedirs(DT_BRONZE_DIR, exist_ok=True)
os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(GOLD_DIR, exist_ok=True)


def main():
    # === OSM ===
    log.info("OSM - Démarrage pipeline Bronze => Silver")

    # Bronze: ajout & lecture des fichiers bruts 
    pois_gdf    = osm_bronze.get_pois()
    G           = osm_bronze.get_road_networks()
    osm_bronze.ingest_bronze(pois_gdf, G)

    # Silver: transformation  & nettoyage
    slim_pois_gdf = osm_silver.transform_silver(pois_gdf)

    # Export
    osm_silver.export_silver(slim_pois_gdf, G)

    # Aperçu
    log.info("OSM - Pipeline terminé")


    # === DATATOURISME ===
    log.info("DATATOURSIME - Démarrage pipeline DataTourisme Bronze => Silver")

    # Bronze : ajout & lecture des fichiers bruts
    dt_bronze.get_dump()
    dt_bronze.extract_dump()
    index       = dt_bronze.load_index()
    raw_entries = dt_bronze.ingest_bronze(index)

    # Silver : transformation & nettoyage
    gdf = dt_silver.transform_silver(raw_entries)

    # Export
    dt_silver.export_silver(gdf)

    # Aperçu
    log.info("\nDATATOURSIME - Aperçu (5 premières lignes)")
    print(gdf[["name_fr", "types", "city", "latitude", "longitude"]].head())

    log.info("\nDATATOURSIME - Répartition par type (top 10)")
    type_counts = (
        gdf["types"]
        .str.split("|")
        .explode()
        .value_counts()
        .head(10)
    )
    print(type_counts.to_string())

    log.info("DATATOURSIME - Pipeline terminé")


if __name__ == "__main__":
    main()