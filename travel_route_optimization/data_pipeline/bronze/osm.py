"""
OSM - Pipeline d'ingestion Source => Bronze
"""

import osmnx as ox
import geopandas as gpd
import logging

from travel_route_optimization.data_pipeline.utils.config import OSM_BRONZE_GEOPARQUET, OSM_BRONZE_GRAPHML, OSM_PLACE_NAME, OSM_SILVER_GEOPARQUET, OSM_SILVER_GRAPHML
from travel_route_optimization.data_pipeline.utils.pipeline_helpers import print_len_col_head

ox.settings.use_cache = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# CONFIG - Définition de la zone d'étude

TAGS = {
    "tourism": ["museum", "attraction", "viewpoint", "hotel", "hostel"],
    "amenity": ["restaurant", "cafe", "bar"],
    "leisure": ["park", "nature_reserve"]
}


# BRONZE

def get_pois() -> gpd.GeoDataFrame:
    """Récupère les POIs sous format GeoPandas."""
    log.info(f"OSM - Bronze : Téléchargement POIs ({OSM_PLACE_NAME}).")
    pois_gdf = ox.features_from_place(OSM_PLACE_NAME, tags=TAGS)
    print_len_col_head(pois_gdf)
    return pois_gdf


def get_road_networks() -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sour format GeoPandas."""
    log.info(f"OSM - Bronze : Téléchargement réseau de routes ({OSM_PLACE_NAME}).")
    G = ox.graph_from_place(OSM_PLACE_NAME, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def ingest_bronze(pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Bronze)."""
    pois_gdf = pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    pois_gdf.to_parquet(OSM_BRONZE_GEOPARQUET)
    log.info("OSM - Bronze : GeoParquet POIs sauvegardés à data/bronze/osm/osm_pois.geoparquet")

    ox.save_graphml(G, filepath=OSM_BRONZE_GRAPHML)
    log.info(f"OSM - Bronze : {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges) dans le réseau de route.")
    log.info(f"OSM - Bronze : Graphml sauvegardés à {OSM_BRONZE_GRAPHML}")
