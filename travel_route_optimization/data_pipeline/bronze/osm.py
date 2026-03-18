"""
OSM - Pipeline d'ingestion Source => Bronze
"""

import osmnx as ox
import geopandas as gpd
import logging

from travel_route_optimization.utils.config import BRONZE_DRIVE_GRAPHML, BRONZE_WALK_GRAPHML, DEFAULT_CRS, OSM_BRONZE_GEOPARQUET, OSM_PLACE_NAME

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

# si lieu nommé
def get_pois() -> gpd.GeoDataFrame:
    """Récupère les POIs sous format GeoPandas."""
    log.info(f"Source => Bronze (OSM) : Téléchargement POIs ({OSM_PLACE_NAME}).")
    pois_gdf = ox.features_from_place(OSM_PLACE_NAME, tags=TAGS)
    log.info(f"Source => Bronze (OSM) : POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    return pois_gdf


# si boundary box
def get_pois(left: float, right: float, bottom: float, top: float) -> gpd.GeoDataFrame:
    """Récupère les POIs sous format GeoPandas."""
    log.info(f"Source => Bronze (OSM) : Téléchargement POIs ({left}:{right}, {bottom}:{top}).")
    pois_gdf = ox.features_from_bbox(bbox=(left, bottom, right, top), tags=TAGS)
    log.info(f"Source => Bronze (OSM) : POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    return pois_gdf


# si lieu nommé
def get_road_networks(network_type: str) -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sous format GeoPandas."""
    log.info(f"Source => Bronze (OSM) : Téléchargement réseau de routes '{network_type}' ({OSM_PLACE_NAME}).")
    G = ox.graph_from_place(OSM_PLACE_NAME, network_type=network_type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


# si boundary box
def get_road_networks(network_type: str, left: float, right: float, bottom: float, top: float) -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sous format GeoPandas."""
    log.info(f"Source => Bronze (OSM) : Téléchargement réseau de routes '{network_type}' ({left}:{right}, {bottom}:{top}).")
    G = ox.graph_from_bbox(bbox=(left, bottom, right, top), network_type=network_type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def ingest_bronze(pois_gdf: gpd.GeoDataFrame, G_drive: gpd.GeoDataFrame, G_walk: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Bronze)."""
    if pois_gdf.crs is None:
        pois_gdf = pois_gdf.set_crs(DEFAULT_CRS)
    pois_gdf = pois_gdf.to_crs(DEFAULT_CRS)  # ensure standard WGS84 coordinates

    pois_gdf.to_parquet(OSM_BRONZE_GEOPARQUET)
    log.info(f"Source => Bronze (OSM) : GeoParquet POIs ({pois_gdf.crs}) sauvegardés à {OSM_BRONZE_GEOPARQUET}")

    pois = gpd.read_parquet(OSM_BRONZE_GEOPARQUET)
    from pprint import pprint
    # pprint(pois.crs)

    ox.save_graphml(G_drive, filepath=BRONZE_DRIVE_GRAPHML)
    log.info(f"Source => Bronze (OSM) : {len(G_drive.nodes)} noeuds (nodes) et {len(G_drive.edges)} arrêtes (edges) dans le réseau de route 'drive'.")
    log.info(f"Source => Bronze (OSM) : Graphml sauvegardés à {BRONZE_DRIVE_GRAPHML}")

    ox.save_graphml(G_walk, filepath=BRONZE_WALK_GRAPHML)
    log.info(f"Source => Bronze (OSM) : {len(G_walk.nodes)} noeuds (nodes) et {len(G_walk.edges)} arrêtes (edges) dans le réseau de route 'walk'.")
    log.info(f"Source => Bronze (OSM) : Graphml sauvegardés à {BRONZE_WALK_GRAPHML}")
