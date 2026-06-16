"""
OSM - Pipeline d'ingestion Source => Bronze
"""

import osmnx as ox
import geopandas as gpd
import logging

# from src.utils.config import BRONZE_DRIVE_GRAPHML, BRONZE_WALK_GRAPHML, DEFAULT_CRS, OSM_BRONZE_GEOPARQUET, OSM_PLACE_NAME
from src.common.config_loader import load_config

cfg = load_config()

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

def get_pois(place: str=None, left: float=None, right: float=None, bottom: float=None, top: float=None) -> gpd.GeoDataFrame:
    """Récupère les POIs sous format GeoPandas."""
    # si lieu nommé
    if place is not None:
        log.info(f"Source => Bronze (OSM) : Téléchargement POIs ({cfg.bronze.osm_place_name}).")
        pois_gdf = ox.features_from_place(place, tags=TAGS)
        log.info(f"Source => Bronze (OSM) : POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    # si boundary box
    else:
        log.info(f"Source => Bronze (OSM) : Téléchargement POIs ({left}:{right}, {bottom}:{top}).")
        pois_gdf = ox.features_from_bbox(bbox=(left, bottom, right, top), tags=TAGS)
        log.info(f"Source => Bronze (OSM) : POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    return pois_gdf


def get_road_networks(network_type: str, place: str=None, left: float=None, right: float=None, bottom: float=None, top: float=None) -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sous format GeoPandas."""
    # si lieu nommé
    if place is not None:
        log.info(f"Source => Bronze (OSM) : Téléchargement réseau de routes '{network_type}' ({cfg.bronze.osm_place_name}).")
        G = ox.graph_from_place(place, network_type=network_type)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    # si boundary box
    else:
        log.info(f"Source => Bronze (OSM) : Téléchargement réseau de routes '{network_type}' ({left}:{right}, {bottom}:{top}).")
        G = ox.graph_from_bbox(bbox=(left, bottom, right, top), network_type=network_type)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    return G


def ingest_bronze(pois_gdf: gpd.GeoDataFrame, G_drive: gpd.GeoDataFrame=None, G_walk: gpd.GeoDataFrame=None) -> None:
    """Sauvegarde en GeoParquet (Bronze)."""
    if pois_gdf.crs is None:
        pois_gdf = pois_gdf.set_crs(cfg.crs.default)
    pois_gdf = pois_gdf.to_crs(cfg.crs.default)  # ensure standard WGS84 coordinates

    pois_gdf.to_parquet(cfg.bronze.osm_geoparquet)
    log.info(f"Source => Bronze (OSM) : GeoParquet POIs ({pois_gdf.crs}) sauvegardés à {cfg.bronze.osm_geoparquet}")

    pois_gdf.to_csv(cfg.bronze.osm_csv)
    log.info(f"Source => Bronze (OSM) : CSV POIs ({pois_gdf.crs}) sauvegardés à {cfg.bronze.osm_csv}")

    # pois = gpd.read_parquet(cfg.bronze.osm_geoparquet)
    # from pprint import pprint
    # pprint(pois.crs)

    if G_drive is not None:
        ox.save_graphml(G_drive, filepath=cfg.bronze.drive_graphml)
        log.info(f"Source => Bronze (OSM) : {len(G_drive.nodes)} noeuds (nodes) et {len(G_drive.edges)} arrêtes (edges) dans le réseau de route 'drive'.")
        log.info(f"Source => Bronze (OSM) : Graphml sauvegardés à {cfg.bronze.drive_graphml}")

    if G_walk is not None:
        ox.save_graphml(G_walk, filepath=cfg.bronze.walk_graphml)
        log.info(f"Source => Bronze (OSM) : {len(G_walk.nodes)} noeuds (nodes) et {len(G_walk.edges)} arrêtes (edges) dans le réseau de route 'walk'.")
        log.info(f"Source => Bronze (OSM) : Graphml sauvegardés à {cfg.bronze.walk_graphml}")
