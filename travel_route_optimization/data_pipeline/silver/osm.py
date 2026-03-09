"""
OSM - Transformation Bronze => Silver
"""

import osmnx as ox
import geopandas as gpd
import logging

from travel_route_optimization.data_pipeline.utils.config import DEFAULT_CRS, OSM_SILVER_GEOPARQUET, SILVER_DRIVE_GRAPHML, SILVER_WALK_GRAPHML
from travel_route_optimization.data_pipeline.utils.pipeline_helpers import print_len_col_head

ox.settings.use_cache = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# CONFIG - Définition des champs à conserver

RELEVANT_FIELDS = [
    "geometry",
    "name",
    "tourism",
    "amenity",
    "historic",
    "leisure",
    "natural",
    "opening_hours",
    "website",
    "phone",
    "addr:city",
    "addr:postcode",
    "wheelchair",
    "stars",
    "wikidata",
]


# SILVER

def transform_silver(pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Réduit aux colonnes utiles + convertit les batîments en point unique (centroïde)."""
    log.info("Bronze => Silver (OSM) : Réduction des POIs aux colonnes pertinentes")
    slim_pois_gdf = pois_gdf[RELEVANT_FIELDS].copy()
    slim_pois_gdf["geometry"] = slim_pois_gdf["geometry"].apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    print_len_col_head(slim_pois_gdf)
    return slim_pois_gdf


def export_silver(slim_pois_gdf: gpd.GeoDataFrame, G_drive: gpd.GeoDataFrame, G_walk: gpd.GeoDataFrame) -> None:
    """ Sauvegarde en  GeoParquet & Graphml (passage Bronze => Silver)."""
    if slim_pois_gdf.crs is None:
        slim_pois_gdf = slim_pois_gdf.set_crs(DEFAULT_CRS)
    slim_pois_gdf = slim_pois_gdf.to_crs(DEFAULT_CRS)  # ensure standard WGS84 coordinates
    slim_pois_gdf.to_parquet(OSM_SILVER_GEOPARQUET)
    log.info(f"Bronze => Silver (OSM) : GeoParquet POIs sauvegardés à {OSM_SILVER_GEOPARQUET}")

    ox.save_graphml(G_drive, filepath=SILVER_DRIVE_GRAPHML)
    log.info(f"Bronze => Silver (OSM) : {len(G_drive.nodes)} noeuds (nodes) et {len(G_drive.edges)} arrêtes (edges) dans le réseau de route 'drive'.")
    log.info(f"Bronze => Silver (OSM) : Graphml sauvegardés à {SILVER_DRIVE_GRAPHML}")

    ox.save_graphml(G_walk, filepath=SILVER_WALK_GRAPHML)
    log.info(f"Bronze => Silver (OSM) : {len(G_walk.nodes)} noeuds (nodes) et {len(G_walk.edges)} arrêtes (edges) dans le réseau de route 'walk'.")
    log.info(f"Bronze => Silver (OSM) : Graphml sauvegardés à {SILVER_WALK_GRAPHML}")
