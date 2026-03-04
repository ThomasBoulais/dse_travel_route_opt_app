"""
OSM - Transformation Bronze => Silver
"""

import osmnx as ox
import geopandas as gpd
import logging

from travel_route_optimization.data_pipeline.utils.config import OSM_BRONZE_GEOPARQUET, OSM_BRONZE_GRAPHML, OSM_PLACE_NAME, OSM_SILVER_GEOPARQUET, OSM_SILVER_GRAPHML
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
    log.info("OSM - Silver : Réduction des POIs aux colonnes pertinentes")
    slim_pois_gdf = pois_gdf[RELEVANT_FIELDS].copy()
    slim_pois_gdf["geometry"] = slim_pois_gdf["geometry"].apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    print_len_col_head(slim_pois_gdf)
    return slim_pois_gdf


def export_silver(slim_pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """ Sauvegarde en  GeoParquet & Graphml (passage Bronze => Silver)."""
    slim_pois_gdf = slim_pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    slim_pois_gdf.to_parquet(OSM_SILVER_GEOPARQUET)
    log.info(f"OSM - Silver : GeoParquet POIs sauvegardés à {OSM_SILVER_GEOPARQUET}")

    ox.save_graphml(G, filepath=OSM_SILVER_GRAPHML)
    log.info(f"OSM - Silver : {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges) dans le réseau de routes.")
    log.info(f"OSM - Silver : Graphml sauvegardés à {OSM_SILVER_GRAPHML}")
