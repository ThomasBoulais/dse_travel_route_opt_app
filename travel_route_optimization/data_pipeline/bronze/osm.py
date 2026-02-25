"""
OSM - Pipeline d'ingestion Bronze => Silver
====================================================


Usage :
    python3 ingest_osm.py
"""

import osmnx as ox
import geopandas as gpd
import logging

from travel_route_optimization.data_pipeline.utils.config import OSM_BRONZE_GEOPARQUET, OSM_BRONZE_GRAPHML, OSM_PLACE_NAME, OSM_SILVER_GEOPARQUET, OSM_SILVER_GRAPHML

ox.settings.use_cache = True

# CONFIG

# Step 1: Définition de la zone détude

TAGS = {
    "tourism": ["museum", "attraction", "viewpoint", "hotel", "hostel"],
    "amenity": ["restaurant", "cafe", "bar"],
    "leisure": ["park", "nature_reserve"]
}

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# HELPERS

def print_len_col_head(pois_gdf: gpd.GeoDataFrame) -> None:
    log.info(f"POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    if len(pois_gdf.columns.to_list()) < 20:
        log.info(pois_gdf.columns.tolist())
    log.info(pois_gdf.head())


# BRONZE

def get_pois() -> gpd.GeoDataFrame:
    """Récupère les POIs sous format GeoPandas."""
    log.info(f"Téléchargement des POIs OSM de: {OSM_PLACE_NAME}.")
    pois_gdf = ox.features_from_place(OSM_PLACE_NAME, tags=TAGS)
    print_len_col_head(pois_gdf)
    return pois_gdf


def get_road_networks() -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sour format GeoPandas."""
    log.info(f"Téléchargement du réseau de routes de: {OSM_PLACE_NAME}")
    G = ox.graph_from_place(OSM_PLACE_NAME, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def ingest_bronze(pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Bronze)."""
    pois_gdf = pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    pois_gdf.to_parquet(OSM_BRONZE_GEOPARQUET)
    log.info("GeoParquet des POIs bruts sauvegardés à l'endroit suivant: data/bronze/osm/osm_pois.geoparquet")

    ox.save_graphml(G, filepath=OSM_BRONZE_GRAPHML)
    log.info(f"Le graphe a {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges).")
    log.info(f"Graphml des réseaux de routes sauvegardés à l'endroit suivant: {OSM_BRONZE_GRAPHML}")


# SILVER

def transform_silver(pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Réduit aux colonnes utiles + convertit les batîments en point unique (centroïde)."""
    log.info("Réduction aux colonnes pertinentes")
    slim_pois_gdf = pois_gdf[RELEVANT_FIELDS].copy()
    slim_pois_gdf["geometry"] = slim_pois_gdf["geometry"].apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    print_len_col_head(slim_pois_gdf)
    return slim_pois_gdf


# EXPORT

def export_silver(slim_pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """ Sauvegarde en  GeoParquet (passage Bronze => Silver)."""
    slim_pois_gdf = slim_pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    slim_pois_gdf.to_parquet(OSM_SILVER_GEOPARQUET)
    log.info(f"GeoParquet des POIs affinés sauvegardés à l'endroit suivant: {OSM_SILVER_GEOPARQUET}")

    ox.save_graphml(G, filepath=OSM_SILVER_GRAPHML)
    log.info(f"Le graphe a {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges).")
    log.info(f"Graphml des réseaux de routes sauvegardés à l'endroit suivant: {OSM_SILVER_GRAPHML}")


# MAIN

def main():
    log.info("=== Démarrage pipeline OSM Bronze => Silver ===")

    # Bronze: ajout & lecture des fichiers bruts 
    pois_gdf = get_pois()
    G = get_road_networks()
    ingest_bronze(pois_gdf, G)

    # Silver: transformation  & nettoyage
    slim_pois_gdf = transform_silver(pois_gdf)

    # Export
    export_silver(slim_pois_gdf, G)

    # Aperçu
    log.info("=== Pipeline terminé ===")

if __name__ == "__main__":
    main()