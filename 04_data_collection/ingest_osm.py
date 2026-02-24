"""
OSM - Pipeline d'ingestion Bronze => Silver
====================================================

Usage :
    python3 ingest_osm.py
"""

import osmnx as ox
import geopandas as gpd
import logging

ox.settings.use_cache = True

# CONFIG

# Step 1: Définition de la zone détude
PLACE_NAME = "Hérault, Occitanie, France"
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
    """Récupère les POIs sous format GeoPandas"""
    log.info(f"Téléchargement des POIs OSM de: {PLACE_NAME}.")
    pois_gdf = ox.features_from_place(PLACE_NAME, tags=TAGS)
    print_len_col_head(pois_gdf)
    return pois_gdf


def get_road_networks() -> gpd.GeoDataFrame:
    """Récupère les réseaux de routes sour format GeoPandas"""
    log.info(f"Téléchargement du réseau de routes de: {PLACE_NAME}")
    G = ox.graph_from_place(PLACE_NAME, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def ingest_bronze(pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Bronze)"""
    pois_gdf = pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    pois_gdf.to_parquet("data/bronze/osm/osm_pois.geoparquet")
    log.info("GeoParquet des POIs bruts sauvegardés à l'endroit suivant: data/bronze/osm/osm_pois.geoparquet")

    ox.save_graphml(G, filepath="data/bronze/osm/osm_road_network.graphml")
    log.info(f"Le graphe a {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges).")
    log.info("Graphml des réseaux de routes sauvegardés à l'endroit suivant: data/bronze/osm/osm_road_network.graphml")


# SILVER

def transform_silver(pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Réduit aux colonnes utiles + convertit les batîments en point unique (centroïde)"""
    log.info("Réduction aux colonnes pertinentes")
    slim_pois_gdf = pois_gdf[RELEVANT_FIELDS].copy()
    slim_pois_gdf["geometry"] = slim_pois_gdf["geometry"].apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    print_len_col_head(slim_pois_gdf)
    return slim_pois_gdf


# EXPORT

def export_silver(slim_pois_gdf: gpd.GeoDataFrame, G: gpd.GeoDataFrame) -> None:
    """ Sauvegarde en  GeoParquet (passage Bronze => Silver)"""
    slim_pois_gdf = slim_pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
    slim_pois_gdf.to_parquet("data/silver/osm_pois_slim.geoparquet")
    log.info("GeoParquet des POIs affinés sauvegardés à l'endroit suivant: data/silver/osm_pois_slim.geoparquet")

    ox.save_graphml(G, filepath="data/silver/osm/osm_road_network.graphml")
    log.info(f"Le graphe a {len(G.nodes)} noeuds (nodes) et {len(G.edges)} arrêtes (edges).")
    log.info("Graphml des réseaux de routes sauvegardés à l'endroit suivant: data/silver/osm/osm_road_network.graphml")


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