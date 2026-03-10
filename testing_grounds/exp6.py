# ============================================================
#   CREATION DU GRAPHE REDUIT POUR LE RL (VERSION FINALE)
# ============================================================

import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from travel_route_optimization.data_pipeline.utils.config import (
    GOLD_DRIVE_GRAPHML,
    GOLD_POIS_GEOPARQUET,
    BBOX_LEFT,
    BBOX_BOTTOM,
    BBOX_RIGHT,
    BBOX_TOP,
    KNN_DRIVE_TIME_GRAPH_DF
)

DRIVE_SPEED = 40 # km/h

def get_drive_network(left: float, bottom: float, right: float, top: float) -> gpd.GeoDataFrame:
    """Récupère le graphml du réseau de route"""
    ox.settings.default_crs = "EPSG:4326"
    G_drive = ox.load_graphml(GOLD_DRIVE_GRAPHML)
    G_drive = ox.truncate.truncate_graph_bbox(G_drive, bbox=[left, bottom, right, top])
    G_drive = ox.project_graph(G_drive)
    return G_drive


def get_pois(G_drive: gpd.GeoDataFrame, left: float, bottom: float, right: float, top: float) -> gpd.GeoDataFrame:
    """Récupère le GeoDataFrame des POIs dans le même CRS que G_drive"""
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    pois = pois.to_crs("EPSG:4326")
    pois = pois.cx[left:right, bottom:top].reset_index(drop=True)
    pois = pois.to_crs(G_drive.graph["crs"])
    return pois


def nearest_node(pois: gpd.GeoDataFrame, G_drive: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Associe à chaque POI son node le plus proche"""
    X = pois.geometry.x.values
    Y = pois.geometry.y.values
    pois["nearest_node"] = ox.nearest_nodes(G_drive, X, Y)
    return pois


def get_knn_pois(pois: gpd.GeoDataFrame) -> list[int]:
    coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T
    tree = BallTree(np.radians(coords), metric="haversine")

    K = 25
    distances, indices = tree.query(np.radians(coords), k=K + 1)

    neighbors = [set(idx[1:]) for idx in indices]
    return neighbors


def add_travel_time(G: gpd.GeoDataFrame, speed_kmh: int) -> None:
    for u, v, data in G.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        data["travel_time"] = length / (speed_kmh * 1000 / 3600)


def travel_time(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v, weight="travel_time")
    except:
        return np.inf


def create_graph(
        left:    float=BBOX_LEFT,
        bottom:  float=BBOX_BOTTOM, 
        right:   float=BBOX_RIGHT, 
        top:     float=BBOX_TOP) -> pd.DataFrame:
    """Génère le graphe d'arrêtes de K voisins les plus proches (KNN) avec temps de trajet"""
    
    G_drive = get_drive_network(left, bottom, right, top)
    # print("Graph loaded and truncated:", G_drive)

    pois = get_pois(G_drive, left, bottom, right, top)
    # print("POIs after bbox filter:", pois.shape)
    # print("POIs projected to graph CRS:", pois.shape)

    pois = nearest_node(pois, G_drive)
    neighbors = get_knn_pois(pois)

    add_travel_time(G_drive, DRIVE_SPEED)

    edges = []
    for i, poi in pois.iterrows():
        u = poi["nearest_node"]
        for j in neighbors[i]:
            v = pois.iloc[j]["nearest_node"]
            w_drive = travel_time(G_drive, u, v)
            edges.append({
                "poi_from": i,
                "poi_to": j,
                "node_from": u,
                "node_to": v,
                "drive_time": w_drive,
            })
    
    edges_df = pd.DataFrame(edges)
    return edges_df


def export_graph(edges_df: pd.DataFrame) -> None:
    edges_df.to_csv(KNN_DRIVE_TIME_GRAPH_DF, index=False)


def main():
    edges_df = create_graph()
    export_graph(edges_df)

if __name__ == "__main__":
    main()