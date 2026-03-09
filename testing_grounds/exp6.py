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
    BBOX_TOP
)

def set_boundaries(
        left:    float=BBOX_LEFT,
        bottom:  float=BBOX_BOTTOM, 
        right:   float=BBOX_RIGHT, 
        top:     float=BBOX_TOP
        ) -> list[float]:
    return left, bottom, right, top

# -----------------------------
# 1. Charger le graphe en WGS84 et TRONQUER AVANT projection
# -----------------------------

left, bottom, right, top = set_boundaries()

ox.settings.default_crs = "EPSG:4326"

G_drive = ox.load_graphml(GOLD_DRIVE_GRAPHML)

# Troncature en WGS84
G_drive = ox.truncate.truncate_graph_bbox(
    G_drive,
    bbox=[left, bottom, right, top]
)

G_drive = ox.project_graph(G_drive)
graph_crs = G_drive.graph["crs"]

print("Graph loaded and truncated:", G_drive)

# -----------------------------
# 2. Charger et préparer les POIs
# -----------------------------
pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
# print("Original POI CRS:", pois.crs)

pois = pois.to_crs("EPSG:4326")
pois = pois.cx[left:right, bottom:top].reset_index(drop=True)

print("POIs after bbox filter:", pois.shape)

# 3) Reprojeter dans le CRS du graphe
pois = pois.to_crs(graph_crs)
print("POIs projected to graph CRS:", pois.shape)

# -----------------------------
# 3. Associer chaque POI à son node le plus proche
# -----------------------------
X = pois.geometry.x.values
Y = pois.geometry.y.values

pois["nearest_node"] = ox.nearest_nodes(G_drive, X, Y)

# print(pois.head())

# -----------------------------
# 4. KNN entre POIs (BallTree)
# -----------------------------
coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T
tree = BallTree(np.radians(coords), metric="haversine")

K = 25
distances, indices = tree.query(np.radians(coords), k=K + 1)

neighbors = [set(idx[1:]) for idx in indices]

# -----------------------------
# 5. Ajouter travel_time au graphe
# -----------------------------
def add_travel_time(G, speed_kmh):
    for u, v, data in G.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        data["travel_time"] = length / (speed_kmh * 1000 / 3600)

add_travel_time(G_drive, 40)

def travel_time(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v, weight="travel_time")
    except:
        return np.inf

# -----------------------------
# 6. Construire les arêtes du graphe réduit
# -----------------------------
edges = []
count = 0
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
        # if count % 1000 == 0:
        #     print(f"{count} - ",{
        #         "poi_from": i,
        #         "poi_to": j,
        #         "node_from": u,
        #         "node_to": v,
        #         "drive_time": w_drive,
        #     })
        # count += 1

edges_df = pd.DataFrame(edges)

print(edges_df.shape)
print(edges_df[edges_df['drive_time'] != edges_df.loc[0]['drive_time']].shape)
print(edges_df[edges_df['drive_time'] == edges_df.loc[0]['drive_time']].shape)
print(edges_df.head(10))

edges_df.to_csv(r'testing_grounds\exp6.csv', index=False)