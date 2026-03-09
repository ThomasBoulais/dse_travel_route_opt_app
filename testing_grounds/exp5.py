# CREATION DU GRAPHE REDUIT POUR LE RL

# Récupère le node (road_network) le plus proche de chaque POI
from travel_route_optimization.data_pipeline.utils.config import DEFAULT_CRS


if True:
    # 1. Charger road networks via OSMnx & projeter sur même CRS
    # 2. Charger POIs via geopandas & projeter sur même CRS
    # 3. Utiliser index spatial (R-tree) pour récup le node le plus proche pour chaque POI

    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point

    from travel_route_optimization.data_pipeline.utils.config import GOLD_DRIVE_GRAPHML, GOLD_WALK_GRAPHML, GOLD_POIS_GEOPARQUET

    left =      3.13745 # bbox de délimitation autour de Bédarieux
    bottom =    43.5910
    right =     3.18947
    top =       43.6367


    # 1. Charger road networks via OSMnx & projeter sur même CRS
    ox.settings.default_crs = DEFAULT_CRS
    G_drive = ox.load_graphml(GOLD_DRIVE_GRAPHML)
    G_drive = ox.truncate.truncate_graph_bbox(G_drive, bbox=[left, bottom, right, top])

    print(G_drive)


    # 2. Charger POIs via geopandas & projeter sur même CRS
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    pois = pois.cx[left:right, bottom:top].reset_index(drop=True)
    pois = pois.to_crs(DEFAULT_CRS)

    print(pois.shape)


    # 3. Utiliser index spatial (R-tree) pour récup le node le plus proche pour chaque POI
    nodes, edges = ox.graph_to_gdfs(G_drive, nodes=True, edges=True)

    # Récup les coordonnées
    X = pois.geometry.x.values
    Y = pois.geometry.y.values

    # Trouve le node le plus proche
    nearest_node_ids = ox.nearest_nodes(G_drive, X, Y)

    # Rattache la donnée au POI
    pois["nearest_node"] = nearest_node_ids

    print(pois.head())


# Crée pour chaque POI k arrêtes (k = 25)
# 1. stratégie = les 15 voisins les plus proches + 1 de chaque catégorie différente
# 2. poids de la décision : w(A,B) = travel_time(A,B) + visit_duration(B) 
# \\ bonus: pts -/+ répét/divers, opening hours, preference user
#  

# 1. Création du Spatial Index 
import numpy as np
from sklearn.neighbors import BallTree

# Extract coordinates
coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T

# BallTree in radians
tree = BallTree(np.radians(coords), metric="haversine")

K = 25  # nb de nearest neighbors (NN)

distances, indices = tree.query(np.radians(coords), k=K+1)
# indices[:, 0] is the point itself → drop it
knn_overall = [set(idx[1:]) for idx in indices]

# # 3. Nearest neighbors avec category specific
# category_groups = pois.groupby("category").groups
# category_knn = [set() for _ in range(len(pois))]

# M = 2  # nearest per category

# for cat, idxs in category_groups.items():
#     sub_coords = coords[idxs]
#     sub_tree = BallTree(np.radians(sub_coords), metric="haversine")
    
#     dist, sub_idx = sub_tree.query(np.radians(sub_coords), k=min(M+1, len(idxs)))
    
#     for i, poi_idx in enumerate(idxs):
#         neighbors = [idxs[j] for j in sub_idx[i] if idxs[j] != poi_idx]
#         category_knn[poi_idx].update(neighbors)

# # 4. Fusion des neighbors
# neighbors = []
# for i in range(len(pois)):
#     merged = knn_overall[i].union(category_knn[i])
#     neighbors.append(list(merged))

neighbors = knn_overall

# 5. Calcul du temps de transport avec OSMnx
import osmnx as ox

def travel_time(G, u, v):
    try:
        return ox.shortest_path_length(G, u, v, weight="travel_time")
    except:
        return np.inf
    
def add_travel_time(G, speed_kmh):
    for u, v, data in G.edges(data=True):
        length_m = data.get("length", 0)
        data["travel_time"] = length_m / (speed_kmh * 1000 / 3600)
        # print(f"{data["travel_time"]} \t= {length_m} / ({speed_kmh} * 1000 * 3600)")

add_travel_time(G_drive, speed_kmh=40)
# add_travel_time(G_walk, speed_kmh=5)

# 6. Constrution des arrêts pour le graphe
edges = []

for i, poi in pois.iterrows():
    u = poi.nearest_node
    
    for j in neighbors[i]:
        v = pois.loc[j]["nearest_node"]
        
        w_drive = travel_time(G_drive, u, v)
        # w_walk = travel_time(G_walk, u, v)
        
        # Hybrid rule: choose best mode
        # w_hybrid = min(w_drive, w_walk)
        
        edges.append({
            "poi_from": i,
            "poi_to": j,
            "node_from": u,
            "node_to": v,
            "car_time": w_drive,
            # "walk_time": w_walk,
            # "hybrid_time": w_hybrid
        })

import pandas as pd
edges_df = pd.DataFrame(edges)

print(edges_df.shape)
print(edges_df.head(10))
print(edges_df.loc[0]['car_time'])
print(edges_df[edges_df['car_time'] != edges_df.loc[0]['car_time']].shape)
print(edges_df[edges_df['car_time'] == edges_df.loc[0]['car_time']].shape)