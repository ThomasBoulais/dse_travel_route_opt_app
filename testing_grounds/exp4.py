import fastparquet
import osmnx as ox 
import geopandas as gpd
from pprint import pprint

from travel_route_optimization.data_pipeline.utils.config import GOLD_GRAPHML, GOLD_POIS_GEOPARQUET

# street_networks = ox.load_graphml(GOLD_GRAPHML)

import random

from travel_route_optimization.data_pipeline.utils.pipeline_helpers import to_geopandas

def weighted_sample_without_replacement(population, weights, k, rng=random):
    v = [rng.random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]


# population = range(1, 100)
# weights = sorted(population,reverse=True)
# k = 20

# i = 1
# nb_round = 1

# rng = random.Random(42)

# val_dict = {}

# weighted_sample_without_replacement(population, weights, k, rng)


# street_networks = ox.load_graphml(GOLD_GRAPHML)

# G = ox.graph.graph_from_place("Bédarieux, Hérault, France", network_type="drive")


# COOL !!

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors

N_NEIGHBORS = 100
WEIGHTS = sorted(range(1,N_NEIGHBORS+1),reverse=True)

K_PICK = 20

G = fastparquet.ParquetFile(GOLD_POIS_GEOPARQUET).to_pandas()
G = to_geopandas(G)

# 2d numpy array of the coordinates
coords = np.array(G.geometry.map(lambda p: [p.x, p.y]).tolist())

# "train"/initialize the NearestNeighbors model 
knn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, algorithm='kd_tree').fit(coords)
knn_dist, knn_idx = knn.kneighbors(coords)

# KNN_IDX renvoie une liste de 100 points les plus proches pour chaque POIs du dataset, soit 100 * 12250
# on souhaite récupérer dans chacune des liste de 100 points 20 points avec tirage pondéré sans remise 
# Qu'est-ce qu'on souhaite récupérer ? Les idx pour avoir la geometry + les distances (déjà calculées)
# Il ne reste qu'à créer le graphe contenant les arrêtes + valeurs de coût


# # add results to dataframe:
# G[list(map("NEIGHBOR_{}".format, range(1, N_NEIGHBORS + 1)))] = \
#         G.geometry.values.to_numpy()[knn_idx[:, 1:]]

# print(G)

res = weighted_sample_without_replacement(knn_idx[:N_NEIGHBORS], WEIGHTS, K_PICK)

print(knn_idx[:N_NEIGHBORS])

print(len(knn_idx[:N_NEIGHBORS]))
print(len(WEIGHTS))
print(len(res))

# for i in range(1, 10):
#     print(G.loc[int(knn_idx[0][i])])