"""
Datatourisme & OSM - Transformation Silver => Gold
"""

import fastparquet
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import osmnx as ox
from rapidfuzz import fuzz, utils # https://github.com/rapidfuzz/RapidFuzz


from travel_route_optimization.data_pipeline.bronze.osm import get_pois
from travel_route_optimization.data_pipeline.utils.config import BBOX_BOTTOM, BBOX_LEFT, BBOX_RIGHT, BBOX_TOP, DEFAULT_CRS, DRIVE_SPEED, DT_SILVER_GEOPARQUET, GOLD_DRIVE_GRAPHML, GOLD_POIS_CSV, GOLD_POIS_GEOPARQUET, KNN_DRIVE_TIME_GRAPH_DF, OSM_SILVER_GEOPARQUET
from travel_route_optimization.data_pipeline.utils.pipeline_helpers import add_travel_time, add_visit_duration, dt_add_category, dt_add_open_hour_mask, get_drive_network, get_knn_pois, nearest_node, osm_add_category, osm_add_open_hour_mask, to_geopandas, travel_time


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# OSM

def osm_transform_gold():
    """
    Prépare OSM Silver pour le merge final
    - standardise les colonnes
    - supprime les noms de POIs vides
    - ajoute les catégories & le masque d'horaires d'ouverture
    - fixe le CRS (Coordinate Reference System)
    """
    osm_df = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET).to_pandas()
    osm_gdf = to_geopandas(osm_df)
    osm_gdf = osm_add_category(osm_gdf)
    osm_gdf = osm_add_open_hour_mask(osm_gdf)
    osm_gdf = add_visit_duration(osm_gdf, 'OSM')

    osm_gdf = osm_gdf[['name', 'geometry', 'categories', 'types', 'phone', 'website', 'opening_hours', 'opening_mask', 'visit_duration']]
    osm_gdf.set_crs(DEFAULT_CRS, inplace=True)
    return osm_gdf


# DATATOURISME

def dt_transform_gold():
    """
    Prépare DT Silver pour le merge final
    - standardise les colonnes
    - supprime les noms de POIs vides
    - ajoute les catégories & le masque d'horaires d'ouverture
    - fixe le CRS (Coordinate Reference System)
    """
    dt_df = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET).to_pandas()
    dt_gdf = to_geopandas(dt_df)
    dt_gdf = dt_add_category(dt_gdf)
    dt_gdf = dt_add_open_hour_mask(dt_gdf)
    dt_gdf = add_visit_duration(dt_gdf, 'DATATOURISME')

    dt_gdf['name'] = dt_gdf['name_fr'].apply(lambda x: str(x[0].title()))
    dt_gdf = dt_gdf.rename({'id': 'id_dt'}, axis=1)[['id_dt', 'name', 'geometry', 'categories', 'types', 'email', 'phone', 'website', 'opening_hours', 'opening_mask', 'visit_duration']]
    dt_gdf.set_crs(DEFAULT_CRS, inplace=True)
    return dt_gdf


# MERGE

def get_id_equivalent(dt_row: gpd.GeoSeries, osm_m: gpd.GeoDataFrame) -> int|None:
    """Renvoie l'index OSM du POI équivalent côté DATATourisme"""

    # 1. on veut filtrer sur les résultats proches (~20m) et garder tous les candidats,
    delta  = .0002 # .0001 => 11.132 m (https://www.tuto-carto.fr/longitude-latitude-precision/)
    osm_nearest = osm_m.cx[dt_row['geometry'].x - delta : dt_row['geometry'].x + delta, 
                           dt_row['geometry'].y - delta : dt_row['geometry'].y + delta]
    # print('\ndt_row\n', dt_row[['name', 'geometry']])
    # print('\nosm_nearest\n', osm_nearest[['name', 'geometry']].head())

    # 2. puis déterminer un score de ressemblance des noms sur les candidats restants
    osm_nearest['lev_score'] = osm_nearest['name'].apply(
        lambda x: fuzz.partial_ratio(str(x), str(dt_row['name']), processor=utils.default_process)
        )
    # print(osm_nearest[['name', 'geometry', 'lev_score']])

    # 3. et enfin renvoyer une valeur si lev_score > 75%
    try:
        return osm_nearest[osm_nearest['lev_score'] > 75].head(1).index[0][1]
    except IndexError:
        return None


def merge_gold(dt_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fusionne les 2 sources de données"""
    dt_m = dt_gdf.to_crs(DEFAULT_CRS) # EPSG:2154 => projection métrique française officielle
    osm_m = osm_gdf.to_crs(DEFAULT_CRS)

    dt_m['osm_match_id'] = dt_m.apply(get_id_equivalent, args=(osm_m,), axis=1)
    # print(dt_m.notnull().sum())

    # split en 3 chunk : inter, DT uniq. et OSM uniq.
    # inter         => dt_m['osm_match_id'].notnull()
    # DT uniq.      => dt_m['osm_match_id'].isnull()
    # OSM uniq.     => restriction sur la liste des osm_match_id 

    dt_uniq = dt_m[dt_m['osm_match_id'].isnull()]
    dt_osm = dt_m[dt_m['osm_match_id'].notnull()]
    
    # besoin de rajouter les horaires de OSM car ils sont mieux que DT
    dt_osm = dt_osm.drop(['opening_hours', 'opening_mask'], axis=1)\
        .join(osm_gdf.reset_index().rename({'id':'osm_match_id'}), on='osm_match_id',rsuffix='_osm')\
        .rename({'opening_hours_osm':'opening_hours', 'opening_mask_osm': 'opening_mask'})

    osm_match_id = list(map(int, dt_m.get('osm_match_id').dropna().tolist()))
    osm_uniq = osm_m.reset_index()[(osm_m.reset_index()['name'].notnull()) & (~osm_m.reset_index()['id'].isin(osm_match_id))]
    
    gold_gdf = pd.concat([dt_osm, dt_uniq, osm_uniq])[['name', 'geometry', 'categories', 'types', 'email', 'phone', 
                                                    'website', 'opening_hours', 'opening_mask', 'visit_duration']]
    
    log.info(f"Silver => Gold (MERGE) : {gold_gdf.shape[0]} POIs récupérés après fusion,"
             f" entre {dt_uniq.shape[0]} POIs DATATourisme et {osm_uniq.shape[0]} POIs OSM ({dt_osm.shape[0]} POIs en commun)")
    
    return gold_gdf


# CREATION GRAPHE POUR RL

def create_knn_drive_graph(G_drive: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """Génère le graphe de training du RL (K voisins les plus proches (KNN) par POI avec temps de trajet pour chaque arrête)"""
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


# EXPORT

def flatten_mask(x):
    """Applatit le mask (7, 1440) en (10080,)"""
    if not isinstance(x, np.ndarray):
        x = np.zeros((7, 1440), dtype=np.uint8)
    return x.flatten().tolist()


def export_gold(gold_gdf: gpd.GeoDataFrame, 
                G_drive : gpd.GeoDataFrame, 
                # G_walk  : gpd.GeoDataFrame,
                edges_df: pd.DataFrame) -> None:
    """Sauvegarde :
    - les POIs en GeoParquet (Gold) avec CSV optionnel pour analyse,
    - le réseau de route en voiture en Geoparquet,
    - le graphe des KNN avec tps de trajet en pd.DataFrame
    """
    if gold_gdf.crs is None:
        gold_gdf = gold_gdf.set_crs(DEFAULT_CRS)
    gold_gdf = gold_gdf.to_crs(DEFAULT_CRS)

    gold_gdf["opening_mask_flat"] = gold_gdf["opening_mask"].apply(flatten_mask)
    gold_gdf = gold_gdf.drop(columns=["opening_mask"])
    # pour pouvoir le reconstruire => mask = np.array(flat).reshape(7, 1440)

    gold_gdf.to_parquet(GOLD_POIS_GEOPARQUET, index=False)
    log.info(f"Silver => Gold (MERGE) : GeoParquet sauvegardé : {GOLD_POIS_GEOPARQUET}  ({len(gold_gdf):,} lignes)")

    # CSV sans géométrie pour exploration rapide
    gold_gdf.drop(columns="geometry").to_csv(GOLD_POIS_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Silver => Gold (MERGE) : CSV sauvegardé       : {GOLD_POIS_CSV}")

    ox.save_graphml(G_drive, filepath=GOLD_DRIVE_GRAPHML)
    log.info(f"Silver => Gold (MERGE) : {len(G_drive.nodes)} noeuds (nodes) et {len(G_drive.edges)} arrêtes (edges) dans le réseau de route 'drive'.")
    log.info(f"Silver => Gold (MERGE) : Graphml sauvegardés à {GOLD_DRIVE_GRAPHML}")

    # ox.save_graphml(G_walk, filepath=GOLD_WALK_GRAPHML)
    # log.info(f"Silver => Gold (MERGE) : {len(G_walk.nodes)} noeuds (nodes) et {len(G_walk.edges)} arrêtes (edges) dans le réseau de route 'walk'.")
    # log.info(f"Silver => Gold (MERGE) : Graphml sauvegardés à {GOLD_WALK_GRAPHML}")

    edges_df.to_csv(KNN_DRIVE_TIME_GRAPH_DF, index=False)
    log.info(f"Silver => Gold (MERGE) : {len(edges_df)} arrêtes dans le graphe KNN.")
    log.info(f"Silver => Gold (MERGE) : pd.DataFrame sauvegardés à {KNN_DRIVE_TIME_GRAPH_DF}")

