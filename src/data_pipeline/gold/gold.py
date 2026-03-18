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


# from src.utils.config import DEFAULT_CRS, DRIVE_SPEED, DT_SILVER_GEOPARQUET, GOLD_DRIVE_GRAPHML, GOLD_POIS_CSV, GOLD_POIS_GEOPARQUET, KNN_DRIVE_TIME_GRAPH_DF, OSM_SILVER_GEOPARQUET
from src.data_pipeline.utils.pipeline_helpers import extract_categories, add_interest_score, add_travel_time, add_visit_duration, dt_add_category, dt_add_open_hour_mask, get_knn_pois, nearest_node, osm_add_category, osm_add_open_hour_mask, to_geopandas, travel_time
from src.common.config_loader import load_config

cfg = load_config()

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
    osm_df = fastparquet.ParquetFile(cfg.silver.osm_geoparquet).to_pandas()
    osm_gdf = to_geopandas(osm_df)
    osm_gdf = osm_add_category(osm_gdf)
    osm_gdf = osm_add_open_hour_mask(osm_gdf)
    osm_gdf = add_visit_duration(osm_gdf, 'OSM')

    osm_gdf = osm_gdf[['name', 'geometry', 'categories', 'types', 'phone', 'website', 'opening_hours', 'opening_mask', 'visit_duration']]
    osm_gdf.set_crs(cfg.crs.default, inplace=True)
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
    dt_df = fastparquet.ParquetFile(cfg.silver.dt_geoparquet).to_pandas()
    dt_gdf = to_geopandas(dt_df)
    dt_gdf = dt_add_category(dt_gdf)
    dt_gdf = dt_add_open_hour_mask(dt_gdf)
    dt_gdf = add_visit_duration(dt_gdf, 'DATATOURISME')

    dt_gdf['name'] = dt_gdf['name_fr'].apply(lambda x: str(x[0].title()))
    dt_gdf = dt_gdf.rename({'id': 'id_dt'}, axis=1)[['id_dt', 'name', 'geometry', 'categories', 'types', 'email', 'phone', 'website', 'opening_hours', 'opening_mask', 'visit_duration']]
    dt_gdf.set_crs(cfg.crs.default, inplace=True)
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
    dt_m = dt_gdf.to_crs(cfg.crs.default) # EPSG:2154 => projection métrique française officielle
    osm_m = osm_gdf.to_crs(cfg.crs.default)

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
    
    gold_gdf["interest_score"] = gold_gdf.apply(add_interest_score, axis=1)
    gold_gdf["main_category"] = gold_gdf["categories"].apply(lambda x: extract_categories(x)[0] if extract_categories(x) else "")

    log.info(f"Silver => Gold (MERGE) : {gold_gdf.shape[0]} POIs récupérés après fusion,"
             f" entre {dt_uniq.shape[0]} POIs DATATourisme et {osm_uniq.shape[0]} POIs OSM ({dt_osm.shape[0]} POIs en commun)")
    
    gold_gdf = gold_gdf.reset_index(drop=True)

    return gold_gdf


# CREATION GRAPHE POUR RL

def create_knn_drive_graph(G_drive: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """Génère le graphe de training du RL (K voisins les plus proches (KNN) par POI avec temps de trajet pour chaque arrête)"""
    pois = pois.reset_index(drop=True)
    pois = nearest_node(pois, G_drive)
    neighbors = get_knn_pois(pois)

    add_travel_time(G_drive, cfg.parameters.drive_speed)
    
    log.info(f"Silver => Gold (KNN_GRAPH) : Création du graphe KNN des temps de route entre POIs")
    
    edges = []
    for i, poi in pois.iterrows():
        u = poi["nearest_node"]
        for j in neighbors[i]:
            v = pois.iloc[j]["nearest_node"]
            w_drive = travel_time(G_drive, u, v)
            try:
                w_drive = int(w_drive)+1
            except OverflowError:
                continue
            edges.append({
                "poi_from": i,
                "poi_to": j,
                "node_from": u,
                "node_to": v,
                "drive_time": w_drive,
            })
    edges_df = pd.DataFrame(edges)

    len_raw_edges_df = len(edges_df)

    edges_df = edges_df.dropna()
    edges_df = edges_df[edges_df['poi_from'] != edges_df['poi_to']]
    edges_df['drive_time'] = edges_df['drive_time'].astype(float)
    edges_df = edges_df[edges_df['drive_time'] < 1e9]

    unique_from = set(edges_df["poi_from"])
    all_pois = set(range(len(pois)))
    isolated = all_pois - unique_from

    for poi in isolated:
        nearest = neighbors[poi][0]
        u = pois.loc[poi, "nearest_node"]
        v = pois.loc[nearest, "nearest_node"]
        w_drive = travel_time(G_drive, u, v)
        try:
            w_drive = int(w_drive)+1
        except OverflowError:
            continue
        edges_df = pd.concat([
            edges_df,
            pd.DataFrame([{
                "poi_from": int(poi),
                "poi_to": int(nearest),
                "node_from": u,
                "node_to": v,
                "drive_time": int(w_drive)+1,
            }])
        ], ignore_index=True)

    # On inverse les from/to pour obtenir les edges dans l'autre sens et obtenir un graphe symétrique
    reverse_edges = edges_df.rename(columns={
        "poi_from": "poi_to",
        "poi_to": "poi_from",
        "node_from": "node_to",
        "node_to": "node_from"
    })

    edges_df = pd.concat([edges_df, reverse_edges], ignore_index=True)
    edges_df.drop_duplicates(subset=["poi_from", "poi_to"], inplace=True)


    log.info(f"Silver => Gold (KNN_GRAPH) : {len(edges_df)} arrêtes créées dans le graphe KNN ({len_raw_edges_df} avant suppression des doublons et NA)")

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
        gold_gdf = gold_gdf.set_crs(cfg.crs.default)
    gold_gdf = gold_gdf.to_crs(cfg.crs.default)

    gold_gdf["opening_mask_flat"] = gold_gdf["opening_mask"].apply(flatten_mask)
    gold_gdf = gold_gdf.drop(columns=["opening_mask"])
    # pour pouvoir le reconstruire => mask = np.array(flat).reshape(7, 1440)

    gold_gdf.to_parquet(cfg.gold.pois_geoparquet, index=False)
    log.info(f"Silver => Gold (MERGE) : GeoParquet sauvegardé : {cfg.gold.pois_geoparquet}  ({len(gold_gdf):,} lignes)")

    # CSV sans géométrie pour exploration rapide
    gold_gdf.drop(columns="geometry").to_csv(cfg.gold.pois_csv, index=False, encoding="utf-8-sig")
    log.info(f"Silver => Gold (MERGE) : CSV sauvegardé       : {cfg.gold.pois_csv}")

    ox.save_graphml(G_drive, filepath=cfg.gold.drive_graphml)
    log.info(f"Silver => Gold (MERGE) : {len(G_drive.nodes)} noeuds (nodes) et {len(G_drive.edges)} arrêtes (edges) dans le réseau de route 'drive'.")
    log.info(f"Silver => Gold (MERGE) : Graphml sauvegardés à {cfg.gold.drive_graphml}")

    # ox.save_graphml(G_walk, filepath=GOLD_WALK_GRAPHML)
    # log.info(f"Silver => Gold (MERGE) : {len(G_walk.nodes)} noeuds (nodes) et {len(G_walk.edges)} arrêtes (edges) dans le réseau de route 'walk'.")
    # log.info(f"Silver => Gold (MERGE) : Graphml sauvegardés à {GOLD_WALK_GRAPHML}")

    edges_df.to_csv(cfg.gold.knn_drive_time_graph_df, index=False)
    log.info(f"Silver => Gold (MERGE) : {len(edges_df)} arrêtes dans le graphe KNN.")
    log.info(f"Silver => Gold (MERGE) : pd.DataFrame sauvegardés à {cfg.gold.knn_drive_time_graph_df}")

