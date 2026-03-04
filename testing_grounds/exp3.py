import pprint
import fastparquet
import pandas as pd
import geopandas as gpd
import logging
from rapidfuzz import fuzz, utils # https://github.com/rapidfuzz/RapidFuzz


from testing_grounds.exp2 import to_geopandas
from travel_route_optimization.data_pipeline.utils.config import DT_DICT_TYPES_DETAILED, DT_SILVER_GEOPARQUET, GOLD_POIS_CSV, GOLD_POIS_GEOPARQUET, OSM_DICT_TYPES_DETAILED, OSM_SILVER_GEOPARQUET


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# DATATOURISME

def summarize_types(types: str, dict_types_detailed: dict) -> str:
    """Ajoute les catégoires à une chaîne de caractères des types"""
    set_cat = set()
    for item in types.split('|'):
        try:
            set_cat.add(dict_types_detailed[item])
        except KeyError:
            pass
    return "|".join(map(str, list(set_cat)))


def dt_add_category(dt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte les catégories correspondantes aux types des POIs (DATATourisme)"""
    dt_gdf['categories'] = dt_gdf['types'].apply(summarize_types, args=(DT_DICT_TYPES_DETAILED,))
    log.info("DATATOURISME - Gold : Catégories ajoutées aux POIs")
    return dt_gdf

def dt_transform_gold():
    """
    Prépare DT Silver pour le merge final
    - standardise les colonnes
    - supprime les noms de POIs vides
    - fixe le CRS (Coordinate Reference System)
    """
    dt_df = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET).to_pandas()
    dt_gdf = to_geopandas(dt_df)
    dt_gdf = dt_add_category(dt_gdf)
    dt_gdf = dt_gdf.rename({'name_fr': 'name', 'id': 'id_dt'}, axis=1)[['id_dt', 'name', 'geometry', 'categories', 'types', 'email', 'phone', 'website', 'opening_hours']]
    dt_gdf.set_crs("EPSG:2154", inplace=True)
    return dt_gdf

# OSM

def osm_get_types(tourism: str|None, amenity: str|None, historic: str|None, leisure: str|None, natural: str|None) -> str|None:
    """Affecte les types correspondants aux POIs (OSM)"""
    set_types = set()
    set_types.add(tourism)
    set_types.add(amenity)
    set_types.add(historic)
    set_types.add(leisure)
    set_types.add(natural)
    set_types.remove(None)
    return "|".join(map(str, list(set_types)))


def osm_add_category(osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte les catégories correspondantes aux types des POIs (OSM)"""
    osm_gdf['types'] = osm_gdf.apply(lambda x: osm_get_types(x.tourism, x.amenity, x.historic, x.leisure, x.natural), axis=1)
    osm_gdf['categories'] = osm_gdf['types'].apply(summarize_types, args=(OSM_DICT_TYPES_DETAILED,))
    log.info("OSM - Gold : Catégories ajoutées aux POIs")
    return osm_gdf


def osm_transform_gold():
    """
    Prépare OSM Silver pour le merge final
    - standardise les colonnes
    - supprime les noms de POIs vides
    - fixe le CRS (Coordinate Reference System)
    """
    osm_df = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET).to_pandas()
    osm_gdf = to_geopandas(osm_df)
    osm_gdf = osm_add_category(osm_gdf)
    osm_gdf = osm_gdf[['name', 'geometry', 'categories', 'types', 'phone', 'website', 'opening_hours']]
    osm_gdf.set_crs("EPSG:2154", inplace=True)
    return osm_gdf


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
    dt_m = dt_gdf.to_crs("EPSG:2154") # EPSG:2154 => projection métrique française officielle
    osm_m = osm_gdf.to_crs("EPSG:2154")

    dt_m['osm_match_id'] = dt_m.apply(get_id_equivalent, args=(osm_m,), axis=1)
    print(dt_m.notnull().sum())

    # split en 3 chunk : inter, DT uniq. et OSM uniq.
    # inter déjà fait       => dt_m['osm_match_id'].notnull()
    # DT uniq. déjà fait    => dt_m['osm_match_id'].isnull()
    # OSM uniq. à faire     => restriction sur la liste des osm_match_id 

    dt_uniq = dt_m[dt_m['osm_match_id'].isnull()]
    dt_osm = dt_m[dt_m['osm_match_id'].notnull()]

    osm_match_id = list(map(int, dt_m.get('osm_match_id').dropna().tolist()))
    osm_uniq = osm_m.reset_index()[(osm_m.reset_index()['name'].notnull()) & (~osm_m.reset_index()['id'].isin(osm_match_id))]

    gold_gdf = pd.concat([dt_osm, dt_uniq, osm_uniq])[['name', 'geometry', 'categories', 'types', 'email', 'phone', 
                                                    'website', 'opening_hours']]
    
    log.info(f"MERGE - Gold : {gold_gdf.shape[0]} POIs récupérés après fusion,"
             f"après fusion entre {dt_uniq.shape[0]} POIs DATATourisme et {osm_uniq.shape[0]} POIs OSM ({st_osm.shape[0]} POIs en commun)")
    
    return gold_gdf


def export_gold(gold_gdf: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Silver) et CSV optionnel."""
    gold_gdf.to_parquet(GOLD_POIS_GEOPARQUET, index=False)
    log.info(f"GOLD - GeoParquet sauvegardé : {GOLD_POIS_GEOPARQUET}  ({len(gold_gdf):,} lignes)")

    # CSV sans géométrie pour exploration rapide
    gold_gdf.drop(columns="geometry").to_csv(GOLD_POIS_CSV, index=False, encoding="utf-8-sig")
    log.info(f"GOLD - CSV sauvegardé       : {GOLD_POIS_CSV}")



def main():

    dt_gdf = dt_transform_gold()
    osm_gdf = osm_transform_gold()
    gold_gdf = merge_gold(dt_gdf, osm_gdf)    
    export_gold(gold_gdf)



if __name__ == "__main__":
    main()