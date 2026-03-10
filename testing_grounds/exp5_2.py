
from logging import log
import geopandas as gpd
from travel_route_optimization.data_pipeline.gold.gold import dt_transform_gold


import re

from travel_route_optimization.data_pipeline.utils.pipeline_helpers import HORAIRE_ACCOMODATION, HORAIRE_GENERIQUE, HORAIRE_RESTAURATION

def bool_restau(categories: str) -> str:
    """Ajoute les catégoires à une chaîne de caractères des types"""
    if re.search('restauration', categories):
        return True
    return False

def dt_select_opening_mask_type(categories: str) -> str:
    """Attribue le type de opening_mask selon s'il s'agit d'un restaurant ou pas"""
    if re.search('restauration', categories):
        return HORAIRE_RESTAURATION
    if re.search('accomodation', categories):
        return HORAIRE_ACCOMODATION
    return HORAIRE_GENERIQUE


def dt_add_open_hour_mask(dt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte un masque d'horaires d'ouverture (vu le format en date et non horaire, toutes les horaires sont fixés en générique)"""
    dt_gdf["opening_mask"] = dt_gdf['categories'].apply(dt_select_opening_mask_type)
    log.info("Silver => Gold (DATATOURISME) : Opening mask ajoutées aux POIs")
    return dt_gdf


def select_visit_type(categories: str) -> str:
    """Attribue le type de visit_duration selon s'il s'agit d'une accomodation ou d'un POI générique"""
    if re.search('accomodation', categories):
        return 480 # 8h en minutes
    return 60 # 1h en minutes

def add_visit_duration(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pois['visit_duration'] = pois['categories'].apply(select_visit_type)
    log.info("Silver => Gold (DATATOURISME) : Durées de visite ajoutées aux POIs")
    return pois

pois = dt_transform_gold()

pois = dt_add_open_hour_mask(pois)

# pois['opening_mask'] = pois['categories'].apply(dt_select_opening_mask_type)

pois['bool_restau'] = pois['categories'].apply(bool_restau)

print(pois[pois['bool_restau']].head())
print(pois[~pois['bool_restau']].head())