
# l'objectif est de fusionner les points qui sont similaires
# 1. récupérer la structure des 2 

import geopandas as gpd
import fastparquet
import pandas as pd
from shapely import wkb

from travel_route_optimization.data_pipeline.utils.config import DT_SILVER_GEOPARQUET, OSM_SILVER_GEOPARQUET


def convert_wkb_to_geom(wkb_bytes: bytearray) -> gpd.GeoDataFrame | None: # obligé de passer par là vu que `gpd.read_parquet()` renvoie une erreur
    """Convertit les WKB (byterarray) en shapely.geometry"""
    try:
        return wkb.loads(wkb_bytes)
    except Exception as e:
        return None


def to_geopandas(df: pd.DataFrame) -> gpd.GeoDataFrame: 
    """
    Récupère la géométrie d'un DataFrame.
    Renvoie un GeoDataFrame.
    """
    df['geometry'] = df['geometry'].apply(convert_wkb_to_geom)
    df = df[df['geometry'].notnull()]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf


# DATATOURISME

df_dt = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET)
df_dt = df_dt.to_pandas()
gdf_dt = to_geopandas(df_dt)

# Index(['id', 'dc_identifier', 'source_file', 'name_fr', 'name_en',
#        'description_fr', 'types', 'latitude', 'longitude', 'street',
#        'postal_code', 'city', 'city_insee', 'dept_insee', 'email', 'phone',
#        'website', 'opening_hours', 'allowed_persons', 'creation_date',
#        'last_update', 'language', 'geometry'],
#       dtype='str')

print(gdf_dt.shape, '\n',
      gdf_dt.notna().sum(), '\n',
      gdf_dt[['name_fr', 'geometry']].head(5),
      gdf_dt['types'].unique()) # besoin de transformer en liste


# OSM

df_osm = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET)
df_osm = df_osm.to_pandas()
gdf_osm = to_geopandas(df_osm)

# Index(['geometry', 'name', 'tourism', 'amenity', 'historic', 'leisure',
#        'natural', 'opening_hours', 'website', 'phone', 'addr:city',
#        'addr:postcode', 'wheelchair', 'stars', 'wikidata'],

print(gdf_osm.shape, '\n',
      gdf_osm.notna().sum(), '\n',
      gdf_osm[['name', 'geometry']].head(5))