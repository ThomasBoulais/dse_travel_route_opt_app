
# l'objectif est de fusionner les points qui sont similaires
# 1. récupérer la structure des 2 

import geopandas as gpd
import fastparquet
from shapely import wkb

from travel_route_optimization.data_pipeline.utils.config import DT_SILVER_GEOPARQUET, OSM_SILVER_GEOPARQUET

# DATATOURISME

df = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET)
df = df.to_pandas()

def convert_wkb_to_geom(wkb_bytes): # obligé de passer par là vu que `gpd.read_parquet()` renvoie une erreur
    """Convertit les WKB (byterarray) en shapely.geometry"""
    try:
        return wkb.loads(wkb_bytes)
    except Exception as e:
        return None


df['geometry'] = df['geometry'].apply(convert_wkb_to_geom)

print(df.shape)

# supprime les nuls
df = df[df['geometry'].notnull()]

# convertit le df en gdf
gdf = gpd.GeoDataFrame(df, geometry='geometry')

print(gdf.shape)
print(gdf.columns)
print(gdf[['name_fr', 'geometry']].head(20))