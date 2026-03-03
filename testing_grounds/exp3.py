import pprint
import fastparquet
import geopandas as gpd

from testing_grounds.exp2 import to_geopandas
from travel_route_optimization.data_pipeline.utils.config import DT_DICT_TYPE_CAT, DT_SILVER_GEOPARQUET, OSM_SILVER_GEOPARQUET

# DATATOURISME

def convert_to_cat(types: str) -> str:
    """Ajoute les catégoires à une chaîne de caractères des types"""
    set_cat = set()
    for item in types.split('|'):
        try:
            set_cat.add(DT_DICT_TYPE_CAT[item])
        except KeyError:
            pass
    return str(list(set_cat))
 

def add_category(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte les catégories correspondantes aux types des POIs (DATATourisme)"""
    gdf['categories'] = gdf['types'].apply(convert_to_cat)
    return gdf


dt_df = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET)
dt_df = dt_df.to_pandas()
dt_gdf = to_geopandas(dt_df)

# for i, row in add_category(gdt_df)[['name_fr', 'types', 'categories']].head(50).iterrows():
#     print(row['name_fr'], row['categories'], '\n', row['types'], '\n\n')

# reste à créer la strctuure standardisée pour toutes les données

osm_df = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET)
osm_df = osm_df.to_pandas()
osm_gdf = to_geopandas(osm_df)

print('\nDT\n', dt_gdf.notnull().sum())
print('\nOSM\n', osm_gdf.notnull().sum())

# print(osm_gdf[osm_gdf['natural'].notnull()].head(10))



# Re-projection métrique en CRS (Coordinate Reference System) pour le calcul des distances
dt_gdf.set_crs("EPSG:2154", inplace=True)
osm_gdf.set_crs("EPSG:2154", inplace=True)

dt_m = dt_gdf.to_crs("EPSG:2154") # EPSG:2154 => projection métrique française officielle
osm_m = osm_gdf.to_crs("EPSG:2154")

merged = gpd.sjoin_nearest(
    dt_m,
    osm_m,
    how = 'left', # les données DT sont enrichies des données OSM si elles existent
    max_distance = 5, # en mètres
    distance_col = "match_distance"
)

print('\n',merged[merged['id_right'].isnull()].notnull().sum())

merged = gpd.sjoin_nearest(
    dt_m,
    osm_m,
    how = 'right', # les données DT sont enrichies des données OSM si elles existent
    max_distance = 5, # en mètres
    distance_col = "match_distance"
)

print('\n',merged[merged['id_left'].isnull()].notnull().sum())
# voir option pour merge (how = 'full outer' inexistant) maybe en faisant un écart et concat ?

# name + geometry + category + opening hours + website + phone + address
