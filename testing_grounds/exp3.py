import pprint
import fastparquet
import geopandas as gpd

from testing_grounds.exp2 import to_geopandas
from travel_route_optimization.data_pipeline.utils.config import DT_DICT_TYPE_CAT, DT_SILVER_GEOPARQUET

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


df_dt = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET)
df_dt = df_dt.to_pandas()
gdf_dt = to_geopandas(df_dt)

for i, row in add_category(gdf_dt)[['name_fr', 'types', 'categories']].head(50).iterrows():
    print(row['name_fr'], row['categories'], '\n', row['types'], '\n\n')

# reste à créer la strctuure standardisée pour toutes les données

