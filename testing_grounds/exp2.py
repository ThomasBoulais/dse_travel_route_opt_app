import fastparquet
import geopandas as gpd
import pandas as pd
from shapely import wkb
from travel_route_optimization.data_pipeline.utils.config import DT_SILVER_GEOPARQUET, OSM_SILVER_GEOPARQUET

# HELPERS

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


def get_gdf(geoparquet_path: str) -> gpd.GeoDataFrame:
    df = fastparquet.ParquetFile(geoparquet_path)
    df = df.to_pandas()
    gdf = to_geopandas(df)

    return gdf


def describe_gdf(gdf: gpd.GeoDataFrame) -> None:
    
    print(gdf.shape)
    print(gdf.columns)
    print(gdf.notna().sum())
    try:
        print(gdf[['name', 'geometry']].head(5))
    except KeyError:
        print(gdf[['name_fr', 'geometry']].head(5))
    try:
        print(gdf['types'].unique()) # besoin de transformer en liste
    except KeyError:
        pass

# DATATOURISME

gdf_dt = get_gdf(DT_SILVER_GEOPARQUET)
# describe_gdf(gdf_dt)
    # Index(['id', 'dc_identifier', 'source_file', 'name_fr', 'name_en',
    #        'description_fr', 'types', 'latitude', 'longitude', 'street',
    #        'postal_code', 'city', 'city_insee', 'dept_insee', 'email', 'phone',
    #        'website', 'opening_hours', 'allowed_persons', 'creation_date',
    #        'last_update', 'language', 'geometry'],
    #       dtype='str')

# ls_types = set()

# for item in gdf_dt['types'].unique():
#     types = item.split('|')
#     for el_types in types:
#         ls_types.add(el_types)

# ls_schema = [str(x) for x in ls_types if 'schema' in x]
# # les entrées précédées de "schema:" (schema.org = convention internationale de nommage)
# print(sorted(ls_schema)) 
# print('---')
# print(sorted(ls_types.difference(ls_schema)))
# print('---')
# print('---')

# # OSM 

# gdf_osm = get_gdf(OSM_SILVER_GEOPARQUET)
# describe_gdf(gdf_osm)

# print(gdf_osm.head(30))
