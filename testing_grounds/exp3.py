import pprint
import fastparquet
import geopandas as gpd

from rapidfuzz import process, fuzz, utils
# https://github.com/rapidfuzz/RapidFuzz


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

# # Transfo des types en catégories côté DATATourisme
# for i, row in add_category(gdt_df)[['name_fr', 'types', 'categories']].head(50).iterrows():
#     print(row['name_fr'], row['categories'], '\n', row['types'], '\n\n')

# reste à créer la strctuure standardisée pour toutes les données
# name + geometry + category + opening hours + website + phone + address

osm_df = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET)
osm_df = osm_df.to_pandas()
osm_gdf = to_geopandas(osm_df)

print('\nDT\n', dt_gdf.notnull().sum())
print('\nOSM\n', osm_gdf.notnull().sum())

# print(osm_gdf[osm_gdf['natural'].notnull()].head(10))

# MERGE PROPOSITION 1 : utilisation de gpd_sjoin_nearest avec une approx à 50m
# pb: pas de full outer possible, juste left ou inner + pas de multicritères sur l'égalité des rows

# Re-projection métrique en CRS (Coordinate Reference System) pour le calcul des distances
dt_gdf.set_crs("EPSG:2154", inplace=True)
osm_gdf.set_crs("EPSG:2154", inplace=True)

dt_m = dt_gdf.to_crs("EPSG:2154") # EPSG:2154 => projection métrique française officielle
osm_m = osm_gdf.to_crs("EPSG:2154")

# merged = gpd.sjoin_nearest(
#     dt_m,
#     osm_m,
#     how = 'left', # les données DT (left) sont enrichies des données OSM (right) si elles existent
#     max_distance = 5, # en mètres
#     distance_col = "match_distance"
# )

# print('\n',merged[merged['id_right'].isnull()].notnull().sum()) # vérif du nb d'él sans correspondance

# merged = gpd.sjoin_nearest(
#     dt_m,
#     osm_m,
#     how = 'right', # les données OSM (right) sont enrichies des données DT (left) si elles existent
#     max_distance = 5, # en mètres
#     distance_col = "match_distance"
# )

# print('\n',merged[merged['id_left'].isnull()].notnull().sum()) # vérif du nb d'élt sans correspondance

# MERGE PROPOSITION 2 : création d'une colonne d'égalité dans DT avec approche multi critères (approx. nom + géom.)
# todo: 
# - fonction d'affectation à créer
# - split clair (inter DT OSM + DT uniq + OSM uniq) pour concat
# - purge noms vides + normalisation category pour OSM

# limiter le sampling sur la ville de bédarieux (trouver bornes coordonnées)

# osm_pois = osm_m.cx[3.151:3.163, 43.608:43.625]
# dt_pois = dt_m.cx[3.151:3.163, 43.608:43.625]

# print(osm_pois.shape, osm_pois[['name', 'geometry']].head(10), '\n')
# print(dt_pois.shape, dt_pois[['name_fr', 'geometry']].head(10))

def get_id_equivalent(dt_row: gpd.GeoSeries, osm_gdf: gpd.GeoDataFrame) -> int|None:
    """Renvoie l'index OSM du POI équivalent côté DATATourisme"""

    # 1. on veut filtrer sur les résultats proches (~20m) et garder tous les candidats,
    delta  = .0002 # .0001 => 11.132 m (https://www.tuto-carto.fr/longitude-latitude-precision/)
    osm_nearest = osm_m.cx[dt_row['geometry'].x - delta : dt_row['geometry'].x + delta, 
                           dt_row['geometry'].y - delta : dt_row['geometry'].y + delta]
    
    # print('\ndt_row\n', dt_row[['name_fr', 'geometry']])
    # print('\nosm_nearest\n', osm_nearest[['name', 'geometry']].head())

    # 2. puis déterminer un score de ressemblance des noms sur les candidats restants
    osm_nearest['lev_score'] = osm_nearest['name'].apply(
        lambda x: fuzz.partial_ratio(str(x), str(dt_row['name_fr']), processor=utils.default_process)
        )

    # print(osm_nearest[['name', 'geometry', 'lev_score']])

    # 3. et enfin renvoyer une valeur si lev_score > 75%
    try:
        return osm_nearest[osm_nearest['lev_score'] > 75].head(1).index[0][1]
    except IndexError:
        return None


dt_m['osm_match_id'] = dt_m.apply(get_id_equivalent, args=(osm_m,), axis=1)

print(dt_m.notnull().sum())

print(dt_m[dt_m['osm_match_id'].notnull()][['name_fr', 'geometry', 'osm_match_id']])