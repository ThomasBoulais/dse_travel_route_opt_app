import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
# 1. merge OSM & DT into gold 
# 1. get some datapoints from gold 

# geom = Polygon([Point(2,4), Point(4,2), Point(2,2), Point(4,4)])

# print(geom.centroid)

# df = pd.DataFrame(
#     columns=['name', 'longitude', 'latitude'],
#     data=[['eiffel',4, 2],
#           ['kebab', 5, 5]], 
#     )
# print(df)
# geometry = [Point(row.longitude, row.latitude) for row in df.itertuples()]
    
# gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# print(gdf)

# import pyarrow.parquet as pq
# pq.read_table('04_data_collection/data/silver/datatourisme_pois.geoparquet')

# gdf = gpd.read_parquet(r'04_data_collection/data/silver/datatourisme_pois.geoparquet') # renvoie une erreur (certainement un pb d'archi)

# print(gdf.head(2))


import geopandas as gpd
import fastparquet
from shapely import wkb

# Load the Parquet file using fastparquet
df = fastparquet.ParquetFile('04_data_collection/data/silver/datatourisme_pois.geoparquet')
df = df.to_pandas()

# Convert WKB (bytearray) into Shapely geometries
def convert_wkb_to_geom(wkb_bytes):
    try:
        return wkb.loads(wkb_bytes)
    except Exception as e:
        return None  # Return None for invalid geometries

# Apply the function to the 'geometry' column (assuming the 'geometry' column contains bytearray)
df['geometry'] = df['geometry'].apply(convert_wkb_to_geom)

# Remove rows with invalid geometries (None)
df = df[df['geometry'].notnull()]

# Convert the DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Now you can use the GeoDataFrame
print(gdf.columns)
print(gdf[['name_fr', 'geometry']].head(20))
# print(gdf[gdf['geometry'].geom_type != "Point"])