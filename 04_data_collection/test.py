import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd

# pq.read_table(r'data/silver/osm_pois.parquet', columns=['one', 'three'])
# table = pq.read_table(r'data/silver/osm_pois.geoparquet')

# i = 10
# print(str(table.to_pandas()['name'].head(i)))
# print(str(table.to_pandas()['tourism'].head(i)))
# print(str(table.to_pandas()['amenity'].head(i)))
# print(str(table.to_pandas()['geometry'].head(i)))

# print((table.to_pandas()['geometry'].iloc[0]))
# print(table.to_pandas().columns)

# parquet_file = pq.ParquetFile('data/silver/osm_pois.geoparquet')

# print(parquet_file.metadata)

# print(parquet_file.schema)

#  pour lire les geometry il faut passer par geopandas, sinon le format est un bytearray
pois = gpd.read_parquet("data/silver/osm_pois.geoparquet")
print(type(pois))           # <class 'geopandas.geodataframe.GeoDataFrame'>
print(pois.geometry.head()) # POINT (2.64289 43.18521) ...