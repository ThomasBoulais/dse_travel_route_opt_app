

import geopandas as gpd
from pprint import pprint

from travel_route_optimization.data_pipeline.utils.config import DT_SILVER_GEOPARQUET
from travel_route_optimization.data_pipeline.utils.pipeline_helpers import HORAIRE_GENERIQUE, dt_add_open_hour_mask


pois = gpd.read_parquet(DT_SILVER_GEOPARQUET)

pois = dt_add_open_hour_mask(pois)

print(pois["opening_mask"].head())