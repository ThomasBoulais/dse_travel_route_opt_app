from pathlib import Path

DATA_DIR    = Path(__file__).parents[2] / "data"
BRONZE_DIR  = Path(__file__).parents[2] / "data/bronze"
SILVER_DIR  = Path(__file__).parents[2] / "data/silver"
GOLD_DIR    = Path(__file__).parents[2] / "data/gold"

OSM_PLACE_NAME              = "Hérault, Occitanie, France"
OSM_BRONZE_DIR              = BRONZE_DIR / "osm"
OSM_BRONZE_GEOPARQUET       = OSM_BRONZE_DIR / "osm_pois.geoparquet"
OSM_BRONZE_GRAPHML          = OSM_BRONZE_DIR / "osm_road_network.graphml"
OSM_SILVER_GEOPARQUET       = SILVER_DIR / "osm_pois.geoparquet"
OSM_SILVER_GRAPHML          = SILVER_DIR / "osm_road_network.graphml"

DT_BRONZE_DIR           = BRONZE_DIR / "datatourisme"
DT_DUMP_URL             = 'https://diffuseur.datatourisme.fr/webservice/e1ec2f4e53628162352a8067eb6ac3e7/071d1b42-f48c-4350-826c-e92199a99bdf'
DT_DUMP_PATH            = DT_BRONZE_DIR / "dt_dump_gz"
DT_DUMP_DIR             = DT_BRONZE_DIR / "dump"
DT_INDEX_FILE           = DT_DUMP_DIR / "index.json"
DT_SILVER_GEOPARQUET    = SILVER_DIR / "datatourisme_pois.geoparquet"
DT_SILVER_CSV           = SILVER_DIR / "datatourisme_pois.csv"   # debug/exploration
