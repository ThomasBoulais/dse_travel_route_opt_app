import fastparquet
import geopandas as gpd
import folium

from testing_grounds.exp2 import to_geopandas
from travel_route_optimization.data_pipeline.utils.config import DT_SILVER_GEOPARQUET, OSM_SILVER_GEOPARQUET

# Load your slim POI GeoParquet
osm_pois = gpd.read_parquet(OSM_SILVER_GEOPARQUET)

# Make sure we're in WGS84 (latitude/longitude) — Folium expects this
osm_pois = osm_pois.to_crs("EPSG:4326")

# Drop rows with no name and no geometry (can't plot them)
osm_pois = osm_pois.dropna(subset=["geometry"])

# --- Compute map center from the data itself ---
center_lat = osm_pois.geometry.y.mean()
center_lon = osm_pois.geometry.x.mean()

# --- Create the base map ---
# zoom_start: 6=country, 10=city area, 14=street level
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
# m = folium.Map(location=[43.6163376, 3.1575542], zoom_start=17)

# --- Add each POI as a marker ---
# We iterate over a sample first (plotting 23k markers will freeze your browser)
sample = osm_pois.sample(n=500, random_state=42)
# osm_pois = osm_pois.cx[3.151:3.163, 43.608:43.625]

osm_layer = folium.FeatureGroup(name="OSM")
for _, row in osm_pois.iterrows():
    name = row.get("name", "Unknown")
    category = row.get("tourism") or row.get("amenity") or row.get("leisure") or "other"
    lat = row.geometry.y
    lon = row.geometry.x

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color="steelblue",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"{name} ({category}) - [{lat}, {lon}]",   # shown on hover
        popup=folium.Popup(f"<b>{name}</b><br>Type: {category}<br>[{lat}, {lon}]", max_width=200)
    ).add_to(osm_layer)


# ----------------

# Load your slim POI GeoParquet
dt_pois = fastparquet.ParquetFile(DT_SILVER_GEOPARQUET)
dt_pois = dt_pois.to_pandas()
dt_pois = to_geopandas(dt_pois)
dt_pois.set_crs("EPSG:2154", inplace=True)

# Make sure we're in WGS84 (latitude/longitude) — Folium expects this
dt_pois = dt_pois.to_crs("EPSG:2154")

# Drop rows with no name and no geometry (can't plot them)
dt_pois = dt_pois.dropna(subset=["geometry"])

# --- Add each POI as a marker ---
# We iterate over a sample first (plotting 23k markers will freeze your browser)
sample = dt_pois.sample(n=500, random_state=42)
# dt_pois = dt_pois.cx[3.151:3.163, 43.608:43.625]

dt_layer = folium.FeatureGroup(name="DATATourisme")
for _, row in dt_pois.iterrows():
    name = row.get("name_fr", "Unknown")
    category = row.get("tourism") or row.get("amenity") or row.get("leisure") or "other"
    lat = row.geometry.y
    lon = row.geometry.x

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color="red",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"{name} ({category}) - [{lat}, {lon}]",   # shown on hover
        popup=folium.Popup(f"<b>{name}</b><br>Type: {category}<br>[{lat}, {lon}]", max_width=200)
    ).add_to(dt_layer)


osm_layer.add_to(m)
dt_layer.add_to(m)

folium.LayerControl().add_to(m)

# Save and open in browser
m.save("visualization\map_osm_dt_pois.html")
print("Saved visualization\map_osm_dt_pois.html — open it in your browser!")