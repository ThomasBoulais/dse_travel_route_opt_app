import geopandas as gpd
import folium

from travel_route_optimization.data_pipeline.utils.config import DEFAULT_CRS

# Load your slim POI GeoParquet
pois = gpd.read_parquet("data/silver/osm_pois_slim.geoparquet")

# Make sure we're in WGS84 (latitude/longitude) — Folium expects this
pois = pois.to_crs(DEFAULT_CRS)

# Drop rows with no name and no geometry (can't plot them)
pois = pois.dropna(subset=["geometry"])

# --- Compute map center from the data itself ---
center_lat = pois.geometry.y.mean()
center_lon = pois.geometry.x.mean()

# --- Create the base map ---
# zoom_start: 6=country, 10=city area, 14=street level
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# --- Add each POI as a marker ---
# We iterate over a sample first (plotting 23k markers will freeze your browser)
sample = pois.sample(n=500, random_state=42)

for _, row in sample.iterrows():
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
        tooltip=f"{name} ({category})",   # shown on hover
        popup=folium.Popup(f"<b>{name}</b><br>Type: {category}", max_width=200)
    ).add_to(m)

# Save and open in browser
m.save("map_pois.html")
print("Saved map_pois.html — open it in your browser!")