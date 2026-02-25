import osmnx as ox
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

G_sub = ox.truncate.truncate_graph_bbox(
    ox.load_graphml("data/silver/osm_road_network.graphml"),
    bbox=(3.80, 43.55, 3.95, 43.65)
)
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_sub)
pois = gpd.read_parquet("data/silver/osm_pois_slim.geoparquet").to_crs("EPSG:4326")

# Filter POIs to the same bounding box
pois_local = pois.cx[3.80:3.95, 43.55:43.65]  # .cx is GeoPandas spatial indexer

center = [43.60, 3.875]
m = folium.Map(location=center, zoom_start=14)

# Layer 1: Road network
road_layer = folium.FeatureGroup(name="Road Network")
for _, edge in edges_gdf.iterrows():
    coords = [(lat, lon) for lon, lat in edge.geometry.coords]
    # Note: Shapely gives (lon, lat), Folium wants (lat, lon) — swap!
    folium.PolyLine(
        locations=coords,
        color="gray",
        weight=1.5,
        opacity=0.6
    ).add_to(road_layer)

for _, node in nodes_gdf.iterrows():
    folium.CircleMarker(
        location=[node.geometry.y, node.geometry.x],
        radius=2,
        color="red",
        fill=True
    ).add_to(road_layer)

road_layer.add_to(m)

# Layer 2: POIs
poi_layer = folium.FeatureGroup(name="Points of Interest")
for _, row in pois_local.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        color="steelblue",
        fill=True,
        tooltip=str(row.get("name", "?"))
    ).add_to(poi_layer)
poi_layer.add_to(m)

# Layer control: toggle layers on/off in the browser
folium.LayerControl().add_to(m)

m.save("map_combined.html")