import osmnx as ox
import folium

# Load the full graph
G = ox.load_graphml("data/silver/osm_road_network.graphml")

# --- Option A: Render a small geographic sub-area ---
# Get the bounding box of Montpellier and filter the graph
G_montpellier = ox.truncate.truncate_graph_bbox(
    G,
    # bbox=(43.65, 43.55, 3.95, 3.80)  # (north, south, east, west)
    bbox=(3.80, 43.55, 3.95, 43.65)    # (left, bottom, right, up)
)

print(f"Sub-graph: {len(G_montpellier.nodes)} nodes, {len(G_montpellier.edges)} edges")

# Convert graph nodes and edges to GeoDataFrames
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_montpellier)

# --- Build the Folium map ---
center = [nodes_gdf.geometry.y.mean(), nodes_gdf.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=14)

# Draw road edges as polylines
# edges_gdf has a 'geometry' column with LineString geometries
for _, edge in edges_gdf.iterrows():
    coords = [(lat, lon) for lon, lat in edge.geometry.coords]
    # Note: Shapely gives (lon, lat), Folium wants (lat, lon) — swap!
    folium.PolyLine(
        locations=coords,
        color="gray",
        weight=1.5,
        opacity=0.6
    ).add_to(m)

# Draw nodes as small dots
for _, node in nodes_gdf.iterrows():
    folium.CircleMarker(
        location=[node.geometry.y, node.geometry.x],
        radius=2,
        color="red",
        fill=True
    ).add_to(m)

m.save("map_network.html")
print("Saved map_network.html")