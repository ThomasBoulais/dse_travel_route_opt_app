import json

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import osmnx as ox
import requests

from travel_route_optimization.utils.config import GOLD_POIS_GEOPARQUET, SILVER_DRIVE_GRAPHML

# =====================================================================
# PAGE CONFIGURATION — must be the first Streamlit call
# =====================================================================
st.set_page_config(
    page_title="Travel Route Explorer",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Travel Route Explorer")
st.markdown("Explore points of interest and road networks.")

# =====================================================================
# DATA LOADING
# Use @st.cache_data to avoid reloading files on every interaction.
# This is CRITICAL for performance — Streamlit reruns the entire script
# on every user interaction, so caching prevents re-reading large files.
# =====================================================================

# @st.cache_data
def load_pois():
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    return pois.to_crs("EPSG:4326")

# @st.cache_resource  # cache_resource for non-serializable objects like graphs
def load_graph():
    return ox.load_graphml(SILVER_DRIVE_GRAPHML)

pois = load_pois()
# G = load_graph()

# =====================================================================
# SIDEBAR — user controls
# =====================================================================
st.sidebar.header("Filters")

city = st.sidebar.text_input("City", value="Salagou")
nb_jour = st.sidebar.text_input("Nb jours", value=3)
start_poi = st.sidebar.text_input("ID start POI", value=0)
start_day = st.sidebar.text_input("Start day (0= lundi)", value=0)
radius_km = st.sidebar.slider("Radius (km)", min_value=1, max_value=20, value=5)
show_network = st.sidebar.checkbox("Show road network", value=True)
show_pois = st.sidebar.checkbox("Show POIs", value=True)

poi_types = st.sidebar.multiselect(
    "POI types to show",
    options=["museum", "restaurant", "cafe", "hotel", "attraction", "viewpoint"],
    default=[]
)

url = 'http://localhost:8000/itinerary'
myobj = {
  "start_poi": start_poi,
  "start_day": start_day,
  "num_days": nb_jour,
  "model_name": "4287957a48224b1c97cbf3e610c6aaa0",
  "config_path": "config.yaml"
}

@st.cache_data
def geocode_city(city_name: str):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="dse_travel_route_opt_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return 43.6502211, 3.3741647 # fallback to lac du Salagou
    # return 43.6, 3.88  # fallback to Montpellier

if st.sidebar.button("Générer itinéraire", type="primary"):
    x = requests.post(url, json=myobj)
    x = json.loads(x.text)

l_poi_idx = [item['poi_idx'] for item in x]
st.write(l_poi_idx)

# =====================================================================
# GEOCODE THE CITY NAME TO GET COORDINATES
# =====================================================================


center_lat, center_lon = geocode_city(city)
pois_filtered = pois.loc[l_poi_idx]

# # =====================================================================
# # FILTER DATA
# # =====================================================================
# # Filter POIs by type and bounding box
delta = radius_km / 1.11  # rough conversion: 1 degree ≈ 111 km
# pois_filtered = pois.cx[
#     center_lon - delta : center_lon + delta,
#     center_lat - delta : center_lat + delta
# ]

# if poi_types:
#     mask = pois_filtered["main_category"].isin(poi_types)
#     pois_filtered = pois_filtered[mask]

# # Filter graph to bounding box
# try:
#     G_sub = ox.truncate.truncate_graph_bbox(
#         G,
#         bbox=[
#             center_lat - delta, # left
#             center_lon - delta, # bottom
#             center_lat + delta, # right
#             center_lon + delta, # top
#         ]
#     )
# except ValueError:
#     G_sub = G

# nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_sub)

# =====================================================================
# BUILD THE FOLIUM MAP
# =====================================================================
center = [pois_filtered.geometry.y.mean(), pois_filtered.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=13)

# if show_network:
#     road_layer = folium.FeatureGroup(name="Road Network", show=True)
#     for _, edge in edges_gdf.iterrows():
#         coords = [(lat, lon) for lon, lat in edge.geometry.coords]
#         folium.PolyLine(coords, color="#888888", weight=1.5, opacity=0.5).add_to(road_layer)
#     road_layer.add_to(m)

if show_pois and not pois_filtered.empty:
    poi_layer = folium.FeatureGroup(name="POIs", show=True)
    for _, row in pois_filtered.iterrows():
        name = str(row.get("name", "Unknown"))
        category = row.get("main_category") or "?"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color="steelblue",
            fill=True,
            fill_opacity=0.8,
            icon=folium.Icon(color='white',
                               prefix='fa',icon='1'),
            tooltip=f"{name} ({category})",
            popup=folium.Popup(f"<b>{name}</b><br>{category}", max_width=200)
        ).add_to(poi_layer)
    poi_layer.add_to(m)

folium.LayerControl().add_to(m)

# =====================================================================
# RENDER IN STREAMLIT
# =====================================================================
col1, col2 = st.columns([3, 1])  # 3/4 map, 1/4 info panel

with col1:
    st.subheader(f"Map around {city}")
    # st_folium renders the Folium map and returns click/interaction data
    map_data = st_folium(m, width=900, height=600)
    st.subheader("Summary")
    st.metric("POIs shown", len(pois_filtered))

with col2:
    for poi in pois_filtered.iterrows():
        st.write(poi[['name', 'main_category', 'opening_hours', 'email', 'telephone']])
    # st.metric("Road nodes", len(nodes_gdf))
    # st.metric("Road edges", len(edges_gdf))

    # Show clicked POI details
    if map_data and map_data.get("last_object_clicked_tooltip"):
        st.info(f"Selected: {map_data['last_object_clicked_tooltip']}")