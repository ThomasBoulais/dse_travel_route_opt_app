import json

import fastparquet
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import folium.plugins as plugins
import osmnx as ox
import networkx as nx

from travel_route_optimization.utils.config import DEFAULT_CRS, GOLD_DRIVE_GRAPHML, GOLD_POIS_GEOPARQUET
from travel_route_optimization.utils.pipeline_helpers import to_geopandas

# ----------------- CONFIG -----------------
ITINERARY_URL = "http://localhost:8000/itinerary"

st.set_page_config(
    page_title="Travel Route Optimization",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Travel Route Optimization")
st.markdown("Vision des POIs & routes disponibles pour des itinéraires de voyage.")

# ----------------- DATA LOADERS -----------------
@st.cache_data
def load_pois():
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    return pois.to_crs(DEFAULT_CRS)


@st.cache_data
def load_drive_graph():
    G = ox.load_graphml(GOLD_DRIVE_GRAPHML)
    if G.graph.get("crs") and G.graph["crs"] != DEFAULT_CRS:
        G = ox.project_graph(G, to_crs=DEFAULT_CRS)
    return G


pois = load_pois()
G = load_drive_graph()

# ----------------- SIDEBAR -----------------
st.sidebar.header("Filtres")

nb_jour = st.sidebar.text_input("Nb jours", value="3")
start_poi = st.sidebar.text_input("ID start POI", value="0")
start_day = st.sidebar.text_input("Jour de départ (0 = lundi)", value="0")

request_body = {
    "start_poi": int(start_poi),
    "start_day": int(start_day),
    "num_days": int(nb_jour),
    "model_name": "4287957a48224b1c97cbf3e610c6aaa0",
    "config_path": "config.yaml",
}

if "itinerary" not in st.session_state:
    x = requests.post(ITINERARY_URL, json=request_body)
    st.session_state.itinerary = json.loads(x.text)

if st.sidebar.button("Générer Itinéraire !", type="primary"):
    x = requests.post(ITINERARY_URL, json=request_body)
    st.session_state.itinerary = json.loads(x.text)

itinerary = st.session_state.itinerary

# ----------------- FILTER POIS -----------------
l_poi_idx = [item["poi_idx"] for item in itinerary]
pois_filtered = pois.loc[pois.index.isin(l_poi_idx)].copy()

# Garde l'ordre d'itinéraire des POIs
order_map = {poi["poi_idx"]: i for i, poi in enumerate(itinerary)}
pois_filtered["order"] = pois_filtered.index.map(order_map)
pois_filtered = pois_filtered.sort_values("order")

# ----------------- BUILD MAP -----------------
try:
    center = [pois_filtered.geometry.y.mean(), pois_filtered.geometry.x.mean()]
    bounds = [
        [pois_filtered.geometry.y.min(), pois_filtered.geometry.x.min()],
        [pois_filtered.geometry.y.max(), pois_filtered.geometry.x.max()],
    ]

    m = folium.Map(location=center, zoom_start=13)
    m.fit_bounds(bounds, padding=(50, 50))

    # ---- POI MARKERS ----
    poi_layer = folium.FeatureGroup(name="POIs", show=True)
    for i, (_, row) in enumerate(pois_filtered.iterrows(), start=1):
        name = row.get("name", "Unknown")
        category = row.get("main_category") or "?"
        oh = row.get("opening_hours", "")
        em = row.get("email", "")
        te = row.get("telephone", "")
        ws = row.get("website", "")

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(
                f"<h4>{i}. {name}</h4><ul>"
                f"<li><b>Catégorie:</b> {category}</li>"
                f"<li><b>Horaires:</b> {oh}</li>"
                f"<li><b>Email:</b> {em}</li>"
                f"<li><b>Téléphone:</b> {te}</li>"
                f"<li><b>Website:</b> {ws}</li></ul>",
                max_width=350,
            ),
            tooltip=f"{i}. {name} ({category})",
            icon=plugins.BeautifyIcon(
                icon="arrow-down",
                icon_shape="marker",
                number=i,
                background_color="#F54927",
                border_color="#F54927",
                text_color="#000000",
            ),
        ).add_to(poi_layer)

    poi_layer.add_to(m)

    # ---- ROAD EDGES FOR ITINERARY ----
    route_layer = folium.FeatureGroup(name="Route", show=True)

    # Récup le node le plus proche de chaque POI
    poi_nodes = {}
    for _, row in pois_filtered.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        node = ox.distance.nearest_nodes(G, lon, lat)
        poi_nodes[row.name] = node

    # Récup les nodes d'origine et de destinatin de chaque étape
    ordered_indices = list(pois_filtered.index)
    for i in range(len(ordered_indices) - 1):
        idx_from = ordered_indices[i]
        idx_to = ordered_indices[i + 1]
        u = poi_nodes[idx_from]
        v = poi_nodes[idx_to]
        # Puis crée le chemin le plus court entre chaque paire de nodes
        try:
            path = nx.shortest_path(G, u, v, weight="length")
        except nx.NetworkXNoPath:
            continue

        # Récup les coordonnées du chemin
        lats = [G.nodes[n]["y"] for n in path]
        lons = [G.nodes[n]["x"] for n in path]
        coords = list(zip(lats, lons))

        folium.PolyLine(
            locations=coords,
            color="#F54927",
            weight=4,
            opacity=0.7,
        ).add_to(route_layer)

    route_layer.add_to(m)
    folium.LayerControl().add_to(m)

except ValueError:
    m = folium.Map(location=[43.6502211, 3.3741647], zoom_start=11)

# ----------------- LAYOUT -----------------
col1, col2 = st.columns([3, 1])

with col1:
    map_data = st_folium(m, width=900, height=600)
    st.title("Résumé")

    total_travel_time = sum([float(x['travel_time']) for x in itinerary])
    st.metric("Durée totale du voyage", f"{round(total_travel_time/60)}h {round(total_travel_time%60)}m")

with col2:
    st.subheader("Itinéraire brut")
    for i, row in enumerate(itinerary, start=1):
        st.markdown(f"**Étape {i}**")
        st.json(row)
