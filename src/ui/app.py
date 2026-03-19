import json
import os
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import folium.plugins as plugins
import osmnx as ox
import networkx as nx

# from src.utils.config import DEFAULT_CRS, GOLD_DRIVE_GRAPHML, GOLD_POIS_GEOPARQUET
from src.common.config_loader import load_config

cfg = load_config()

# ----------------- CONFIG -----------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
ITINERARY_URL = f"{API_URL}/itinerary"

st.write("API_URL =", API_URL)
st.write("ITINERARY_URL =", ITINERARY_URL)

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
    pois = gpd.read_parquet(cfg.gold.pois_geoparquet)
    pois = pois.to_crs(cfg.crs.default)
    # st.table(pois[pois.index==0])
    return pois.to_crs(cfg.crs.default)


@st.cache_data
def load_drive_graph():
    G = ox.load_graphml(cfg.gold.drive_graphml)
    if G.graph.get("crs") and G.graph["crs"] != cfg.crs.default:
        G = ox.project_graph(G, to_crs=cfg.crs.default)
    return G


pois = load_pois()
G = load_drive_graph()

# ----------------- SIDEBAR -----------------
st.sidebar.header("Filtres")

nb_jour = st.sidebar.text_input("Nb jours", value="3")
start_poi = st.sidebar.text_input("ID start POI", value="100")
start_day = st.sidebar.text_input("Jour de départ (0 = lundi)", value="0")

request_body = {
    "start_poi": int(start_poi),
    "start_day": int(start_day),
    "num_days": int(nb_jour),
    "model_name": "tdtoptw_dqn",
    "config_path": "training.yaml",
}

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# 1. Initialize session state
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# 2. Sidebar button
generate = st.sidebar.button("Générer Itinéraire !", type="primary")

# 3. If button clicked, call API ONCE
if generate:
    try:
        x = requests.post(ITINERARY_URL, json=request_body, timeout=10)
        x.raise_for_status()
        st.session_state.itinerary = x.json()
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        st.stop()

# 4. If no itinerary yet, stop here
if st.session_state.itinerary is None:
    st.info("Cliquez sur le bouton pour générer un itinéraire.")
    st.stop()

# 5. From here on, itinerary is guaranteed to exist
itinerary = st.session_state.itinerary


# st.table(itinerary)

# ----------------- FILTER POIS -----------------
l_poi_idx = [int(start_poi)] + [item["poi_idx"] for item in itinerary]
# st.write(l_poi_idx)
pois_filtered = pois.loc[pois.index.isin(l_poi_idx)].copy()
# st.table(pois_filtered)

# Garde l'ordre d'itinéraire des POIs
order_map = {poi["poi_idx"]: i for i, poi in enumerate(itinerary)}
order_map[int(start_poi)] = 0
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
    for i, (_, row) in enumerate(pois_filtered.iterrows(), start=0):
        name = row.get("name", "None")
        category = row.get("main_category") or "?"
        oh = row.get("opening_hours", "None")
        em = row.get("email", "None")
        te = row.get("telephone", "None")
        ws = row.get("website", "None")

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
    st.markdown(f"""**Étape 0**  
 | Nom | **{pois_filtered.loc[int(start_poi)]['name']}**  
 | :-- | :--
""")
    for i, row in enumerate(itinerary, start=1):
        st.markdown(f"**Étape {i}**")
        # st.write(row)
        st.markdown(f"""
 | Nom | **{row['poi_name']}**  
 | :-- | :--
 | Temps de trajet (voiture) | **{int(row['travel_time']/60)}h {int(row['travel_time']%60)}m**  
 | Heure d'arrivée | **{int(row['arrival_minute']/60)}h {int(row['arrival_minute']%60)}m**  
 | Durée recommandée | **{int(row['visit_duration']/60)}h {int(row['visit_duration']%60)}m**
""")


