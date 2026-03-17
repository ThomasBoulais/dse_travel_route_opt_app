import json

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import osmnx as ox
import requests
import folium.plugins as plugins

from travel_route_optimization.data_pipeline.utils.config import GOLD_POIS_GEOPARQUET, SILVER_DRIVE_GRAPHML


st.set_page_config(
    page_title="Travel Route Optimization",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Travel Route Optimization")
st.markdown("Vision des POIs & routes disponibles pour des itinéraires de voyage.")

@st.cache_data
def load_pois():
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    return pois.to_crs("EPSG:4326")
 

pois = load_pois()

st.sidebar.header("Filtres")

nb_jour = st.sidebar.text_input("Nb jours", value=3)
start_poi = st.sidebar.text_input("ID start POI", value=0)
start_day = st.sidebar.text_input("Jour de départ (0 = lundi)", value=0)

url = 'http://localhost:8000/itinerary'
myobj = {
  "start_poi": start_poi,
  "start_day": start_day,
  "num_days": nb_jour,
  "model_name": "4287957a48224b1c97cbf3e610c6aaa0",
  "config_path": "config.yaml"
}

if "x" not in st.session_state:
    x = requests.post(url, json=myobj)
    st.session_state.x = json.loads(x.text)

 
if st.sidebar.button("Générer Itinéraire !", type="primary"):
    x = requests.post(url, json=myobj)
    st.session_state.x = json.loads(x.text)

# if st.sidebar.button("Générer itinéraire", type="primary"):
# x = requests.post(url, json=myobj)
# x = json.loads(x.text)

l_poi_idx = [item['poi_idx'] for item in st.session_state.x]

pois_filtered = pois.loc[l_poi_idx]

try:
    center = [pois_filtered.geometry.y.mean(), pois_filtered.geometry.x.mean()]
    bounds = [[min(pois_filtered.geometry.y), min(pois_filtered.geometry.x)], 
              [max(pois_filtered.geometry.y), max(pois_filtered.geometry.x)]]
    
    m = folium.Map(location=center)
    m.fit_bounds(bounds, padding=(50, 50))

    poi_layer = folium.FeatureGroup(name="POIs", show=True)
    i = 0
    for _, row in pois_filtered.iterrows():
        i += 1
        name = str(row.get("name", "Unknown"))
        category = row.get("main_category") or "?"

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(f"<b>{i}. {name}</b><br>{category}", max_width=200),
            tooltip=f"{i}. {name} ({category})",
            icon=plugins.BeautifyIcon(
                                icon="arrow-down", icon_shape="marker",
                                number=i,
                                background_color= "#F54927",
                                border_color= "#F54927",
                                text_color= "#000000"
                            )
        ).add_to(poi_layer)
    poi_layer.add_to(m)
    folium.LayerControl().add_to(m)
except ValueError:
    m = folium.Map(location=[43.6502211, 3.3741647], zoom_start=11) # fallback aux coordonnées du Salagou


col1, col2 = st.columns([3, 1])
with col1:
    map_data = st_folium(m, width=900, height=600)

with col2:
    # st.write(l_poi_idx)
    i = 0
    for poi in pois_filtered.iterrows():
        i += 1
        res = {
            'Ordre de passage': i,
            'Nom': str(poi[1].get('name')),
            'Catégorie': poi[1].get('main_category'),
            'Heures d\'ouverture': poi[1].get('opening_hours'),
            'Email': poi[1].get('email'),
            'Téléphone': poi[1].get('telephone'),
            'Website': poi[1].get('website'),
        }
        st.table(res, border="horizontal")
