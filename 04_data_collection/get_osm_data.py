import osmnx as ox
import geopandas as gpd

ox.settings.use_cache = True

# Step 1: Définition de la zone détude
place_name = "Hérault, Occitanie, France"

# Step 2: Téléchargement de POIs avec tags spécifiques utiles pour le tourisme
tags = {
    "tourism": ["museum", "attraction", "viewpoint", "hotel", "hostel"],
    "amenity": ["restaurant", "cafe", "bar"],
    "leisure": ["park", "nature_reserve"]
}

print(f"Downloading POIs of {place_name} from OSM...")
pois_gdf = ox.features_from_place(place_name, tags=tags)

# Step 3: Inspection des résultats
print(f"Downloaded {len(pois_gdf)} POIs")
print(pois_gdf.columns.tolist())
print(pois_gdf.head())

# Step 4: Sauvegarde en GeoParquet (Bronze)
pois_gdf = pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
pois_gdf.to_parquet("data/bronze/osm_pois.geoparquet")
print("Saved to data/bronze/osm_pois.geoparquet.")

# Step 5: Réduction aux colonnes utiles + conversion des batîments en point unique (centroïde)
slim_pois_gdf = pois_gdf[[
    "geometry",
    "name",
    "tourism",
    "amenity",
    "historic",
    "leisure",
    "natural",
    "opening_hours",
    "website",
    "phone",
    "addr:city",
    "addr:postcode",
    "wheelchair",
    "stars",
    "wikidata",
]].copy()
slim_pois_gdf["geometry"] = slim_pois_gdf["geometry"].apply(
    lambda geom: geom.centroid if geom.geom_type != "Point" else geom
)

# Step 6: Inspection des résultats
print(slim_pois_gdf.columns.tolist())
print(slim_pois_gdf.head())

# Step 7: Sauvegarde en  GeoParquet (passage de Bronze à Silver)
slim_pois_gdf = slim_pois_gdf.to_crs("EPSG:4326")  # ensure standard WGS84 coordinates
slim_pois_gdf.to_parquet("data/silver/osm_pois_slim.geoparquet")
print("Saved to data/silver/osm_pois_slim.geoparquet.")


# Download the drivable road network
print(f"Download drivable road network of {place_name}")
G = ox.graph_from_place(place_name, network_type="drive")

# Add travel time to each edge (requires speed data or defaults)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

# Save the graph
ox.save_graphml(G, filepath="data/bronze/osm_road_network.graphml")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
print("Saved to data/bronze/osm_road_network.graphml")

ox.save_graphml(G, filepath="data/silver/osm_road_network.graphml")
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
print("Saved to data/silver/osm_road_network.graphml")