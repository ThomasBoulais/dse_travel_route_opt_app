import logging
import re
import unicodedata
import networkx as nx
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.neighbors import BallTree

from travel_route_optimization.data_pipeline.utils.config import DT_DICT_TYPES_DETAILED, GOLD_DRIVE_GRAPHML, GOLD_POIS_GEOPARQUET, KNN_VALUE, OSM_DICT_TYPES_DETAILED

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============= SILVER =============

# DATATOURISME

def get_multilang(field, lang: str = "fr") -> str | None:
    """Extrait une valeur multilingue (dict ou liste de dicts)."""
    if not field:
        return None
    if isinstance(field, dict):
        return (field.get(lang) or field.get("en") or next(iter(field.values()), None))
    if isinstance(field, list):
        # Parfois c'est une liste de chaînes
        if field and isinstance(field[0], str):
            return field[0]
    return None


def extract_geo(entry: dict) -> tuple[float | None, float | None]:
    """Extrait latitude/longitude depuis isLocatedAt vers schema:geo."""
    for location in entry.get("isLocatedAt", []):
        geo = location.get("schema:geo", {})
        lat = geo.get("schema:latitude")
        lon = geo.get("schema:longitude")
        if lat and lon:
            try:
                return float(lat), float(lon)
            except (ValueError, TypeError):
                pass
    return None, None


def extract_address(entry: dict) -> dict:
    """Extrait les champs d'adresse depuis isLocatedAt."""
    for location in entry.get("isLocatedAt", []):
        for addr in location.get("schema:address", []):
            return {
                "street":      addr.get("schema:streetAddress", [None])[0]
                               if isinstance(addr.get("schema:streetAddress"), list)
                               else addr.get("schema:streetAddress"),
                "postal_code": addr.get("schema:postalCode"),
                "city":        addr.get("schema:addressLocality"),
                "city_insee":  (addr.get("hasAddressCity") or {}).get("insee"),
                "dept_insee":  (
                    (addr.get("hasAddressCity") or {})
                    .get("isPartOfDepartment", {})
                    .get("insee")
                ),
            }
    return {"street": None, "postal_code": None, "city": None,
            "city_insee": None, "dept_insee": None}


def extract_contact(entry: dict) -> dict:
    """Extrait email, téléphone, site web depuis hasContact ou hasBookingContact."""
    sources = entry.get("hasContact", []) or entry.get("hasBookingContact", [])
    email = phone = website = None
    for src in sources:
        emails = src.get("schema:email", [])
        phones = src.get("schema:telephone", [])
        websites = src.get("foaf:homepage", [])
        if not email and emails:
            email = emails[0] if isinstance(emails, list) else emails
        if not phone and phones:
            phone = phones[0] if isinstance(phones, list) else phones
        if not website and websites:
            website = websites[0] if isinstance(websites, list) else websites
    return {"email": email, "phone": phone, "website": website}


def extract_opening_hours(entry: dict) -> str | None:
    """Extrait les périodes d'ouverture sous forme lisible (ex. '2026-06-02 => 2026-09-30')."""
    periods = []
    for location in entry.get("isLocatedAt", []):
        for spec in location.get("schema:openingHoursSpecification", []):
            start = spec.get("schema:validFrom", "")[:10]
            end   = spec.get("schema:validThrough", "")[:10]
            if start or end:
                periods.append(f"{start} => {end}")
    return " | ".join(periods) if periods else None


def extract_types(entry: dict) -> str:
    """Extrait les types sémantiques (@type) filtrés des namespaces techniques."""
    skip = {"schema:Accommodation", "schema:LodgingBusiness",
            "schema:Organization", "schema:Place", "schema:GeoCoordinates",
            "schema:PostalAddress", "foaf:Agent", "Agent", "Place",
            "PostalAddress", "Description", "FeatureSpecification"}
    types = [t for t in entry.get("@type", []) if t not in skip]
    return "|".join(types)


def parse_poi(entry: dict, source_file: str) -> dict | None:
    """
    Transforme un objet JSON-LD DataTourisme en enregistrement plat.
    Retourne None si les coordonnées sont manquantes.
    """
    lat, lon = extract_geo(entry)
    if lat is None or lon is None:
        return None  # les POIs non géolocalisés sont écartés

    name_fr = get_multilang(entry.get("rdfs:label"), "fr")
    name_en = get_multilang(entry.get("rdfs:label"), "en")
    desc_fr = get_multilang(
        (entry.get("hasDescription") or [{}])[0].get("shortDescription"), "fr"
    )

    addr    = extract_address(entry)
    contact = extract_contact(entry)

    return {
        # Identifiant
        "id":              entry.get("@id"),
        "dc_identifier":   entry.get("dc:identifier"),
        "source_file":     source_file,
        # Noms / description
        "name_fr":         name_fr,
        "name_en":         name_en,
        "description_fr":  desc_fr,
        # Catégorisation
        "types":           extract_types(entry),
        # Géolocalisation
        "latitude":        lat,
        "longitude":       lon,
        # Adresse
        "street":          addr["street"],
        "postal_code":     addr["postal_code"],
        "city":            addr["city"],
        "city_insee":      addr["city_insee"],
        "dept_insee":      addr["dept_insee"],
        # Contact
        "email":           contact["email"],
        "phone":           contact["phone"],
        "website":         contact["website"],
        # Horaires
        "opening_hours":   extract_opening_hours(entry),
        # Méta
        "allowed_persons": entry.get("allowedPersons"),
        "creation_date":   entry.get("creationDate"),
        "last_update":     entry.get("lastUpdate"),
        "language":        "|".join(entry.get("availableLanguage", [])),
    }

# ============= GOLD =============

# HELPERS

def convert_wkb_to_geom(wkb_bytes: bytearray) -> gpd.GeoDataFrame | None: # obligé de passer par là vu que `gpd.read_parquet()` renvoie une erreur
    """Convertit les WKB (byterarray) en shapely.geometry"""
    try:
        return wkb.loads(wkb_bytes)
    except Exception as e:
        return None


def to_geopandas(df: pd.DataFrame) -> gpd.GeoDataFrame: 
    """
    Récupère la géométrie d'un DataFrame.
    Renvoie un GeoDataFrame.
    """
    df['geometry'] = df['geometry'].apply(convert_wkb_to_geom)
    df = df[df['geometry'].notnull()]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return gdf


def select_visit_type(categories: str) -> str:
    """Attribue le type de visit_duration selon s'il s'agit d'une accomodation ou d'un POI générique"""
    if re.search('accomodation', categories):
        return 480 # 8h en minutes
    return 60 # 1h en minutes


def add_visit_duration(pois: gpd.GeoDataFrame, src: str) -> gpd.GeoDataFrame:
    pois['visit_duration'] = pois['categories'].apply(select_visit_type)
    log.info(f"Silver => Gold ("src") : Durées de visite ajoutées aux POIs")
    return pois

# OSM

def osm_get_types(tourism: str|None, amenity: str|None, historic: str|None, leisure: str|None, natural: str|None) -> str|None:
    """Affecte les types correspondants aux POIs (OSM)"""
    set_types = set()
    set_types.add(tourism)
    set_types.add(amenity)
    set_types.add(historic)
    set_types.add(leisure)
    set_types.add(natural)
    set_types.remove(None)
    return "|".join(map(str, list(set_types)))


def osm_add_category(osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte les catégories correspondantes aux types des POIs (OSM)"""
    osm_gdf['types'] = osm_gdf.apply(lambda x: osm_get_types(x.tourism, x.amenity, x.historic, x.leisure, x.natural), axis=1)
    osm_gdf['categories'] = osm_gdf['types'].apply(summarize_types, args=(OSM_DICT_TYPES_DETAILED,))
    log.info("Silver => Gold (OSM) : Catégories ajoutées aux POIs")
    return osm_gdf


DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
INT_TYPE = np.uint8

def clean_hours(text):
    """Prépare les opening_hours pour le parsing"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[‐‑‒–—−]", "-", text)                     # dashes
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)  # spaces
    text = re.sub(r"\s+", " ", text)
    text = text.replace("H", ":").replace("h", ":")
    return text.strip()


def detect_special_cases(text):
    """Détecte les cas spéciaux (closed & 24/7)"""
    t = text.lower()
    if "closed" in t or "fermé" in t:
        return "closed"
    if "24/7" in t or "24h/24" in t:
        return "24_7"
    return None


def parse_osm_hours(text):
    """Récupère les horaires par jour depuis opening_hours"""
    results = []
    for seg in text.split(";"):
        m = re.match(r"([A-Za-z]{2})(?:-([A-Za-z]{2}))?\s+(.*)", seg.strip())
        if not m:
            continue

        d1, d2, hours = m.groups()
        if d1 not in DAYS:
            continue
        days = DAYS[DAYS.index(d1):DAYS.index(d2)+1] if d2 in DAYS else [d1]

        for h in hours.split(","):
            parts = h.strip().split("-", 1)
            if len(parts) != 2:
                continue

            start = re.sub(r"[^0-9:]", "", parts[0])
            end   = re.sub(r"[^0-9:]", "", parts[1])

            if re.match(r"^\d{1,2}:\d{2}$", start) and re.match(r"^\d{1,2}:\d{2}$", end):
                for d in days:
                    results.append({"day": d, "start": start, "end": end})
    return results


def to_minutes(hhmm):
    """Transforme une durée hh:mm en durée en minutes"""
    try:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m
    except:
        return None


def build_open_mask(parsed):
    """Construit un masque de minutes d'ouverture chaque jour (1 = ouvert, 0 = fermé)"""
    mask = np.zeros((7, 1440), dtype=INT_TYPE)
    for entry in parsed:
        d = DAYS.index(entry["day"])
        start = to_minutes(entry["start"])
        end   = to_minutes(entry["end"])

        if start is None or end is None:
            continue
        if start < end:
            mask[d, start:end] = 1
        else : # dépassement à J+1
            mask[d, start:1440] = 1
            mask[(d+1) % 7, 0:end] = 1
    return mask


HORAIRE_GENERIQUE = build_open_mask(
    [{'day': 'Mo', 'start': '00:00', 'end': '12:00'}, {'day': 'Mo', 'start': '14:00', 'end': '18:00'},
     {'day': 'Tu', 'start': '08:00', 'end': '12:00'}, {'day': 'Tu', 'start': '14:00', 'end': '18:00'},
     {'day': 'We', 'start': '08:00', 'end': '12:00'}, {'day': 'We', 'start': '14:00', 'end': '18:00'},
     {'day': 'Th', 'start': '08:00', 'end': '12:00'}, {'day': 'Th', 'start': '14:00', 'end': '18:00'},
     {'day': 'Fr', 'start': '08:00', 'end': '12:00'}, {'day': 'Fr', 'start': '14:00', 'end': '18:00'},
     {'day': 'Sa', 'start': '08:00', 'end': '12:00'}, {'day': 'Sa', 'start': '14:00', 'end': '18:00'},]
)

HORAIRE_RESTAURATION = build_open_mask(
    [{'day': 'Mo', 'start': '12:00', 'end': '15:00'}, {'day': 'Mo', 'start': '18:00', 'end': '23:00'},
     {'day': 'Tu', 'start': '12:00', 'end': '15:00'}, {'day': 'Tu', 'start': '18:00', 'end': '23:00'},
     {'day': 'We', 'start': '12:00', 'end': '15:00'}, {'day': 'We', 'start': '18:00', 'end': '23:00'},
     {'day': 'Th', 'start': '12:00', 'end': '15:00'}, {'day': 'Th', 'start': '18:00', 'end': '23:00'},
     {'day': 'Fr', 'start': '12:00', 'end': '15:00'}, {'day': 'Fr', 'start': '18:00', 'end': '23:00'},
     {'day': 'Sa', 'start': '12:00', 'end': '15:00'}, {'day': 'Sa', 'start': '18:00', 'end': '23:00'},]
)

HORAIRE_ACCOMODATION = build_open_mask(
    [{'day': 'Mo', 'start': '00:00', 'end': '11:00'}, {'day': 'Mo', 'start': '16:00', 'end': '00:00'},
     {'day': 'Tu', 'start': '00:00', 'end': '11:00'}, {'day': 'Tu', 'start': '16:00', 'end': '00:00'},
     {'day': 'We', 'start': '00:00', 'end': '11:00'}, {'day': 'We', 'start': '16:00', 'end': '00:00'},
     {'day': 'Th', 'start': '00:00', 'end': '11:00'}, {'day': 'Th', 'start': '16:00', 'end': '00:00'},
     {'day': 'Fr', 'start': '00:00', 'end': '11:00'}, {'day': 'Fr', 'start': '16:00', 'end': '00:00'},
     {'day': 'Sa', 'start': '00:00', 'end': '11:00'}, {'day': 'Sa', 'start': '16:00', 'end': '00:00'},]
)


def opening_hours_to_mask(raw):
    text = clean_hours(raw)
    special = detect_special_cases(text)

    # print(f"{text} - {special}")

    if special == "closed":
        return np.zeros((7, 1440), dtype=INT_TYPE)
    if special == "24_7":
        return np.ones((7, 1440), dtype=INT_TYPE)

    parsed = parse_osm_hours(text)
    # pprint(f"PARSED - {parsed}")
    if not parsed:
        return HORAIRE_GENERIQUE
    return build_open_mask(parsed)


def osm_add_open_hour_mask(osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte un masque d'horaires d'ouverture"""
    osm_gdf["opening_mask"] = osm_gdf["opening_hours"].apply(opening_hours_to_mask)
    log.info("Silver => Gold (OSM) : Opening mask ajoutées aux POIs")
    return osm_gdf


# DATATOURISME

def summarize_types(types: str, dict_types_detailed: dict) -> str:
    """Ajoute les catégoires à une chaîne de caractères des types"""
    set_cat = set()
    for item in types.split('|'):
        try:
            set_cat.add(dict_types_detailed[item])
        except KeyError:
            pass
    return "|".join(map(str, list(set_cat)))


def dt_add_category(dt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte les catégories correspondantes aux types des POIs (DATATourisme)"""
    dt_gdf['categories'] = dt_gdf['types'].apply(summarize_types, args=(DT_DICT_TYPES_DETAILED,))
    log.info("Silver => Gold (DATATOURISME) : Catégories ajoutées aux POIs")
    return dt_gdf


def dt_select_opening_mask_type(categories: str) -> str:
    """Attribue le type de opening_mask selon s'il s'agit d'un restau, accomodation ou POI générique"""
    if re.search('restauration', categories):
        return HORAIRE_RESTAURATION
    if re.search('accomodation', categories):
        return HORAIRE_ACCOMODATION
    return HORAIRE_GENERIQUE


def dt_add_open_hour_mask(dt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Affecte un masque d'horaires d'ouverture (vu le format en date et non horaire, toutes les horaires sont fixés en générique)"""
    dt_gdf["opening_mask"] = dt_gdf['categories'].apply(dt_select_opening_mask_type)
    log.info("Silver => Gold (DATATOURISME) : Opening mask ajoutées aux POIs")
    return dt_gdf


# GRAPHE KNN DE TRAINING POUR LE RL

def get_drive_network(left: float, bottom: float, right: float, top: float) -> gpd.GeoDataFrame:
    """Récupère le graphml du réseau de route"""
    ox.settings.default_crs = "EPSG:4326"
    G_drive = ox.load_graphml(GOLD_DRIVE_GRAPHML)
    G_drive = ox.truncate.truncate_graph_bbox(G_drive, bbox=[left, bottom, right, top])
    G_drive = ox.project_graph(G_drive)
    return G_drive


def get_pois(G_drive: gpd.GeoDataFrame, left: float, bottom: float, right: float, top: float) -> gpd.GeoDataFrame:
    """Récupère le GeoDataFrame des POIs dans le même CRS que G_drive"""
    pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)
    pois = pois.to_crs("EPSG:4326")
    pois = pois.cx[left:right, bottom:top].reset_index(drop=True)
    pois = pois.to_crs(G_drive.graph["crs"])
    return pois


def nearest_node(pois: gpd.GeoDataFrame, G_drive: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Associe à chaque POI son node le plus proche"""
    X = pois.geometry.x.values
    Y = pois.geometry.y.values
    pois["nearest_node"] = ox.nearest_nodes(G_drive, X, Y)
    return pois


def get_knn_pois(pois: gpd.GeoDataFrame) -> list[int]:
    coords = np.vstack([pois.geometry.y.values, pois.geometry.x.values]).T
    tree = BallTree(np.radians(coords), metric="haversine")
    
    distances, indices = tree.query(np.radians(coords), k=KNN_VALUE + 1)
    neighbors = [set(idx[1:]) for idx in indices]
    return neighbors


def add_travel_time(G: gpd.GeoDataFrame, speed_kmh: int) -> None:
    for u, v, data in G.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        data["travel_time"] = length / (speed_kmh * 1000 / 3600)


def travel_time(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v, weight="travel_time")
    except:
        return np.inf