import logging
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ============= SILVER =============

# OSM

def print_len_col_head(pois_gdf: gpd.GeoDataFrame) -> None:
    log.info(f"POIs récupérés: {len(pois_gdf)} avec {len(pois_gdf.columns.to_list())} colonnes.")
    if len(pois_gdf.columns.to_list()) < 20:
        log.info(pois_gdf.columns.tolist())
    log.info(pois_gdf.head())


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

# OSM

# DATATOURISME