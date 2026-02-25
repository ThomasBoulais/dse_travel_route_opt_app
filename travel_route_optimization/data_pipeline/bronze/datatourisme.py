"""
DataTourisme - Pipeline d'ingestion Bronze => Silver
====================================================
Structure du dump :
    data/bronze/datatourisme/dump/
    ├── index.json          # Catalogue : label, lastUpdateDatatourisme, file
    ├── context.jsonld      # Contexte JSON-LD (références struct/prop./... pas clair pourquoi mais pas utile)
    └── objects/
        └── XX/XXXX/        # Arborescence des fichiers avec noms de dossiersn en hexa
            └── <uuid>.json # Un POI par fichier

Usage :
    python3 ingest_datatourisme.py
"""

import json
import logging
from pathlib import Path
import requests
import zipfile

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from utils.config import DT_DUMP_DIR, DT_DUMP_PATH, DT_DUMP_URL, DT_INDEX_FILE, DT_SILVER_CSV, DT_SILVER_GEOPARQUET

# LOG

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# HELPERS

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


# BRONZE

def get_dump() -> None:
    """Récupère les données sous format GZIP depuis https://diffuseur.datatourisme.fr."""
    r = requests.get(DT_DUMP_URL)
    with open(DT_DUMP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def extract_dump() -> None:
    """Extrait le dump GZIP dans le dossier Bronze"""
    with zipfile.ZipFile(DT_DUMP_PATH, 'r') as zObject:
        zObject.extractall(path=DT_DUMP_DIR)


def load_index() -> list[dict]:
    """Charge index.json vers une liste de {label, file, lastUpdateDatatourisme}."""
    log.info(f"Lecture de l'index : {DT_INDEX_FILE}")
    with open(DT_INDEX_FILE, encoding="utf-8") as f:
        index = json.load(f)
    log.info(f"{len(index):,} entrées dans l'index.")
    return index


def ingest_bronze(index: list[dict]) -> list[dict]:
    """
    Parcourt tous les fichiers JSON référencés dans l'index (Bronze).
    Retourne la liste brute des entrées JSON-LD parsées.
    """
    raw_entries = []
    errors = 0

    for item in index:
        filepath = DT_DUMP_DIR / 'objects' / item["file"]
        if not filepath.exists():
            log.warning(f"Fichier introuvable : {filepath}")
            errors += 1
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                entry = json.load(f)
            # Certains fichiers encapsulent l'objet dans un @graph
            if "@graph" in entry:
                raw_entries.extend(entry["@graph"])
            else:
                raw_entries.append(entry)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Erreur lecture {filepath} : {e}")
            errors += 1

    log.info(f"Bronze : {len(raw_entries):,} objets chargés ({errors} erreurs).")
    return raw_entries


# SILVER

def transform_silver(raw_entries: list[dict]) -> gpd.GeoDataFrame:
    """
    Transforme les entrées brutes en GeoDataFrame nettoyé (Silver).
    - Exclut les POI sans coordonnées
    - Crée la géométrie Point (EPSG:4326)
    - Déduplique sur l'identifiant DataTourisme
    """
    records = []
    skipped_no_geo = 0

    for entry in raw_entries:
        source = entry.get("@id", "unknown")
        row = parse_poi(entry, source)
        if row is None:
            skipped_no_geo += 1
        else:
            records.append(row)

    log.info(
        f"Silver : {len(records):,} POI géolocalisés, "
        f"{skipped_no_geo:,} écartés (sans coordonnées)."
    )

    df = pd.DataFrame(records)

    # Déduplication : garde la version la plus récente de chaque POI
    df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
    df = (df
          .sort_values("last_update", ascending=False, na_position="last")
          .drop_duplicates(subset="id", keep="first")
          .reset_index(drop=True))

    log.info(f"Silver : {len(df):,} POI après déduplication.")

    # Création de la géométrie
    geometry = [Point(row.longitude, row.latitude) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


# EXPORT

def export_silver(gdf: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Silver) et CSV optionnel."""
    gdf.to_parquet(DT_SILVER_GEOPARQUET, index=False)
    log.info(f"GeoParquet sauvegardé : {DT_SILVER_GEOPARQUET}  ({len(gdf):,} lignes)")

    # CSV sans géométrie pour exploration rapide
    gdf.drop(columns="geometry").to_csv(DT_SILVER_CSV, index=False, encoding="utf-8-sig")
    log.info(f"CSV sauvegardé       : {DT_SILVER_CSV}")


# MAIN

def main():
    log.info("=== Démarrage pipeline DataTourisme Bronze => Silver ===")

    # Bronze : ajout & lecture des fichiers bruts
    get_dump()
    extract_dump()
    index       = load_index()
    raw_entries = ingest_bronze(index)

    # Silver : transformation & nettoyage
    gdf = transform_silver(raw_entries)

    # Export
    export_silver(gdf)

    # Aperçu
    log.info("\n── Aperçu (5 premières lignes) ──")
    print(gdf[["name_fr", "types", "city", "latitude", "longitude"]].head())

    log.info("\n── Répartition par type (top 10) ──")
    type_counts = (
        gdf["types"]
        .str.split("|")
        .explode()
        .value_counts()
        .head(10)
    )
    print(type_counts.to_string())

    log.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
