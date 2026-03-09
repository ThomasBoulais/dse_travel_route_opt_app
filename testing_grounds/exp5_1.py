import fastparquet
import numpy as np
import re
import unicodedata
from pprint import pprint

from travel_route_optimization.data_pipeline.utils.config import OSM_SILVER_GEOPARQUET
from travel_route_optimization.data_pipeline.utils.pipeline_helpers import to_geopandas


# -----------------------------
# Chargement des données
# -----------------------------
osm_df = fastparquet.ParquetFile(OSM_SILVER_GEOPARQUET).to_pandas()
osm_gdf = to_geopandas(osm_df)

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
INT_TYPE = np.uint8
INT_TYPE = np.int16


# -----------------------------
# Nettoyage
# -----------------------------
def clean_hours(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[‐‑‒–—−]", "-", text)                     # dashes
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)  # spaces
    text = re.sub(r"\s+", " ", text)
    text = text.replace("H", ":").replace("h", ":")
    return text.strip()


# -----------------------------
# Cas spéciaux
# -----------------------------
def detect_special_cases(text):
    t = text.lower()
    if "closed" in t or "fermé" in t:
        return "closed"
    if "24/7" in t or "24h/24" in t:
        return "24_7"
    return None


# -----------------------------
# Parsing OSM simple
# -----------------------------
def parse_osm_hours(text):
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


# -----------------------------
# Conversion en minutes
# -----------------------------
def to_minutes(hhmm):
    try:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m
    except:
        return None


# -----------------------------
# Construction du masque 7×1440
# -----------------------------
def build_open_mask(parsed):
    mask = np.zeros((7, 1440), dtype=INT_TYPE)
    for entry in parsed:
        d = DAYS.index(entry["day"])
        start = to_minutes(entry["start"])
        end   = to_minutes(entry["end"])
        # print(f"\n----------------------------------------------\n{entry}\n{start} - {end}")

        if start is None or end is None:
            continue
        if start < end:
            mask[d, start:end] = 1
        else : # dépassement à J+1
            mask[d, start:1440] = 1
            mask[(d+1) % 7, 0:end] = 1
    return mask


# -----------------------------
# Pipeline complet
# -----------------------------

HORAIRE_GENERIQUE = build_open_mask(
    [{'day': 'Mo', 'start': '08:00', 'end': '12:00'}, {'day': 'Mo', 'start': '14:00', 'end': '18:00'},
     {'day': 'Tu', 'start': '08:00', 'end': '12:00'}, {'day': 'Tu', 'start': '14:00', 'end': '18:00'},
     {'day': 'We', 'start': '08:00', 'end': '12:00'}, {'day': 'We', 'start': '14:00', 'end': '18:00'},
     {'day': 'Th', 'start': '08:00', 'end': '12:00'}, {'day': 'Th', 'start': '14:00', 'end': '18:00'},
     {'day': 'Fr', 'start': '08:00', 'end': '12:00'}, {'day': 'Fr', 'start': '14:00', 'end': '18:00'},
     {'day': 'Sa', 'start': '08:00', 'end': '12:00'}, {'day': 'Sa', 'start': '14:00', 'end': '18:00'},]
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


# -----------------------------
# Application
# -----------------------------



osm_df["opening_mask"] = osm_df["opening_hours"].apply(opening_hours_to_mask)
osm_df["nb_minutes"] = osm_df["opening_mask"].apply(lambda x: sum(x[0]))
print(osm_df[["opening_hours", "nb_minutes"]].head())

# for day in osm_df.iloc[0]['opening_mask']:
#     print(f"{len(day)} - {sum(day)}")