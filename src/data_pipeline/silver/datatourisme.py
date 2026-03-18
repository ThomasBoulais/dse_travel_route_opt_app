"""
Bronze => Silver (DATATOURISME) : Transformation Bronze => Silver
"""

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from src.utils.pipeline_helpers import parse_poi
# from src.utils.config import DEFAULT_CRS, DT_SILVER_CSV, DT_SILVER_GEOPARQUET
from src.common.config_loader import load_config

cfg = load_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# SILVER

def transform_silver(raw_entries: list[dict], left: float, right: float, bottom: float, top: float) -> gpd.GeoDataFrame:
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
        f"Bronze => Silver (DATATOURISME) : {len(records):,} POI géolocalisés, "
        f"{skipped_no_geo:,} écartés (sans coordonnées)."
    )

    df = pd.DataFrame(records)

    # Déduplication : garde la version la plus récente de chaque POI
    df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
    df = (df
          .sort_values("last_update", ascending=False, na_position="last")
          .drop_duplicates(subset="id", keep="first")
          .reset_index(drop=True))

    log.info(f"Bronze => Silver (DATATOURISME) : {len(df):,} POI après déduplication.")

    # Création de la géométrie
    geometry = [Point(row.longitude, row.latitude) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=cfg.crs.default)

    # Troncature sur la boundary box
    gdf = gdf.cx[left:right, bottom:top].reset_index(drop=True)

    return gdf


def export_silver(gdf: gpd.GeoDataFrame) -> None:
    """Sauvegarde en GeoParquet (Silver) et CSV optionnel."""
    gdf.to_parquet(cfg.silver.dt_geoparquet, index=False)
    log.info(f"Bronze => Silver (DATATOURISME) : GeoParquet sauvegardé {cfg.silver.dt_geoparquet}  ({len(gdf):,} lignes)")

    # CSV sans géométrie pour exploration rapide
    gdf.drop(columns="geometry").to_csv(cfg.silver.dt_csv, index=False, encoding="utf-8-sig")
    log.info(f"Bronze => Silver (DATATOURISME) : CSV sauvegardé {cfg.silver.dt_csv}")
