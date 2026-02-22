import json
import zipfile
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

def parse_datatourisme_entry(data: dict) -> dict | None:
    """
    Extract relevant fields from one DataTourisme JSON-LD object.
    Returns a flat dict or None if the entry lacks coordinates.
    """
    try:
        # The @graph key contains the main objects in JSON-LD
        graph = data.get("@graph", [data])
        main = next(
            (obj for obj in graph if "schema:geo" in obj),
            None
        )
        if not main:
            return None

        geo = main["schema:geo"]
        lat = float(geo.get("schema:latitude", 0))
        lon = float(geo.get("schema:longitude", 0))

        label = main.get("rdfs:label", {})
        name = label.get("fr", label.get("en", "Unknown"))

        return {
            "id": main.get("@id", ""),
            "name": name,
            "type": main.get("@type", []),
            "latitude": lat,
            "longitude": lon,
        }
    except (KeyError, TypeError, ValueError):
        return None


def ingest_datatourisme_dump(zip_path: str, output_path: str):
    records = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        print(f"Found {len(json_files)} JSON files in dump")

        for i, fname in enumerate(json_files):
            with zf.open(fname) as f:
                try:
                    data = json.load(f)
                    record = parse_datatourisme_entry(data)
                    if record:
                        records.append(record)
                except json.JSONDecodeError:
                    continue  # skip malformed files

            if i % 10000 == 0:
                print(f"Processed {i}/{len(json_files)}...")

    print(f"Successfully parsed {len(records)} POIs")

    # Build a GeoDataFrame
    df = pd.DataFrame(records)
    geometry = [Point(row.longitude, row.latitude) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf.to_parquet(output_path)
    print(f"Saved to {output_path}")


ingest_datatourisme_dump(
    "data/bronze/dt_dump.zip",
    "data/silver/dt_pois.geoparquet"
)