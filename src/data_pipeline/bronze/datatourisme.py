"""
DataTourisme - Pipeline d'ingestion Source => Bronze

Structure du dump :
    data/bronze/datatourisme/dump/
    ├── index.json          # Catalogue : label, lastUpdateDatatourisme, file
    ├── context.jsonld      # Contexte JSON-LD (références struct/prop./... pas clair pourquoi mais pas utile)
    └── objects/
        └── XX/XXXX/        # Arborescence des fichiers avec noms de dossiersn en hexa
            └── <uuid>.json # Un POI par fichier
"""

import json
import logging
from pathlib import Path
import requests
import zipfile

# from src.utils.config import DT_DUMP_DIR, DT_DUMP_PATH, DT_DUMP_URL, DT_INDEX_FILE
from src.common.config_loader import load_config

cfg = load_config()


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# BRONZE

def get_dump() -> None:
    """Récupère les données sous format GZIP depuis https://diffuseur.datatourisme.fr."""
    r = requests.get(cfg.bronze.dt_dump_url, stream=True)
    with open(cfg.bronze.dt_dump_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def extract_dump() -> None:
    """Extrait le dump GZIP dans le dossier Bronze"""
    with zipfile.ZipFile(cfg.bronze.dt_dump_path, 'r') as zObject:
        zObject.extractall(path=cfg.bronze.dt_dump_dir)


def load_index() -> list[dict]:
    """Charge index.json vers une liste de {label, file, lastUpdateDatatourisme}."""
    log.info(f"Source => Bronze (DATATOURISME) : Lecture de l'index {cfg.bronze.dt_index_file}")
    with open(cfg.bronze.dt_index_file, encoding="utf-8") as f:
        index = json.load(f)
    log.info(f"Source => Bronze (DATATOURISME) : {len(index):,} entrées dans l'index.")
    return index


def ingest_bronze(index: list[dict]) -> list[dict]:
    """
    Parcourt tous les fichiers JSON référencés dans l'index (Bronze).
    Retourne la liste brute des entrées JSON-LD parsées.
    """
    raw_entries = []
    errors = 0

    for item in index:
        filepath = Path(cfg.bronze.dt_dump_dir) / 'objects' / item["file"]
        if not filepath.exists():
            log.warning(f"Source => Bronze (DATATOURISME) : Fichier introuvable {filepath}")
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
            log.warning(f"Source => Bronze (DATATOURISME) : Erreur lecture {filepath} : {e}")
            errors += 1

    log.info(f"Source => Bronze (DATATOURISME) : {len(raw_entries):,} objets chargés ({errors} erreurs).")
    return raw_entries
