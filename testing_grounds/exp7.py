# OBJECTIF : affecter interest_score, category, visit_duration pour chaque POI

# 1. category
# already done :)

# 2. interest_score
# A partir des category, définir un score d'intérêt (aka profit) à maximiser lors de l'entrainement
# - pouvoir donner des notes générales par categories
# - comment gérer les multiples categories de plusieurs types ? prendre la note de la meilleure
# (- pouvoir zoomer sur les sous-catégories)

# 3. visit_duration
# Fixer la durée des visites à sommer aux temps de trajet en secondes pour obtenir le poids de chaque arrête 
# 

# 4. opening_hours
# Normaliser les OH si elles existent, en définir des génériques le cas échéant selon le type de POI
# 08h00 => 12h00 puis 14h00 => 18h00

import pandas as pd
import geopandas as gpd
from pprint import pprint
import re

from travel_route_optimization.data_pipeline.utils.config import GOLD_POIS_GEOPARQUET

pois = gpd.read_parquet(GOLD_POIS_GEOPARQUET)

# for i in range(len(pois)):
#     if "|" in pois.loc[i]['categories']:
#         print(pois.loc[i])

dict_interest_score = {
    'leisure & entertainment'                           : 8,
    'cultural, historical & religious events or sites'  : 10,
    'parks, garden & nature'                            : 6,
    'sportive'                                          : 4,
    'restauration'                                      : 2,
    'accomodation'                                      : 0,
    'transport & mobility'                              : 0,
    'utilitaries'                                       : 0,
    ''                                                  : 0
}


def add_interest_score_score(poi: gpd.GeoSeries, dict_interest_score: dict) -> int:

    m_interest_score = 0
    # print(poi['categories'])
    for cat in poi['categories'].split("|"):
        # print(f"{cat} : {dict_interest_score[cat]}")
        if dict_interest_score[cat] > m_interest_score:
            m_interest_score = dict_interest_score[cat]
    # print(f"{poi['categories']} => {m_interest_score}")
    return m_interest_score

pois['interest_score'] = pois.apply(add_interest_score_score, args=(dict_interest_score,), axis=1)
pois['visit_duration'] = 30 * 60 # en secondes

print(pois.loc[0])

# rajouter OH !
oh_year_set = set()

oh_hour_set = set()

for i in range(len(pois)):
    if not pd.isna(pois.loc[i]['opening_hours']):
        if re.search("20[0-9]+", pois.loc[i]['opening_hours']) :
            oh_year_set.add(pois.loc[i]['opening_hours'])
        else:
            oh_hour_set.add(pois.loc[i]['opening_hours'])

pprint(oh_hour_set)