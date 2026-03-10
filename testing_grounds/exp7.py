# OBJECTIF : affecter interest_score, category, visit_duration pour chaque POI

# 1. category
# already done :)

# 2. interest_score
# A partir des category, définir un score d'intérêt (aka profit) à maximiser lors de l'entrainement
# - pouvoir donner des notes générales par categorie
# - comment gérer les multiples categories de plusieurs types ? prendre la note de la meilleure
# (- pouvoir donner du détail sur les sous-catégories)

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
    'parks, garden & nature'                            : 7,
    'sportive'                                          : 5,
    'restauration'                                      : 6,
    'accomodation'                                      : 4,
    'transport & mobility'                              : 0,
    'utilitaries'                                       : 0,
    ''                                                  : 0
}


def add_interest_score_score(poi: gpd.GeoSeries, dict_interest_score: dict) -> int:
    """Ajoute la valeur la plus élevée d'intérêt par catégorie pour chaque POI"""
    m_interest_score = 0
    for cat in poi['categories'].split("|"):
        if dict_interest_score[cat] > m_interest_score:
            m_interest_score = dict_interest_score[cat]
    return m_interest_score




pois['interest_score'] = pois.apply(add_interest_score_score, args=(dict_interest_score,), axis=1)
pois['visit_duration'] = 45 * 60 # en secondes

print(pois.loc[0])

# rajouter OH -> rajouté sous forme de mask pour le RL 
# 
# définir une zone avec assez de points pour établir un réseau

# left = 3.80 # Bédarieux
# right = 3.95
# top = 43.65
# bottom = 43.55

# bottom, left    = 43.55, 3.03 # zone entre Lamalou-les-Bains & Clermont l'Hérault
# top, right      = 43.70, 3.62

lon_centre, lat_centre = 43.6502211, 3.3741647 # lac du salagou

left    = lat_centre - .155
right   = lat_centre + .155
bottom  = lon_centre - .125
top     = lon_centre + .125


print(len(pois.cx[left:right, bottom:top]))

