# Travel Route Optimization - DataScientest Final Project for Data Engineering training

Le projet a pour vocation de proposer l'optimisation d'un itinéraire de voyage à partir de contraintes.  

## RoadMap
Les différentes étapes du projet sont à ce jour : 
1. Pipeline multi-sources (OSM & DATATourisme) des POIs et réseaux de routes ✅
2. Visualisation des POIs & réseaux de route ✅
3. Premier jet d'optimisation d'un itinéraire en RL ✅
4. Ajout de versioning (MLFlow) + Passage en API (fastapi) 👀
5. Architecture complète de la donnée avec contraintes utilisateur 😴
6. UX/UI 😴
7. Branchement Front-Back 😴
8. CICD 😴

## Comment lancer

1. Créer un virtualenv `.venv` :
    ```sh
    python3 -m venv .venv
    ```

2. Activer l'environnement virtuel si besoin :
    ```sh
    # shell
    source .venv\bin\activate

    # powershell
    Set-ExecutionPolicy Unrestricted -Scope Process
    .venv\Scripts\activate.ps1
    ```

3. Installer les librairies requises
    ```sh
    pip3 install -r requirements.txt
    ```

Puis parmi les actions possibles : 

4. Lancer la pipeline pour récupérer les données
    ```sh
    python3 -m travel_route_optimization.data_pipeline.pipeline
    ```

5. Lancer l'entrainement de RL
    ```sh
    python3 -m travel_route_optimization.model_training.train_dqn

    python -m travel_route_optimization.model_training.eval_route <RUN_ID>
    ```

6. Pour voir les modèles déjà entraînés, installer MLFlow et lancer un serveur
    ```
    pip install mlflow

    mlflow server --host 0.0.0.0 --port 5000 --backend-store- file:C:\Users\thoma\Documents\python_projects\dse_travel_route_opt_app\mlruns --default-artifact-root file:C:\Users\thoma\Documents\python_projects\dse_travel_route_opt_app\mlruns --serve-artifacts
    ```

7. Pour accéder au modèle, lancer un serveur UVICORN
    ```
    uvicorn travel_route_optimization.api.fastapi_app:app --reload
    ```

    Then access fastapi with `http://localhost:8000/docs` and try out itinerary or in shell:
    ```
    curl -X 'POST' \
    'http://localhost:8000/itinerary' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "start_poi": 0,
        "start_day": 0,
        "num_days": 3,
        "model_name": "4287957a48224b1c97cbf3e610c6aaa0",
        "config_path": "config.yaml"
        }'
```


## Todo du moment

1. Enrichir les données OSM + DATATourisme dans gold 
    11. Construire une structure commune
    12. Fusionner les points commun

11. Construire une structure commune
- name                  : nom du POI (noms vides dans OSM à investiguer)
- geometry              : géoloc => clef approx. d'égalité entre les src
- category              : tag de référence pour les utilisateurs (culture, sport, etc.)
- type                  : à rationnaliser => clef approx. d'égalité entre les src
- opening hours         : gérer les vides
- address               : pour l'export utilisateur
- email, phone, website : 

## RoadMap
- ~Vérifier que les geometry DATATourisme sont des points et pas des polygones~
- ~Retaper l'archi du projet pour qu'il soit clair~
- ~Faire l'enrichissement des données OSM & DATATourisme pour passer en Gold~
- Intégrer la logique de création de route (Algorithme de Dijkstra, TDTOPTW, peut-être démarrer par un TSP ou OP pour roter du sang direct)
- Ajouter une brique ML (piste de chercher statistiquement le prochain noeud qui a le plus de sens dans itinéraire en cours de construction)
- Découvrir Streamlit
- Réfléchir à une interface minimaliste mais fonctionnelle
- Mettre en place une pipeline CICD pour avoir une couverture de code 

## Conseils de Dan
- Réaliser chacune des briques de manière extrêmement simple, puis complexifier si possible pour augmenter le scope
- Utiliser Streamlit pour la partie UX/UI
- Pour la CICD faire un classique build-test-deploy
