# Travel Route Optimization - DataScientest Final Project for Data Engineering training

Le projet a pour vocation de proposer l'optimisation d'un itinéraire de voyage à partir de contraintes.  

## RoadMap
Les différentes étapes du projet sont à ce jour : 
1. Pipeline multi-sources (OSM & DATATourisme) des POIs et réseaux de routes ✅
2. Visualisation des POIs & réseaux de route ✅
3. Premier jet d'optimisation d'un itinéraire en RL ✅
4. Ajout de versioning (MLFlow) + Passage en API (fastapi) ✅
5. Architecture complète de la donnée avec contraintes utilisateur 👀
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
    python3 -m src.data_pipeline.pipeline_runner
    ```

5. Pour voir les modèles déjà entraînés et en entraîner d'autres, installer MLFlow et lancer un serveur
    ```
    pip install mlflow

    mlflow server  --host 0.0.0.0  --port 5000  --backend-store-uri sqlite:///mlruns/mlflow.db  --default-artifact-root file:mlruns

    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

    python -m src.model_training.train_dqn

    python -m src.model_training.register_model <RUN_ID>
    ```  

    ⚠️ Vérifier qu'un modèle est en production afin d'y avoir accès via fastapi (UX MLFlow)

6. Pour accéder au modèle via fastapi, lancer un serveur UVICORN
    ```
    uvicorn src.api.fastapi_app:app --reload
    ```

    Puis accéder à fastapi via `http://localhost:8000/docs` ou shell et essayer le modèle pour obtenir un itinéraire:
    ```
    curl -X 'POST' \
    'http://localhost:8000/itinerary' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "start_poi": 0,
        "start_day": 0,
        "num_days": 3,
        "model_name": "tdtoptw_dqn",
        "config_path": "training.yaml"
        }'
    ```

7. 

## Todo du moment

- Découvrir Streamlit et ses fonctionnalités
- Penser une UX avec prise en charge des inputs User
- Prendre en main Folium / voir si compat possible avec Streamlit & faire affichage clair d'un itinéraire
- (modifier RL pour améliorer le bail et/ou rendre les profits des différentes catégories modifiables)

## RoadMap
- ~Vérifier que les geometry DATATourisme sont des points et pas des polygones~
- ~Retaper l'archi du projet pour qu'il soit clair~
- ~Faire l'enrichissement des données OSM & DATATourisme pour passer en Gold~
- ~Intégrer la logique de création de route (Algorithme de Dijkstra, TDTOPTW, peut-être démarrer par un TSP ou OP pour roter du sang direct) + Ajouter une brique ML (piste de chercher statistiquement le prochain noeud qui a le plus de sens dans itinéraire en cours de construction)~ => ~Mettre en place un RL pour déterminer la meilleure route~
- ~Exposer le modèle via API~
- ~Découvrir Streamlit~
- ~Réfléchir à une interface minimaliste mais fonctionnelle~
- Containeriser l'application
- Mettre en place une pipeline CICD pour avoir une couverture de code 

## Conseils de Dan
- Réaliser chacune des briques de manière extrêmement simple, puis complexifier si possible pour augmenter le scope
- Utiliser Streamlit pour la partie UX/UI
- Pour la CICD faire un classique build-test-deploy
