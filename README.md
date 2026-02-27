# Travel Route Optimization - DataScientest Final Project for Data Engineering training

Le projet a pour vocation de proposer l'optimisation d'un itinéraire de voyage à partir de contraintes.  

## RoadMap
Les différentes étapes du projet sont à ce jour : 
1. Pipeline multi-sources (OSM & DATATourisme) des POIs et réseaux de routes ✅
2. Visualisation des POIs & réseaux de route ✅
3. Premier jet d'optimisation d'un itinéraire 👀
4. Ajout d'une brique de ML dans l'optimisation + passage en API 😴
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
    .venv\Scripts\activate.ps1
    ```

3. Installer les librairies requises
    ```sh
    pip3 install -r requirements.txt
    ```

4. Lancer la pipeline 
    ```sh
    python3 -m travel_route_optimization.data_pipeline.pipeline
    ```


## Todo du moment
- ~Vérifier que les geometry DATATourisme sont des points et pas des polygones~
- ~Retaper l'archi du projet pour qu'il soit clair~
- Faire l'enrichissement des données OSM & DATATourisme pour passer en Gold 
- Intégrer la logique de création de route (Algorithme de Dijkstra, TDTOPTW, peut-être démarrer par un TSP ou OP pour roter du sang direct)
- Ajouter une brique ML (piste de chercher statistiquement le prochain noeud qui a le plus de sens dans itinéraire en cours de construction)
- Découvrir Streamlit
- Réfléchir à une interface minimaliste mais fonctionnelle
- Mettre en place une pipeline CICD pour avoir une couverture de code 

## Conseils de Dan
- Réaliser chacune des briques de manière extrêmement simple, puis complexifier si possible pour augmenter le scope
- Utiliser Streamlit pour la partie UX/UI
- Pour la CICD faire un classique build-test-deploy
