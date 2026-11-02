# GeoCongo AI — Geological, Geospatial & Mining AI Python SDK (v0.4.3)

> **The Python SDK for geological RAG, geospatial remote sensing, 3D mining resource modeling, natural language QGIS workflows, and real-time streaming.**

`geocongoai` est le SDK officiel Python pour **GeoCongo AI**. Il est structuré autour de **4 moteurs principaux** en Swahili (Pekua, Gundua, Chimbua et Mtumishi) appuyés par des outils scientifiques utilitaires (visualisation 3D, vision, inférence IA modèles de fondation géospatiale, DevTools d'ingénierie logicielle, streaming en temps réel).

---

## 🏛️ L'Architecture des 4 Moteurs GeoCongo AI

```text
                                         GEOCONGO AI SDK (v0.4.3)
                                                    │
        ┌───────────────────────────┬───────────────┴───────────────┬───────────────────────────┐
        ▼                           ▼                               ▼                           ▼
  01 PEKUA ENGINE             02 GUNDUA ENGINE                03 CHIMBUA ENGINE           04 MTUMISHI ENGINE
  (Moteur de Recherche)          (Moteur de Découverte)          (Moteur d'Extraction)       (Serviteur Langage Naturel)
        │                           │                               │                           │
  • Agent RAG Géoscientifique • Analyse basée sur des Règles  • Drillhole DB & QA/QC      • Orchestration Langage Naturel
  • Recherche Documentaire    • AI Foundation Models          • Compositing 3D            • QGIS Desktop Bridge (PyQGIS/RPC)
  • Recherche Vectorielle     • Télédétection Hyperspectrale  • Domaines & Block Model    • Streaming Temps Réel (stream)
  • Citations PDF & Cartes                                   • Krigeage & Gisement       • Multi-Runtime (Native/Antigravity)
                                                                                         • DevTools (Filesystem & Exec)

  ── UTILITAIRES / TOOLBOX ────────────────────────────────────────────────────────────────────────────────────────────
  • datasets (DrillholeDataset, SampleDataset)   • visualization (Plotly 3D, Export HTML)
  • results (GeoResult universel)                • ia (Clay v1.5, Prithvi v2, AlphaEarth)
  • vision (Pansharpen, Contrast, Rembg, qr)     • tools.filesystem & tools.execution (DevTools)
```

---

## 🚀 Présentation des 3 Moteurs

### 1. 🔍 Pekua Engine (`geocongoai.pekua_engine`)
>
> *"Pekua"* signifie *fouiller / rechercher dans les livres* en Swahili.

* **Rôle** : Moteur de recherche géoscientifique, d'extraction documentaire.
* **Fonctionnalités** :
  * Interfaçage avec les Edge Functions Supabase (`/rag-agent`, `/search-documents`, `/search-geological`).
  * Recherche vectorielle 1536D dans pgvector (documents, cartes, roches, jeux de données).
  * Agent IA géoscientifique multilingue.

### 2. 🛰️ Gundua Engine (`geocongoai.gundua_engine`)
>
> *"Gundua"* signifie *découvrir / explorer* en Swahili.

* **Rôle** : Moteur de découverte minière par télédétection, imagerie satellite et IA géospatiale.
* **Fonctionnalités** :
  * Module d'analyse basée sur des règles et indices spétraux (`greenfield`, `illegal_mining`, `lineaments`, `landcover`, `landslide`).
  * Module d'analyse basées sur les modèles de fondation géospatiaux (`mining_sites_monitoring`, `structral_lineaments`, `geological_units`, `lithology`, `hydrothermal_alteration`, `mineral_detection`).
  * Module d'analyse des données hyperspectrales (`metal_stressed_vegetation`, `structral_lineaments`, `lithology`, `hydrothermal_alteration`, `mineral_detection`).

### 3. ⛏️ Chimbua Engine (`geocongoai.chimbua_engine`)
>
> *"Chimbua"* signifie *extraire / miner / exploiter* en Swahili.

* **Rôle** : Moteur de gestion des forages, modélisation géologique 3D, géostatistique, estimation de ressources et évaluation technico-économique.
* **Fonctionnalités** :
  * **Drillholes & QA/QC** : Ingestion, validation topologique des sondages et trajectoires 3D.
  * **Compositing** : Régularisation des longueurs d'échantillonnage.
  * **Domaines Géologiques & Wireframes 3D** : Délimitation déterministe et assistée par IA des enveloppes minéralisées.
  * **Block Model 3D** : Grille régularisée, sous-blocs et contraintes de domaine.
  * **Géostatistique & Krigeage** : Variogrammes empiriques/modélisés, Krigeage Ordinaire (OK), IDW, NN.
  * **Resource Estimation & Classification** : Tonnage, teneurs, métal contenu et classification (Mesuré, Indiqué, Inféré).
  * **Optimisation de Fosse & Économie** : Cônes emboîtés, ratio de stérile, CAPEX, OPEX, Cash-Flow, NPV, IRR.

### 4. 🤖 Mtumishi Engine (`geocongoai.mtumishi` / `geocongoai.agent`)
>
> *"Mtumishi"* signifie *serviteur / assistant* en Swahili.

* **Rôle** : Serviteur intelligent orchestrant en langage naturel les moteurs géoscientifiques (Pekua, Gundua, Chimbua) et l'interaction avec **QGIS Desktop**.
* **Fonctionnalités** :
  * **Interaction en Langage Naturel** : Exécution synchrone (`run()`) et asynchrone (`run_async()`).
  * **Pont QGIS Desktop (`QGISBridge`)** : Synchronisation bidirectionnelle PyQGIS, RPC/Socket avec le plugin QGIS Desktop et génération de scripts PyQGIS.
  * **Catalogue GeoTools & Runtimes** : Multi-runtimes (Natif ReAct et Google Antigravity SDK) avec support MCP (Model Context Protocol).
  * **Capacités Natives d'Ingénierie Logicielle (DevTools v0.4.1)** : Manipulation de fichiers locaux (`write_file`, `read_file`, `delete_file`, `list_directory`) et exécution de code Python / commandes shell (`execute_python_code`, `run_shell_command`).

> 🧩 **Qu'est-ce qu'un GeoTool ?**
> Un **GeoTool** est une unité d'action autonome (`name`, `description`, `parameters`, `handler`) que **Mtumishi** sélectionne et exécute automatiquement à partir d'une consigne en langage naturel (*Function Calling / Tool Use*).
> Mtumishi embarque les GeoTools de Pekua, Gundua, Chimbua, QGIS Desktop, de la Toolbox (`ia`, `vision`, `datasets`, `visualization`) et des **DevTools** (`filesystem`, `execution`), et permet d'ajouter des outils personnalisés via `mtumishi.register_tool(...)`.

---

## 🛠️ Installation

```bash
# SDK de base (requiert uniquement requests)
pip install geocongoai

# Extras optionnels — installez uniquement ce dont vous avez besoin :
pip install geocongoai[agent]      # Mtumishi + runtime officiel google-antigravity
pip install geocongoai[stac]       # Accès STAC Microsoft Planetary Computer & Indices Gundua
pip install geocongoai[vision]     # Vision : pansharpening, contrast stretching, QR code, détourage
pip install geocongoai[ia]         # IA Foundation : Clay v1.5, Prithvi v2, AlphaEarth (nécessite PyTorch)
pip install geocongoai[spatial3d]  # Visualisation 3D : Plotly, Scipy, Scikit-Learn

# Installation complète
pip install geocongoai[agent,stac,vision,ia,spatial3d]
```

> **Note** : `geocongoai[ia]` installe PyTorch, torchgeo et earthengine-api (~2 Go). Sans cet extra, le module `geocongoai.ia` est importable mais les classes lèvent une `ImportError` explicite uniquement à l'appel.
>
> **Mtumishi + Antigravity** : pour le runtime agentique officiel, définissez `GEMINI_API_KEY` (ou `GOOGLE_API_KEY`). Sans authentification Gemini/Vertex, `Mtumishi` bascule explicitement sur `NativeRuntime`.


---

## 💻 Exemples d'Utilisation

### 🔑 Initialisation du Client (Authentification API Key)

```python
import os
from geocongoai import GeoCongoClient

# Recommandé : via variable d'environnement
os.environ["GEOCONGOAI_API_KEY"] = "gcg_live_votre_cle_api"
client = GeoCongoClient()

# Ou directement dans le constructeur
# client = GeoCongoClient(api_key="gcg_live_votre_cle_api")
```

---

### 🔍 Exemple 1 : Pekua Engine (Agent RAG & Recherche Géoscientifique)

```python
from geocongoai.pekua_engine import PekuaEngineClient

# Initialisation du moteur Pekua
pekua = PekuaEngineClient(api_key="gcg_live_votre_cle_api")

# 1. Poser une question à l'Agent RAG Pekua
response = pekua.ask_rag("Quels sont les gisements connus de cobalt au Lualaba ?")
print("Réponse Pekua :", response.answer)
for src in response.sources:
    print(f"- Source : {src.title} (similarité : {src.similarity})")

# 2. Recherche documentaire ciblée avec Pekua
docs = pekua.search_documents(
    query="cuivre et cobalt",
    domain="Mines",
    province="Lualaba"
)
print(f"Trouvé {docs.total_found} documents.")

# 3. Recherche géologique vectorielle multimodale avec Pekua
geo_results = pekua.search_geological(
    query="malachite et roche sédimentaire",
    type="rocks",
    province="Haut-Katanga"
)
```

---

### 🛰️ Exemple 2 : Gundua Engine (Télédétection & Prospection IA)

```python
from geocongoai.gundua_engine import GunduaEngineClient
from geocongoai import stac_search_scenes, stac_calc_indices

client = GunduaEngineClient()

# 1. Analyse du potentiel minier (Greenfield)
result = client.analyze(
    "greenfield",
    bbox=[28.5, -11.5, 28.6, -11.4],   # [min_lon, min_lat, max_lon, max_lat]
    datetime="2023-06-01/2023-06-30"
)
print("Potentiel minier :", result.get("potential"))

# 2. Extraction de linéaments (failles / structures)
lineaments = client.analyze("lineaments", bbox=[28.5, -11.5, 28.6, -11.4])

# 3. Accès satellitaire STAC (Microsoft Planetary Computer) & Indices Spectraux
#    Recherche et calcul lazy-loading d'indices miniers (oxydes de fer, argilisation, NDVI)
indices_result = stac_calc_indices(
    bbox=[25.4, -10.75, 25.55, -10.65],
    datetime="2026-05-01/2026-09-30",
    indices=["NDVI", "IRON_OXIDE", "CLAY_OH", "FERROUS"],
    collection="sentinel-2-l2a"
)
print("Indices calculés :", list(indices_result.get("indices", {}).keys()))

# 4. Altération hydrothermale fine par capteur ASTER L1T (SWIR/TIR)
#    Alunite (argilique avancée), Kaolinite, Séricite/Muscovite (phyllique),
#    Carbonates, Chlorite/Épidote (propylitique), Quartz/Silice (TIR Ninomiya)
aster_result = stac_calc_indices(
    bbox=[25.4, -10.75, 25.55, -10.65],
    datetime="2005-01-01/2007-12-31",
    indices=["gundua_aster"],  # ou "aster_alteration"
    collection="aster-l1t"
)
print("Indices ASTER calculés :", list(aster_result.get("indices", {}).keys()))
```

---

### ⛏️ Exemple 3 : Chimbua Engine (Gestion Forages, Clustering 3D & Visualisation)

```python
from geocongoai.chimbua_engine import (
    analyse_et_visualiser_forages_3d,
    compute_drillhole_intervals,
    cluster_assay_points,
    generate_cluster_hulls
)

collars = [
    {"hole_id": "DH01", "x": 500000, "y": 9200000, "z": 1200, "dip": -90, "azimuth": 0},
    {"hole_id": "DH02", "x": 500050, "y": 9200050, "z": 1205, "dip": -90, "azimuth": 0},
]
assays = [
    {"hole_id": "DH01", "from_m": 0, "to_m": 5, "cu_pct": 1.8},
    {"hole_id": "DH02", "from_m": 0, "to_m": 5, "cu_pct": 2.1},
]

# Pipeline complet 3D & Clustering DBSCAN
result = analyse_et_visualiser_forages_3d(
    collars=collars,
    assays=assays,
    grade_field="cu_pct",
    grade_threshold=1.0,
    output_html_path="rapport_3d.html"
)
print("Nombre d'intervalles analysés :", result["total_intervals"])
```

--- 

### 🤖 Exemple 4 : Mtumishi (Serviteur Langage Naturel, QGIS Desktop & Toolbox)

```python
import os
from geocongoai import Mtumishi, GeoSpatialContext, GeoTool

# Runtime agentique officiel google-antigravity
os.environ["GEMINI_API_KEY"] = "votre_cle_gemini"

# Initialisation du serviteur Mtumishi (Antigravity par défaut, fallback natif explicite)
mtumishi = Mtumishi(runtime_kwargs={"workspaces": ["."]})

# 1. Définition du contexte géospatial (optionnel)
ctx = GeoSpatialContext(province="Lualaba", target_commodity="Cu")

# 2. Interaction en langage naturel (Pekua, Gundua, Chimbua & QGIS Desktop)
result = mtumishi.run(
    "Analyse la zone de Kolwezi pour le cuivre, "
    "affiche la carte dans QGIS Desktop.",
    context_override=ctx
)
print(result.summary)
print("Actions exécutées :", result.actions_executed)

# 3. Utilisation directe des utilitaires de la Toolbox (Vision & QR Code)
res_qr = mtumishi.run("Génère un QR Code pour les métadonnées de l'échantillon SP-2026")
print(res_qr.summary)

# 4. Enregistrement d'un GeoTool personnalisé
def analyser_qualite_roche(echantillon_id: str):
    return f"Échantillon {echantillon_id} validé avec teneur 2.4% Cu."

mtumishi.register_tool(GeoTool(
    name="analyser_qualite_roche",
    description="Vérifie la teneur et la qualité d'un échantillon géologique.",
    parameters={
        "type": "object",
        "properties": {
            "echantillon_id": {"type": "string", "description": "Identifiant de l'échantillon"}
        },
        "required": ["echantillon_id"]
    },
    handler=analyser_qualite_roche
))
```

---

### 🧰 Exemple 5 : Toolbox Utilitaires (`geocongoai.vision`, `geocongoai.datasets`, `geocongoai.visualization`)

```python
# 1. Vision & Traitement d'Images Satellites (Pansharpening, Contrast, QR, Détourage)
from geocongoai.vision import (
    contrast_stretching,
    pansharpen_brovey,
    generate_qr,
    remove_background,
)

# Étirement de contraste linéaire avec Percentile Clipping (2-98%) sur GeoTIFF satellite
# (Préserve le CRS, la transformation géométrique et les métadonnées NoData)
contrast_stretching(
    in_path="sentinel2_lualaba_raw.tif",
    out_path="sentinel2_lualaba_enhanced.tif",
    percentile=(2, 98)
)

# Étirement de contraste sur image de terrain classique (JPG/PNG)
contrast_stretching(
    in_path="affleurement_sombre.jpg",
    out_path="affleurement_eclaire.jpg",
    percentile=(1, 99)
)

# Pansharpening Brovey géospatialisé (Fusion RGB 30m + Panchromatique 15m -> 15m HR)
pansharpen_brovey(
    ms_path="image_rgb_30m.tif",
    pan_path="bande_panchromatique_15m.tif",
    out_path="output_brovey_15m.tif"
)

# Génération de QR Code pour la traçabilité des échantillons miniers
generate_qr("GEOCONGO-SAMPLE-LUALABA-2026-001", output_path="qr_sample.png")

# Détourage et suppression du fond sur une photo de lame mince ou roche
remove_background("roche_brute.jpg", output_path="roche_detouree.png")

# 2. Gestion des Jeux de Données Géologiques Unifiés
from geocongoai.datasets import DrillholeDataset, SampleDataset

drillholes = DrillholeDataset("forages_lualaba.csv")
samples = SampleDataset("echantillons_geochimie.csv")

# 3. Rendus et Visualisations 3D Interactives
from geocongoai.visualization import HTMLRenderer, PlotlyRenderer

renderer = HTMLRenderer()
renderer.render_to_file("modele_blocs_3d.html")
```

---

### 🤖 Exemple 6 : IA Foundation Models (`geocongoai.ia`)

> Nécessite `pip install geocongoai[ia]` (PyTorch, torchgeo, earthengine-api)

```python
# ── Clay v1.5 — Embeddings Géospatiaux Précalculés (sans GPU) ──────────────
from geocongoai.ia import ClayClient

clay = ClayClient(use_torchgeo=False)

# Charger des embeddings précalculés depuis Source Cooperative (GeoParquet)
dataset = clay.load_precomputed_embeddings(
    "https://source.coop/clay/embeddings/v1.5/lualaba_2024.parquet"
)

# Filtrer spatialement sur la zone de Kolwezi
kolwezi_bbox = (25.40, -10.75, 25.55, -10.60)
local_emb = dataset.filter_by_bbox(*kolwezi_bbox)
print(f"Embeddings chargés : {len(local_emb)} tuiles")

# Réduction PCA en 3 composantes RGB pour visualisation cartographique
rgb_map = clay.pca_to_rgb(local_emb, out_path="kolwezi_pca.tif")

# ── Prithvi v2 — Deep Features Temporelles (nécessite GPU ou CPU lent) ──────
from geocongoai.ia import PrithviClient

prithvi = PrithviClient(model_name="prithvi_eo_v2_300", device="cpu")
features = prithvi.extract_deep_features("sentinel2_lualaba.tif")
print("Feature shape:", features["features"].shape)

# ── AlphaEarth — Embeddings Satellite Google Earth Engine (64D) ─────────────
from geocongoai.ia import AlphaEarthClient

alpha = AlphaEarthClient()
alpha.authenticate()  # utilise gcloud credentials ou service account

embedding = alpha.get_embedding(
    longitude=25.47, latitude=-10.71, year=2023
)
print("AlphaEarth embedding (64D):", embedding[:5], "…")
```

---

### 💻 Exemple 7 : DevTools d'Ingénierie Logicielle (`geocongoai.mtumishi.tools`)

> Disponible nativement à partir de **v0.4.1**.

Mtumishi intègre désormais des outils officiels pour manipuler le système de fichiers local et exécuter du code / des commandes système, utilisables soit directement en Python, soit par l'agent via le prompt en langage naturel :

#### 1. Utilisation directe des modules

```python
from geocongoai.mtumishi.tools.filesystem import write_file, read_file, list_directory, delete_file
from geocongoai.mtumishi.tools.execution import execute_python_code, run_shell_command

# Manipulation du système de fichiers
write_file("scripts/analyse_cuivre.py", "print('Traitement des teneurs...')\n")
contenu = read_file("scripts/analyse_cuivre.py")
fichiers = list_directory("scripts")
print("Fichiers dans scripts/ :", fichiers)

# Exécution de code Python dynamique (capture stdout, stderr, variables)
res_exec = execute_python_code("""
valeurs = [1.2, 2.4, 3.1]
moyenne = sum(valeurs) / len(valeurs)
print(f"Moyenne : {moyenne:.2f}")
""")
print(res_exec["stdout"])        # "Moyenne : 2.23\n"
print(res_exec["result_vars"])   # {'valeurs': [1.2, 2.4, 3.1], 'moyenne': 2.2333333333333334}

# Exécution de commandes Shell (git, pip, GDAL, etc.)
res_shell = run_shell_command("git status")
print("Code de retour :", res_shell["returncode"])
print("Sortie :", res_shell["stdout"])
```

#### 2. Utilisation autonome par l'Agent Mtumishi

```python
from geocongoai import Mtumishi

agent = Mtumishi()

# Mtumishi peut créer un script, initialiser un dépôt git et lancer l'analyse
res = agent.run(
    "Génère un script Python pour calculer le variogramme des sondages, "
    "sauvegarde-le dans scripts/variogramme.py, puis initialise un repo git."
)
print(res.summary)
```

---

### ⚡ Exemple 8 : Streaming en Temps Réel (`stream()`, `GeoTraceEvent`)

> Disponible nativement à partir de **v0.4.3**.

Mtumishi supporte désormais le **streaming d'événements en temps réel**, permettant aux interfaces (CLI REPL, plugins QGIS Desktop, applications Web/Streamlit) d'afficher au fil de l'eau la réflexion interne de l'agent, le déclenchement des outils et la réponse finale :

#### 1. Générateur Asynchrone (`stream()` / `stream_run()`)

```python
import asyncio
from geocongoai import Mtumishi, GeoTraceEventType

async def main():
    agent = Mtumishi(runtime="native")  # ou "antigravity"

    async for event in agent.stream("Analyse la structure géologique du Katanga"):
        if event.type == GeoTraceEventType.THOUGHT:
            print(f"💭 [Pensée] {event.content}")
        elif event.type == GeoTraceEventType.TOOL_CALL:
            print(f"⚙️  [Outil] {event.metadata['tool']}({event.metadata['arguments']})")
        elif event.type == GeoTraceEventType.TOOL_RESULT:
            print(f"✓  [Résultat] {event.metadata['tool']}")
        elif event.type == GeoTraceEventType.TEXT:
            print(event.content, end="", flush=True)

asyncio.run(main())
```

#### 2. Callback Synchrone (`run_with_stream()`)

Pour les interfaces synchrones, threads d'UI ou GUI QGIS :

```python
from geocongoai import Mtumishi

agent = Mtumishi()

def on_trace(event):
    print(f"[{event.type.name}] {event.content}")

result = agent.run_with_stream(
    "Fais une recherche sur les gisements de cuivre",
    on_event=on_trace
)
print("\nRésumé final :", result.final_text)
```

---

### 🛰️ Exemple 9 : GeoTools STAC & Indices Géologiques Gundua (Planetary Computer)

> Disponible nativement à partir de **v0.4.4**. Accès direct au catalogue Microsoft Planetary Computer via STAC sans clé payante requise (`pc.sign`).

Calcul d'indices spectraux optiques et géologiques d'altération hydrothermale (Sentinel-2, Landsat-C2-L2, ASTER L1T) :

```python
from geocongoai import stac_search_scenes, stac_calc_indices

# 1. Recherche de scènes optiques ou radar sur une zone d'intérêt
scenes = stac_search_scenes(
    bbox=[25.40, -10.75, 25.55, -10.65],
    datetime="2024-05-01/2024-09-30",
    collection="sentinel-2-l2a",
    max_cloud_cover=15.0,
    max_results=3,
)

# 2. Calcul d'indices d'altération minière (Oxydes de fer, Argiles Al-OH, Ferreux)
res = stac_calc_indices(
    bbox=[25.40, -10.75, 25.55, -10.65],
    datetime="2024-05-01/2024-09-30",
    indices=["gundua_mining"],  # ou "gundua_hydrothermal_aster" pour ASTER L1T
    collection="sentinel-2-l2a",
    save_cog=True,
    output_dir="./outputs_geologiques",
)
print("Indices calculés :", res["indices_calculated"])
```

---

### 🌱 Exemple 10 : Gundua MSV GeoTool (Metal-Stressed Vegetation)

> Disponible nativement à partir de **v0.4.4**.

Le GeoTool **MSV** analyse par télédétection multitemporelle les signatures géobotaniques induites par des minéralisations sous-jacentes (Cuivre, Cobalt, Or, Nickel) :
- **Indices spectraux fins** : `NDVI`, `NDRE` (Red-Edge 1 B05), `NDMI`, `SWIR_STRESS` (MSI : SWIR1/NIR)
- **Métriques multitemporelles** : Référence saisonnière pixel par pixel, anomalies Z-Score, carte de persistance
- **Indicateur composite & extraction** : `MSV_COMPOSITE` pondéré et délimitation vectorielle de cibles d'exploration minérale (`MSV_TARGETS`)

#### Appel direct via le SDK Python :

```python
from geocongoai import stac_calc_msv

result = stac_calc_msv(
    bbox=[26.15, -10.60, 26.35, -10.45],
    datetime="2023-01-01/2025-12-31",
    indices=["gundua_msv"],
    collection="sentinel-2-l2a",
    save_cog=True,
    output_dir="./msv_outputs",
)

print(f"Cibles géobotaniques détectées : {len(result['targets'])}")
for cible in result["targets"]:
    print(f" • Cible {cible['target_id']} (Confiance: {cible['confidence']}) - Centroid: {cible['centroid']}")
```

#### Déclenchement automatique par Mtumishi en langage naturel :

```python
from geocongoai import Mtumishi

agent = Mtumishi()
agent.run(
    "Analyse la végétation à contraintes métalliques sur cette concession à Fungurume "
    "entre 2023 et 2025. Élimine les nuages, calcule les anomalies de Red-Edge persistantes "
    "et extrait les polygones cibles d'exploration."
)
```

---

## 📦 Publication sur PyPI (Mainteneurs)

```bash
python3 -m build
python3 -m twine check dist/*
python3 -m twine upload dist/*
```
