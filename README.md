# geocongoai_mtumishi_worker

Worker Python prive pour executer `Mtumishi` avec `AntigravityRuntime` cote serveur, sans jamais exposer `GEMINI_API_KEY` au plugin QGIS ni au SDK client.

## Role

- sonder la table `agent_runs`
- prendre un job `queued`
- executer `geocongoai.Mtumishi`
- ecrire `summary`, `steps`, `qgis_scripts`, `result_json`
- finaliser la facturation via RPC Supabase

## Structure

- `worker/main.py` : boucle principale
- `worker/config.py` : chargement de la configuration
- `worker/supabase_ops.py` : acces base et RPC
- `worker/runtime.py` : execution de Mtumishi
- `worker/models.py` : types legers
- `systemd/geocongoai-mtumishi-worker.service` : exemple de service VPS

## Variables d'environnement

Copier `.env.example` en `.env` sur le VPS, puis completer les valeurs.

Variables obligatoires :

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `GEMINI_API_KEY`

## Lancement local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m worker.main
```

## Lancement Docker

```bash
docker build -t geocongoai-mtumishi-worker .
docker run --env-file .env geocongoai-mtumishi-worker
```

## Deploiement sur Coolify

Le plus simple pour ce worker est d'utiliser le fichier `docker-compose.coolify.yml`.

Pourquoi :

- le worker n'expose pas d'API HTTP publique
- il n'a besoin ni de domaine ni de port public
- il doit juste tourner en boucle et parler a Supabase en sortant

### Option recommandee : Docker Compose prive

1. pousse le depot `geocongoai_mtumishi_worker` sur GitHub
2. dans Coolify, cree une nouvelle ressource de type `Application`
3. choisis ton depot Git
4. selectionne le build pack `Docker Compose`
5. indique `docker-compose.coolify.yml` comme fichier Compose
6. n'ajoute aucun domaine
7. n'expose aucun port
8. ajoute les variables d'environnement suivantes dans Coolify :
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `GEMINI_API_KEY`
   - `MTUMISHI_QGIS_RPC_PORT`
   - `MTUMISHI_WORKER_ID`
   - `MTUMISHI_POLL_SECONDS`
   - `MTUMISHI_LEASE_SECONDS`
   - `MTUMISHI_DEFAULT_RESERVED_UNITS`
   - `MTUMISHI_MODEL_NAME`
   - `MTUMISHI_WORKSPACES`
9. lance le deploiement
10. verifie les logs jusqu'a voir `Worker started with id=...`

### Valeurs minimales conseillees

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-api-key
MTUMISHI_WORKER_ID=mtumishi-worker-prod-01
MTUMISHI_POLL_SECONDS=2
MTUMISHI_LEASE_SECONDS=120
MTUMISHI_DEFAULT_RESERVED_UNITS=12
MTUMISHI_MODEL_NAME=gemini-2.5-flash
MTUMISHI_WORKSPACES=.
```

### Points importants

- ne mets jamais `GEMINI_API_KEY` dans le depot Git
- garde `GEMINI_API_KEY` uniquement dans les variables runtime Coolify
- le plugin QGIS et le SDK client ne doivent connaitre que `GEOCONGOAI_API_KEY`
- si le worker doit acceder a des chemins specifiques sur ton serveur, il faudra ajouter des volumes Docker explicitement

### Quand preferer Dockerfile au lieu de Docker Compose

Tu peux aussi deployer ce depot avec le build pack `Dockerfile`, mais pour un worker prive sans port public, `Docker Compose` est souvent plus lisible et plus simple a maintenir dans Coolify.

## Git

Le dossier est initialise comme depot Git independant pour faciliter un push vers GitHub puis un deploiement sur VPS.
