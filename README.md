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

## Git

Le dossier est initialise comme depot Git independant pour faciliter un push vers GitHub puis un deploiement sur VPS.
