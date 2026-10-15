# evaku-ai

AI/ML service for EvaKu: access engine, internal model API (FastAPI), and the scheduled data
pipeline. Model training lives in the ML repo (`evaku-ml`); this repo only loads the released
model packages.

- Architecture and data flow: `docs/architecture.md`
- Rules for contributors and AI assistants: `md/RULES.MD`
- Task checklist and progress: `md/TASK.MD`

## Quick start

```bash
cp .env.example .env   # then fill MODEL_API_KEY, POSTGRES_PASSWORD, PIPELINE_SECRET
                       # with `openssl rand -hex 32`, and SUPABASE_DB_URL (format in the file)
make install           # uv sync: api extra + dev tools (Python 3.11)
make check             # lint, types, security, tests + coverage
make api               # http://localhost:8000/docs
```

Secrets only ever live in `.env` (git-ignored) or the production secret stores.

With Docker:

```bash
make dev                                              # api with auto-reload on http://localhost:8000
make up                                               # production api image on http://localhost:8000
docker compose --profile db up -d db                  # local PostGIS on 127.0.0.1:5432
docker compose --profile pipeline run --rm pipeline   # pipeline (needs the `pipeline` extra)
make down
```

## Commands

| Command | What it does |
| --- | --- |
| `make install` / `make install-all` | Install the `api` extra + dev tools / every extra + dev tools |
| `make check` | Everything below that CI runs: lint, types, security, tests + coverage |
| `make lint` | ruff (incl. security rules), format check, import layers, task-ID check |
| `make typecheck` | mypy, with explicit `Any` disallowed |
| `make security` | pip-audit and gitleaks |
| `make test` / `make coverage` | Unit tests / unit tests with coverage |
| `make test-integration` | Integration tests against a local PostGIS |
| `make format` | Fix lint issues and format with ruff |
| `make openapi` | Regenerate `contracts/openapi.json` |
| `make api` / `make dev` / `make up` / `make down` | Run the API locally / in Docker with reload / production image / stop all |

## Layout

```
src/evaku_ai/
  core/         settings, logging, errors
  domain/       enums, fixed rules (passability thresholds, score weights), result and array types
  engine/       access engine, pure numpy/scipy
  geo/          CRS, raster sampling, HAND
  inference/    thin wrappers that run the released models
  adapters/     all I/O: Postgres, R2, GEE, Open-Meteo, Next.js callback, Healthchecks
  api/          FastAPI app, X-Model-Key auth, contract schemas, routers
  pipeline/     scheduled run: forecast, radar, write, export, notify
  static_data/  one-off builds: graph, facilities, population, HAND, tiles, release
config/regions/ region settings (bbox, rivers, thresholds, weights)
contracts/      openapi.json consumed by evaku-web
docs/           architecture notes and ADRs
md/             rules, product requirements, task checklist
```

Layer rule (checked by `lint-imports`): `domain`, `engine`, and `geo` never import
`adapters`, `inference`, `api`, `pipeline`, or `static_data`.

## API contract

`contracts/openapi.json` is generated from `src/evaku_ai/api/schemas.py`:

```bash
make openapi
```

A unit test fails when the committed file is out of date. `evaku-web` generates its TypeScript
types from this file. All endpoints except `/health` require the `X-Model-Key` header.

| Endpoint | Status |
| --- | --- |
| `GET /health` | done |
| `POST /simulation` | done; loads `data/static/{DATA_VERSION}/` at startup (production refuses to start without it; locally it answers 503, and `ENGINE_DATA=synthetic` serves a generated demo network) |
| `POST /route` | returns 501 for now |
| `POST /recalc` (Should) | returns 501 for now |

## Dependencies

| Extra | Used by | Contents |
| --- | --- | --- |
| (core) | everything | numpy, scipy, pandas, pydantic, httpx, pyyaml |
| `api` | Cloud Run image | fastapi, uvicorn |
| `geo` | pipeline, static data | geopandas, shapely, pyogrio, rasterio, pyarrow, zstandard |
| `pipeline` | GitHub Actions | `geo` + earthengine-api, onnxruntime, lightgbm, psycopg, boto3, osmnx, scikit-image, typer |

Versions are locked in `uv.lock`. Add a dependency with `uv add --optional <extra> <package>`.

## Data and models

`data/` (raw, interim, processed) and `artifacts/` (downloaded model packages) are git-ignored.
Static data is released to R2 `data/static/{version}/`; model packages come from R2
`model/{version}/`, selected by `MODEL_VERSION`.

## Earth Engine access

The project is registered for noncommercial Earth Engine use (Community tier, no billing needed).
No service account key file is ever created.

- Laptop: sign in with your own Google account, then run a test call.

  ```bash
  uv run --extra pipeline python -c "import ee; ee.Authenticate()"
  uv run --extra pipeline python -c "import ee; ee.Initialize(project='<GCP_PROJECT_ID>'); \
    print(ee.ImageCollection('COPERNICUS/S1_GRD').filterDate('2025-11-28','2025-11-29').size().getInfo())"
  ```

  A number printed means access works. Credentials stay in `~/.config/earthengine/`; never commit or share them.
- GitHub Actions: the pipeline signs in as the `evaku-pipeline` service account through Workload Identity
  Federation (no JSON key). Its email goes in `GEE_SERVICE_ACCOUNT`.

## Versions used for the demo

Record the combination here and in the `evaku-web` README, e.g.
`web v1.0 · ai api-v0.4.0 · evaku_ml v0.2.0 · model v3 · data v2`.
