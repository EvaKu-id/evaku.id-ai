# Architecture

## Place in the system

EvaKu has three repos:

| Repo | Owner | Contents |
| --- | --- | --- |
| `evaku-web` | FE, BE | Next.js UI and main backend, Supabase migrations, `evaku.dbml` |
| `evaku-ai` | AI/ML | this repo: access engine, model API, pipeline, static data |
| `evaku-ml` | ML | `evaku_ml` package (features, radar preprocessing), notebooks, model release |

The Next.js backend on Vercel is the main backend and the only caller of the model API. Browsers
never call the model API, and normal pages never depend on it: only free simulation and on-demand
routes do, and both fall back to precomputed results in R2 when the model API is down.

## Role of this repo

`evaku-ai` owns everything Python that runs in production for EvaKu:

- **Model API** (FastAPI on Cloud Run): interactive simulation and routing, called only by the
  Next.js backend with the `X-Model-Key` header.
- **Pipeline** (GitHub Actions, every 6 hours): forecast and radar runs written to Supabase and R2.
- **Static data** (one-off): road graph, facilities, population, HAND, released to R2.

Model training is NOT here. The `evaku-ml` repo releases model packages to R2 `model/{version}/`
and the shared `evaku_ml` package (features and preprocessing), so features are identical in
training and in the pipeline. This repo pins both.

## Layers

| Layer | Package | May import | Notes |
| --- | --- | --- | --- |
| Entrypoints | `api`, `pipeline`, `static_data` | everything below | wiring only, little logic |
| I/O and models | `adapters`, `inference` | `engine`, `geo`, `domain`, `core` | databases, R2, GEE, Open-Meteo, ONNX, LightGBM |
| Pure logic | `engine`, `geo` | `domain`, `core` | numpy/scipy, no I/O, unit-tested on small graphs |
| Vocabulary | `domain` | `core` | enums, fixed rules, result types |
| Foundation | `core` | nothing in the package | settings, logging, errors |

`lint-imports` enforces this in CI (contracts in `pyproject.toml`).

## Data flow

1. `static_data` builds the graph, HAND, population, and facilities, then releases them to R2
   `data/static/{version}/`.
2. The `api` image loads that bundle into memory at startup. `/simulation` turns a water level
   into road status, reachability, and isolated villages; `/route` returns a land or land + boat
   route.
3. `pipeline` runs forecast (Open-Meteo + GEE features -> LightGBM -> water level -> engine) and
   radar (Sentinel-1 -> U-Net -> depth -> engine), writes one run per radar scene or forecast
   horizon as `draft`, flips it to `published` in the same transaction, uploads flood PMTiles,
   then calls the web app's `/api/pipeline-selesai`.

Forecast, radar, and simulation share one access engine; they differ only in where the flood
depth comes from.

## Contracts and artifacts

- API: `contracts/openapi.json`, generated from `api/schemas.py`. Field names are English and
  match `evaku.dbml`. `evaku-web` generates its TypeScript types from this file.
- Database: schema owned by `evaku-web` (`supabase/migrations`, `evaku.dbml`); this repo only
  writes rows.
- Features: `contracts/fitur.json` in `evaku-ml`, computed by the `evaku_ml` package pinned by tag.
- Model packages: R2 `model/{version}/`, selected with `MODEL_VERSION`, verified against
  `manifest.json` checksums.
- Static data: R2 `data/static/{version}/` (GeoParquet), used by the pipeline, the `api` image,
  and ML for topographic features.
- Other R2 paths: `peta/*.pmtiles` (maps) and `cadangan/...` (fallback scenarios, replay copies).
