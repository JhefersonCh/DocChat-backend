# DocChat — backend

Backend de **DocChat** (chat-pdf): RAG multimodal *local-first*. Conversás con tus PDFs, textos, imágenes, audios y videos, y cada respuesta trae **citas verificables** (página, fragmento, minuto o región de imagen).

## Qué hace

- Chat con documentos y medios, con citas resueltas por el backend (salen de las fuentes recuperadas, no del texto del LLM).
- **Proyectos** y **categorías** para organizar fuentes; **búsqueda global** en la biblioteca.
- **Estudio**: artefactos generados a partir de las fuentes (resúmenes, mapas de conceptos, slides, tutor, notas).
- **Voz**: transcripción (STT) y síntesis (TTS) con perfiles configurables.
- Los modelos se eligen desde el frontend; el backend no trae ninguno embebido.

## Arquitectura

Hexagonal liviana, dependencias hacia adentro: `domain <- application <- adapters <- main`.

| Capa | Rol |
|------|-----|
| `domain/` | Reglas puras (chunking, citas, RRF, prompts, calibración). Solo stdlib. |
| `application/` | Puertos (`ports.py`) y casos de uso (una clase, un `execute`). |
| `adapters/` | HTTP (FastAPI), persistencia (Postgres + pgvector), proveedores, extractores, storage. |
| `main.py` / `bootstrap.py` | Composition root: pool, migraciones, routers. |

Stack: FastAPI, PostgreSQL 16 + pgvector, psycopg 3, httpx. Las migraciones SQL (`migrations/`) se aplican **solas al arrancar**. Detalle completo en [`arquitectura.md`](arquitectura.md).

## Conceptos clave

- **Embeddings multimodales (EmbeddingGemma 2):** texto, imagen y audio comparten espacio vectorial.
- **Búsqueda híbrida:** vector + full-text (`tsvector`, config `es_unaccent`) fusionados con **RRF** ponderado.
- **Calibración por modalidad:** los cosenos solo son comparables dentro de una modalidad; cada perfil de embeddings guarda `floor` y `threshold` por modalidad.
- **Small-to-big:** se recupera con segmentos finos y se entrega el contexto más amplio.
- **`find_all` con verificación visual:** para "mostrame todas las X", un modelo de visión confirma cada candidato antes de citarlo.
- **Determinista por defecto:** el LLM solo decide qué buscar y cuánto (tool `search_document`); el resto es testeable sin red.

## Requisitos

- Docker y Docker Compose (para correrlo).
- Python 3.12 y [uv](https://docs.astral.sh/uv/) (solo para desarrollo).
- `ffmpeg` para audio/video si corrés fuera de Docker.

## Inicio rápido

El `docker-compose.yml` público levanta solo **`db`** (pgvector) y **`api`**.

1. Creá un `.env` junto al compose:

   ```env
   POSTGRES_USER=docchat
   POSTGRES_PASSWORD=<una-clave-segura>
   POSTGRES_DB=docchat
   APP_SECRET_KEY=<ver abajo>

   # opcionales
   # DB_PORT=5433            # puerto publicado de Postgres (default 5433)
   # API_PORT=8000           # puerto publicado de la API (default 8000)
   # CORS_ORIGINS=http://localhost:5173
   ```

   Generá `APP_SECRET_KEY` con:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

2. Levantá:

   ```bash
   docker compose up -d --build
   ```

3. Verificá: `http://localhost:8000/api/v1/health` y la documentación en `http://localhost:8000/docs`.

> Si cambiás `APP_SECRET_KEY` después, las API keys ya guardadas no se podrán descifrar: tendrás que volver a cargarlas.

## Modelos: perfiles de proveedor

Los modelos **no se configuran por variables de entorno**. Se crean como **perfiles de proveedor** desde el frontend (o `POST /providers`) y se activan por tipo:

| `kind` | Para qué |
|--------|----------|
| `chat` | Respuestas y generación (tool calling recomendado) |
| `embedding` | Indexación y búsqueda |
| `vision` | Descripción de imágenes y verificación en `find_all` |
| `stt` | Transcripción de audio/video (protocolo `openai` u `ollama`) |
| `tts` | Síntesis de voz |

Todos hablan **OpenAI-compatible** (`base_url` + `model` + `api_key` opcional). `stt` además soporta la API nativa de Ollama, y los embeddings de imagen/audio detectan solos si el servidor es llama.cpp (`/v1/embeddings` con partes tipadas) u Ollama (`/api/embed`); tras crear el perfil de embeddings, usá **Probar** para que registre qué modalidades soporta. Ejemplos:

- **Ollama** (embeddings multimodales): `base_url` `http://<host>:11434/v1`, modelo `embeddinggemma-2`.
- **DeepSeek** (chat): `https://api.deepseek.com/v1` (verificá el nombre de modelo vigente).
- **Supertonic** (TTS): vía un servidor OpenAI-compatible (`/v1/audio/speech`).

**Vos servís los modelos** donde quieras (tu máquina, otra del LAN, la nube) y el perfil apunta ahí. Ojo: desde el contenedor `api`, `localhost` es el propio contenedor; usá el host real o `host.docker.internal`.

Un perfil de embeddings con documentos indexados no puede cambiar `model`, `dimensions` ni prefijos (`409 provider_in_use`): creá uno nuevo.

### Opcional: compose local con embeddings y TTS

`docker-compose.local.yml` (ignorado por git, es de uso personal) puede sumar al stack un servidor **llama.cpp** con EmbeddingGemma 2 y un contenedor **TTS** (Supertonic, ver `tts/`). Si lo armás, el backend siembra perfiles por defecto apuntando a `http://embeddings:8080/v1` y `http://tts:7788/v1` (ajustables con `SEED_EMBEDDING_BASE_URL` / `SEED_TTS_BASE_URL`). Con el compose público esos perfiles sembrados no tendrán servidor detrás hasta que configures los tuyos.

## Desarrollo

```bash
uv sync                                   # dependencias (incluye dev)
docker compose up -d db                   # Postgres local

# Tests: los que usan DB necesitan un Postgres descartable; sin la variable se SALTAN
export TEST_ADMIN_DATABASE_URL=postgresql://USER:PASS@localhost:5433/postgres
uv run pytest -q

uv run ruff check                         # lint

# API en modo desarrollo (necesita DATABASE_URL y APP_SECRET_KEY en el entorno o .env)
uv run uvicorn chat_pdf.main:create_app --factory --reload
```

Cada test con DB crea y borra su propia base (`chatpdf_test_*`) usando esa URL de administración. Las migraciones se aplican automáticamente al iniciar la app.

## Scripts

| Script | Uso |
|--------|-----|
| `scripts/eval_retrieval.py` | Mide hit@k y MRR de la recuperación sobre `scripts/eval_questions.json`. Requiere `EVAL_ADMIN_DATABASE_URL` (Postgres descartable) y un servicio de embeddings. `uv run python scripts/eval_retrieval.py ARCHIVO.pdf [--no-text] [--chunk-size N]` |
| `scripts/calibrate_modalities.py` | Calcula `floor`/`threshold` por modalidad desde pares etiquetados (ver `scripts/calibration_pairs.example.json`); con `--write` los guarda en el perfil de embeddings vía API. |

## Documentación de la API

- [`docs/frontend-api.md`](docs/frontend-api.md): guía de integración para el frontend (endpoints, errores, SSE, tipos TypeScript).
- OpenAPI interactivo en `/docs` y `/openapi.json`.

## Seguridad

- **No hay autenticación.** Mantené los puertos ligados a `127.0.0.1` (así viene el compose) y no expongas la API a Internet ni a la red sin un proxy con auth delante.
- Las API keys de los perfiles se guardan **cifradas** (Fernet, clave derivada de `APP_SECRET_KEY`) y **nunca se devuelven** (solo `api_key_masked`).
- No subas tu `.env` al repositorio.

## Estructura

```
.
├── docker-compose.yml        # db + api (público)
├── Dockerfile
├── migrations/               # SQL, aplicadas al arrancar
├── scripts/                  # evaluación y calibración
├── tts/                      # imagen TTS opcional (Supertonic)
├── docs/frontend-api.md
├── arquitectura.md
├── src/chat_pdf/
│   ├── domain/               # reglas puras
│   ├── application/          # puertos y casos de uso
│   ├── adapters/             # http, persistence, providers, extractors, storage, audio, video, media
│   ├── bootstrap.py          # construye adapters y casos de uso
│   ├── config.py             # Settings (env)
│   └── main.py               # create_app
└── tests/
```
