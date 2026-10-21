# chat-pdf — Arquitectura y ruta de consolidación

Complementa a `plan.md` (qué construimos). Este documento define **cómo se organiza** el código para que cada pieza tenga un lugar, un contrato y una razón de existir.

> No usamos LangChain ni LlamaIndex como librería, pero **sí adoptamos su vocabulario**: cualquiera que conozca un framework RAG tiene que reconocer las piezas. Lo "liviano" no exime de estructura: si no la trae un framework, la define este documento.

---

## 1. Principios

1. **Vocabulario estándar de RAG.** Cada etapa del pipeline tiene un nombre reconocible y un contrato (§3).
2. **Dirección de dependencias.** `domain ← application ← adapters ← main`. `domain` solo depende de la stdlib. `application` no conoce FastAPI, psycopg ni httpx.
3. **Determinista por defecto, el modelo decide solo donde se lo permitimos.** El único punto donde el LLM decide es la tool `search_document` (qué buscar y cuánto). Todo lo demás (chunking, límites, numeración, citas, persistencia) es determinista y testeable sin red.
4. **Una regla, un lugar.** Cada regla de negocio y cada valor por defecto se definen una sola vez.
5. **Routers delgados.** Un endpoint parsea, llama a un caso de uso y mapea la respuesta. Nada más.
6. **Un composition root.** `main.py` construye todo. Nadie más instancia adapters.

---

## 2. Capas

| Capa | Contiene | No puede |
|------|----------|----------|
| `domain/` | Entidades, value objects, reglas puras: chunking, numeración y citas, recorte de k, armado de prompt, `RagConfig`, errores de dominio | Hacer IO, importar fuera de la stdlib, saber de JSON/HTTP |
| `application/` | **Puertos** (`ports.py`, único lugar) y **casos de uso** (un caso de uso = una clase con `execute`) | Importar adapters, FastAPI, psycopg, httpx |
| `adapters/` | Implementaciones de puertos: `http/` (driver), `persistence/`, `providers/`, `extractors/`, `storage/` | Contener reglas de negocio |
| `main.py` + `bootstrap.py` | Composition root: `main.py` arma la app (pool, migraciones, routers); `bootstrap.py` construye adapters y casos de uso | Contener lógica (`/health` también va a un router) |

---

## 3. Pipeline RAG (etapas y contratos)

### 3.1 Indexación

```
Upload ──► Loader ──► Cleaner ──► TextSplitter ──► Embedder ──► VectorStore (+ FTS)
          (Extractor)  (domain)     (domain)        (port)       (port)
```

| Etapa | Nombre estándar (LangChain / LlamaIndex) | Nuestro contrato | Tipo | Determinista |
|-------|------------------------------------------|------------------|------|--------------|
| Carga | DocumentLoader / Reader | `Loader.load(path) -> list[Page]` (el PDF marca `Page.headings` por tamaño de fuente) | puerto (uno por MIME) | Sí |
| Limpieza | DocumentTransformer | `strip_boilerplate(pages)`: quita encabezados/pies repetidos (≥ 50 % de las páginas) y números de página | función de dominio (`cleaning.py`) | Sí |
| División | TextSplitter / NodeParser | `chunk_pages(pages, cfg) -> list[Chunk]`: recursivo por estructura (título → párrafo → oración → línea → palabra), nunca cruza un título; cada chunk lleva `section` e `index_text` | función de dominio | Sí |
| Embedding | Embeddings / BaseEmbedding | `Embedder.embed(texts)`; texto = `document_prefix` con `{title}` = sección (`none` fuera de sección) + `index_text` | puerto | Sí (el modelo es fijo por perfil) |
| Almacenamiento | VectorStore | `ChunkStore.replace(document_id, chunks, vectors)` (borra los anteriores: re-index) | puerto | Sí |

`content` queda **literal** (el visor lo resalta en la capa de texto del PDF); `index_text` es su versión normalizada (espaciado de letras `S E C C I Ó N` → `SECCIÓN`, guiones de corte de línea, espacios) y es lo que se embebe y se indexa en full-text (`tsvector` generado, config `es_unaccent`, migración 002).

### 3.2 Consulta

```
Pregunta ──► Agent loop ──► Retriever ──► SourceSet ──► PromptBuilder ──► Generator ──► CitationResolver
           (tool search)    (port)        (domain)       (domain)          (ChatModel)     (domain)
```

| Etapa | Nombre estándar | Nuestro contrato | Determinista |
|-------|-----------------|------------------|--------------|
| Planificación | Agent / Tool calling | `RetrievalPlanner`: tool loop con límites | **El modelo decide** `query` y `k`; el backend impone límites |
| Recuperación | Retriever (híbrido) | `Retriever.candidates(document_id, query, vector, n) -> Candidates(by_vector, by_text)` en un SQL; `DocumentSearch` las fusiona con `reciprocal_rank_fusion` (dominio, ponderada por `text_weight`), pasa por el `Reranker` y corta en `k` | Sí |
| Reordenamiento | Reranker / NodePostprocessor | `Reranker.rerank(query, chunks)`; hoy `IdentityReranker` (no-op). Un cross-encoder local midió ~3,8 s por candidato en CPU: no es viable interactivo | Sí |
| Ensamblado de fuentes | Postprocessor | `SourceSet`: deduplica, numera `[1..n]` estable; `fit_sources` recorta al presupuesto `max_context_chars` (prefijo, así la numeración sigue consecutiva) | Sí |
| Prompt | PromptTemplate | `build_answer_messages(sources, history, question, cfg)` | Sí |
| Generación | ChatModel / ResponseSynthesizer | `ChatModel.stream / complete` | No (LLM) |
| Citas | — | `resolve_citations(answer, sources)` | Sí |

**Regla de oro:** las citas salen del `SourceSet`, no del texto del LLM. El LLM solo puede *señalar* cuáles usó.

### 3.3 Puntos de extensión (fases 2 y 3)

| Necesidad futura | Cómo encaja | Qué NO cambia |
|------------------|-------------|---------------|
| Imagen / audio / video | Nuevos `Loader` (OCR, Whisper) o un `Embedder` multimodal | Splitter, Retriever, citas |
| Citas por timestamp (audio) | `Page` → `Segment` con `locator` genérico (página **o** tiempo) | Pipeline |
| Versión B (multi-documento) | `Retriever.search(collection_id, ...)`; se borra la regla `OneDocumentPerChat` | Citas (ya llevan `filename`) |
| Re-index | ✅ `ReindexDocument` (`POST /documents/{id}/reindex`): mismo perfil, re-ejecuta `IngestDocument`; con otro perfil sería el mismo camino | Puertos |
| Búsqueda híbrida / reranker | ✅ vector + full-text fusionados por RRF; `Reranker` es el enchufe (adapter `/v1/rerank` en `bootstrap.py`) | Casos de uso |
| Cola de ingesta | Otro driver (worker) llama al mismo `IngestDocument` | Caso de uso |

---

## 4. Contratos (`application/ports.py`, único lugar)

| Puerto | Implementación |
|--------|----------------|
| `Loader` | `adapters/extractors/{pdf,text}.py` |
| `Embedder`, `ChatModel`, `ProviderFactory` | `adapters/providers/openai_compatible.py` |
| `ProviderRepository` (CRUD + perfiles activos) | `adapters/persistence/providers.py` |
| `ChatRepository`, `DocumentRepository` | `adapters/persistence/{chats,documents}.py` |
| `ChunkStore` + `Retriever` | `adapters/persistence/chunks.py` (`PgChunkStore`, una clase para ambos puertos; pgvector + full-text) |
| `Reranker` | `adapters/providers/reranking.py` (`IdentityReranker`, no-op) |
| `UnitOfWork` | `adapters/persistence/uow.py` (una conexión, una transacción) |
| `FileStorage` (`save`, `delete`, `path`) | `adapters/storage/local.py` |
| `Database` (`ping`, para `/health`) | `adapters/persistence/connection.py` (`PgDatabase`) |

`ProviderFactory` expone además `reachable(profile)` (chequeo de liveness para `/health`). Los repos Pg reciben un `ConnectionSource`: el pool (una transacción por llamada) o una `BoundConnection` que entrega la `UnitOfWork`.

Los repositorios devuelven **tipos de dominio**, nunca dicts. Nombres: `Pg<Entidad>Repository`; métodos `get`, `list`, `create`, `update`, `delete`.

---

## 5. Configuración: tres niveles, una fuente cada uno

| Nivel | Dónde | Qué | Quién lo cambia |
|-------|-------|-----|-----------------|
| Infraestructura | `config.Settings` (env) | `DATABASE_URL`, `APP_SECRET_KEY`, `CORS_ORIGINS`, `UPLOAD_DIR`, timeouts HTTP | Despliegue |
| Proveedor | `provider_profiles` (DB) | URL, modelo, key, prefijos, `default_k`, `max_k`, `max_tool_calls` | Frontend vía API |
| Pipeline | `domain/config.py` → `RagConfig` (dataclass inmutable) | `chunk_size`, `chunk_overlap` (validados en `__post_init__`), `search_candidates`, `rrf_k`, `text_weight`, `embed_batch`, `history_limit`, `snippet_chars`, `max_error_chars`, `max_context_chars` | Código (y en fase 3, admin) |

- Los defaults de perfil (`default_k=5`, `max_k=15`, `max_tool_calls=3`) se definen **una vez** en `domain/providers.py`; el schema HTTP y la migración se alinean a eso.
- Batching de embeddings: **solo** en el adapter `Embedder` (el caso de uso no re-agrupa).
- Prompts del sistema y descripción de la tool: en `domain/prompts.py`, **en un solo idioma** (español, coherente con los usuarios).

---

## 6. Errores: una jerarquía, un formato

```
DomainError(code, message)
├── NotFound            → 404   (chat_not_found, document_not_found, file_not_found, provider_not_found)
├── Conflict            → 409   (chat_has_document, document_not_ready, no_chat_profile, provider_in_use, ...)
├── InvalidInput        → 422   (kind_mismatch, invalid_k, unsupported_file_type → 415)
└── ProviderError       → 502   (provider_unreachable)  · StreamingUnsupported / ToolsUnsupported son internos
```

- Errores definidos en `domain/errors.py`; los lanzan casos de uso y repos. El dominio no conoce HTTP: el mapeo clase → status (y el override `unsupported_file_type` → 415) vive en `adapters/http/errors.py`.
- **Un único** exception handler en `adapters/http/errors.py` → siempre `{"code": "...", "message": "..."}`.
- El evento SSE `error` usa el mismo `{code, message}`.
- Prohibido `HTTPException` y `JSONResponse` de error en los routers.

---

## 7. Casos de uso

| Caso de uso | Responsabilidad | Transacción |
|-------------|-----------------|-------------|
| `CreateChat` | Colección + chat | UoW |
| `DeleteChat` | Borra en cascada + archivos (`FileStorage`) | UoW, archivos después del commit |
| `UploadDocument` | Valida tipo, aplica `OneDocumentPerChat` (y reemplazo de `failed`), guarda archivo, crea documento con perfil activo | UoW; si falla, borra el archivo |
| `IngestDocument` | Loader → Cleaner → Splitter → Embedder → ChunkStore → `ready` | **Reemplazo de chunks + estado `ready` en una sola transacción** |
| `ReindexDocument` | Valida documento y archivo, pasa a `processing` de forma atómica (`mark_processing`; si ya está ocupado → `409 document_busy`); el router encola `IngestDocument` | — |
| `AskQuestion` | Planner → Retriever → SourceSet → Generator → citas → persiste | Persiste el intercambio (pregunta + respuesta, un solo `add_exchange`) al final; si falla a mitad no se persiste nada (*persistir la pregunta con estado de error queda como mejora futura: sería una feature*) |
| `ManageProviders` | CRUD, valida `default_k ≤ max_k` (regla de dominio) y que un perfil de embeddings con documentos no cambie su espacio vectorial (`model`, `dimensions`, prefijos → `provider_in_use`). Clase con varios métodos (`list/create/update/delete/active_ids`), no `execute` | — |
| `ProbeProvider` | Dimensión / soporte de tools | — |
| `SetActiveProfiles` | Valida que el `kind` coincida | — |
| `SeedDefaults` | Perfil de embedding local si no hay ninguno | — |
| `FailInterruptedIngests` | Al arrancar: `pending`/`processing` → `failed` (la ingesta corre en proceso y un reinicio la corta) | — |
| `CheckHealth` | DB + perfiles activos | — |

Casos de uso de lectura (sin transacción, un `execute` cada uno): `ListChats`, `GetChat`, `ListMessages`, `GetDocument`. Existen para que los routers no toquen repositorios.

Reglas de negocio con nombre propio, en dominio: `OneDocumentPerChat`, `clamp_k`, `SourceSet` (dedup y numeración), `validate_k_range`, `resolve_upload_mime` (tipos de archivo admitidos), `fit_sources` (presupuesto de contexto), `strip_boilerplate` / `normalize` (limpieza), `reciprocal_rank_fusion`, y `parse_cited` / `strip_citations` (citas, en `prompting.py`).

Una subida cuyo chat no existe o ya tiene documento se rechaza *después* de guardar el archivo (la validación y las escrituras van en una sola transacción); el archivo se borra al fallar.

---

## 8. HTTP

- `adapters/http/schemas.py`: **todos** los modelos pydantic de entrada y salida (también los de chats; nada de dicts armados a mano).
- `adapters/http/deps.py`: única forma de obtener casos de uso (`Depends(get_ask_question)`, ...); es el único módulo HTTP que lee `app.state.use_cases`. Los routers **no** leen `app.state`.
- `adapters/http/errors.py`: handler único (§6).
- `adapters/http/sse.py`: formato SSE.
- `/health` vive en `adapters/http/health.py` y llama a `CheckHealth`.
- `create_app(settings, provider_factory=None)`: los tests inyectan un `ProviderFactory` falso; no se parchea `app.state`.

---

## 9. Testing

| Nivel | Qué | Con qué |
|-------|-----|---------|
| Dominio | Chunking, SourceSet, clamp_k, citas, prompt | Puro, sin fakes |
| Aplicación | Casos de uso | **Un solo** `tests/fakes.py` con adapters en memoria que implementan los puertos |
| Integración | Flujo completo HTTP + Postgres real | Los mismos fakes, solo para los proveedores |

**Evaluación de recuperación** (fuera de la suite): `uv run python scripts/eval_retrieval.py ARCHIVO.pdf [--chunk-size N] [--no-text] [--query-prefix ...]` ingesta el archivo con el pipeline real (servicio de embeddings + base temporal) y mide hit@1/3/5 y MRR sobre `scripts/eval_questions.json` (24 consultas del plan de negocios de prueba: 18 preguntas parafraseadas + 6 tipo palabra clave). Resultados (all, hit@1/hit@3/hit@5/MRR): base 3200/400 vectorial 0.50/0.75/0.92/0.667 · troceo por secciones 1400/210 vectorial 0.75/0.92/0.92/0.825 · + híbrida (`text_weight` 0.3, `rrf_k` 10) 0.79/0.88/0.96/0.847. Con RRF k=60 y peso 1 la híbrida **empeoraba** las preguntas parafraseadas (0.56 hit@1). El TOC no domina con chunks chicos (no se penaliza).

Huecos a cubrir (un test cada uno, sin redundancia): fallo de ingesta → `failed` y sin chunks huérfanos; reemplazo de documento `failed`; `ProbeProvider` detecta tools; formato de error único.

---

## 10. Estructura objetivo

```
src/chat_pdf/
├── domain/
│   ├── models.py        # Page, Chunk, Document, Source, Chat, Message
│   ├── providers.py     # ProviderProfile (+ defaults únicos)
│   ├── config.py        # RagConfig
│   ├── errors.py        # DomainError y subclases
│   ├── cleaning.py      # encabezados/pies, normalización para embeddings y FTS
│   ├── chunking.py      # TextSplitter (secciones + recursivo)
│   ├── retrieval.py     # SourceSet, clamp_k, fit_sources, reciprocal_rank_fusion
│   ├── prompts.py       # textos de sistema y tool (un idioma)
│   ├── prompting.py     # build_answer_messages, resolve_citations
│   └── rules.py         # OneDocumentPerChat, validate_k_range, resolve_upload_mime
├── application/
│   ├── ports.py         # TODOS los puertos
│   ├── chats.py         # CreateChat, DeleteChat
│   ├── documents.py     # UploadDocument, IngestDocument
│   ├── ask.py           # AskQuestion (+ RetrievalPlanner)
│   └── providers.py     # ManageProviders, ProbeProvider, SetActiveProfiles, SeedDefaults, CheckHealth
├── adapters/
│   ├── http/            # routers (chats, providers, health), schemas.py, deps.py, errors.py, sse.py
│   ├── persistence/     # Pg*Repository, chunks.py, connection.py, uow.py, migrations.py
│   ├── providers/       # openai_compatible.py
│   ├── extractors/      # Loaders
│   └── storage/         # local.py (FileStorage)
├── config.py            # Settings (env)
├── crypto.py
├── bootstrap.py         # construye adapters y casos de uso
└── main.py              # composition root (app, pool, routers)
```

---

## 11. Ruta de consolidación (Fase 1.5)

Orden pensado para que cada paso deje el sistema funcionando y los tests en verde. **No agrega features.**

| # | Paso | Resuelve | Estado |
|---|------|----------|--------|
| 1 | `domain/errors.py` + handler único + migrar routers | 5 formatos de error → 1 | ✅ |
| 2 | `RagConfig` + defaults únicos de perfil; prompts a `domain/prompts.py` en español | Constantes repetidas en 3 lugares, batch doble, idiomas mezclados | ✅ |
| 3 | Todos los puertos a `ports.py` (`Loader`, repos, `ChunkStore`/`Retriever`, `UnitOfWork`, `FileStorage`) | Protocols locales en `ask.py`/`ingest.py`, `repo` sin tipo | ✅ (`UnitOfWork`, `FileStorage` y `Database` completados en los pasos 5 y 6) |
| 4 | Repos devuelven tipos de dominio, naming `Pg*Repository` | dicts vs dataclasses, `list_all` vs `list` | ✅ |
| 5 | Casos de uso de §7; mover las reglas del router al dominio y la aplicación | Reglas de negocio en `chats.py` y `providers.py` | ✅ |
| 6 | `UnitOfWork` en ingesta y subida | Chunks huérfanos, subida no atómica | ✅ |
| 7 | `deps.py` como única vía de DI; schemas pydantic para chats; `/health` a un router | `app.state` en routers, dicts armados a mano | ✅ |
| 8 | Un solo `tests/fakes.py`; tests de los huecos de §9 | Fakes duplicados, huecos de cobertura | ✅ |
| 9 | Borrar código muerto (`get_pool` sin uso) y decidir `sha256` (deduplicación) y `chats.title` | Cosas sueltas | ✅ |

**Para la entrega de mañana:** los pasos 1, 2 y 5 son los que más se notan en una revisión de código (errores coherentes, sin constantes mágicas, routers delgados). El resto puede ir después sin cambiar el contrato de la API, salvo el paso 1, que unifica el formato de error y conviene cerrar **antes** de que el frontend dependa del formato actual.

**Decisiones del paso 9**
- `documents.sha256`: se sigue guardando, **sin** deduplicación. Candidato a **Fase 3** (detectar el mismo archivo en otro chat y reutilizar chunks).
- `chats.title`: al subir un documento, si el chat no tiene título se le asigna el nombre del archivo (así `GET /chats` es útil).
- Nota de contrato: los `created_at` ahora salen serializados por pydantic (`...Z`) en vez de `...+00:00`; es el mismo instante en ISO 8601.
