# chat-pdf: guía de integración para el frontend

Guía autosuficiente para quien consume la API sin haber visto el backend. Todo lo de acá se verificó contra el código y contra respuestas reales (`curl`) de la API corriendo.

## 1. Qué es y arquitectura

- Backend RAG: subís **un documento por chat** (PDF, TXT o MD), se indexa en segundo plano y chateás con preguntas que se responden **con citas** al documento.
- **El frontend nunca llama al LLM.** Configura proveedores (chat y embeddings) vía API y chatea contra el backend por **SSE**.
- Las API keys **no vuelven nunca** en las respuestas: solo `api_key_masked` (ej. `sk-****3456`, o `null` si no hay key).
- Sin autenticación (MVP, solo local). Todos los ids son UUID string, las fechas ISO 8601 UTC con `Z`.

| Dato | Valor |
|---|---|
| Base URL | `http://localhost:8000/api/v1` |
| OpenAPI / Swagger | `http://localhost:8000/docs` · `http://localhost:8000/openapi.json` |
| CORS | env del backend `CORS_ORIGINS` (lista separada por comas). Default (config.py): `http://localhost:3000` |

**CORS:** si tu dev server no corre en `http://localhost:3000` (ej. Vite usa `http://localhost:5173`), pedile a quien opera el backend que ponga `CORS_ORIGINS=http://localhost:5173` (o varios: `http://localhost:3000,http://localhost:5173`) en `backend/.env` y reinicie la api (`docker compose up -d`). Se compara el origin exacto (esquema + host + puerto). Métodos y headers están abiertos (`*`).

## 2. Flujo de pantallas sugerido

1. **Configuración de proveedores** (una vez): crear perfil de chat, probarlo, activarlo. Mostrar `GET /health` como indicador (`providers.chat` es `null` si no hay chat activo; `reachable: false` si el proveedor no responde).
2. **Lista de chats**: `GET /chats` (más nuevo primero), botón "Nuevo chat" (`POST /chats`), borrar (`DELETE`). `title` es `""` hasta subir el primer documento; después es el nombre del archivo del documento actual (si se reemplaza un documento fallido, el título pasa al nuevo).
3. **Chat**: `GET /chats/{id}` (trae `document` o `null`) + `GET /chats/{id}/messages` (historial).
   - `document == null` → zona de upload.
   - `document.status` `pending|processing` → spinner con polling a `GET /documents/{id}`.
   - `failed` → mostrar `error` y permitir re-subir (reemplaza al fallido).
   - `ready` → input habilitado; enviar con `POST /chats/{id}/messages` (SSE).

## 3. Configurar DeepSeek como modelo de chat

> Verificá en la documentación de DeepSeek los nombres vigentes de modelo y la base URL antes de configurar. Al momento de escribir esto: base URL `https://api.deepseek.com/v1` (compatible OpenAI) y modelo `deepseek-chat`. Cualquier proveedor compatible con OpenAI (`/chat/completions`) se configura igual (OpenAI, Groq, OpenRouter, Ollama, etc.). El backend concatena `base_url` + `/chat/completions`, así que `base_url` va **sin** el path final.

**Paso 1. Crear el perfil**

```bash
curl -X POST http://localhost:8000/api/v1/providers -H 'content-type: application/json' -d '{
  "kind": "chat",
  "name": "DeepSeek",
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "api_key": "sk-..."
}'
```
→ `201` con el `ProviderOut` (guardá el `id`; `api_key_masked` en vez de la key).

**Paso 2. Probar** `POST /providers/{id}/test` → `200` siempre (aunque falle; mirá `ok`):

```json
{"ok": true, "detail": "chat ok", "supports_tools": false}
{"ok": false, "detail": "ConnectError: All connection attempts failed"}
```
`supports_tools` indica si el modelo soporta *function calling*. El backend lo detecta y **lo guarda en el perfil**. Si es `true`, el modelo decide él mismo qué buscar en el documento (reescribe la consulta y elige `k`, hasta `max_tool_calls` búsquedas). Si es `false`, se busca directo con la pregunta del usuario y `default_k`. Funciona igual en ambos casos; con tools suele responder mejor.

**Paso 3. Activar** `PUT /settings/active`:

```bash
curl -X PUT http://localhost:8000/api/v1/settings/active -H 'content-type: application/json' \
  -d '{"chat_profile_id": "<id>"}'
```
→ `{"chat_profile_id":"<id>","embedding_profile_id":"<id>"}`. Solo cambian los campos presentes en el body; `null` explícito desactiva. Sin chat activo, chatear devuelve `409 no_chat_profile`.

**Campos de ajuste del perfil de chat** (opcionales):

| Campo | Default | Qué hace |
|---|---|---|
| `default_k` | 5 | Cantidad de fragmentos recuperados cuando no hay tools (o el modelo no indica `k`) |
| `max_k` | 15 | Tope de fragmentos por búsqueda (el backend recorta cualquier `k` pedido por el modelo) |
| `max_tool_calls` | 3 | Máximo de búsquedas por pregunta cuando `supports_tools` es `true` |

Regla: `default_k <= max_k`, si no → `422 invalid_k` (en `POST` se reporta como `invalid_input`, ver §9).

**Embeddings:** ya existe un perfil sembrado y activo, `EmbeddingGemma 2 (local)` (768 dims, servido por llama.cpp en el compose). No hace falta configurar nada. Cada documento queda indexado con el perfil de embeddings que estaba activo **al subirlo**, y las preguntas usan ese mismo perfil (los vectores tienen que ser comparables). Por eso cambiar el embedding activo solo afecta a documentos **nuevos**; los existentes siguen con el suyo, y ese perfil no se puede borrar mientras haya documentos (`409 provider_in_use`).

## 4. Referencia de endpoints

Todos bajo `/api/v1`. Errores siempre `{code, message}` (ver §7). Los `{id}` son UUID; uno mal formado da `422 invalid_input`.

| Método | Path | Body | OK |
|---|---|---|---|
| GET | `/health` | none | 200 |
| GET | `/providers` | none | 200 `Provider[]` |
| POST | `/providers` | `ProviderIn` JSON | 201 `Provider` |
| PATCH | `/providers/{id}` | `ProviderPatch` JSON | 200 `Provider` |
| DELETE | `/providers/{id}` | none | 204 |
| POST | `/providers/{id}/test` | none | 200 `ProbeResult` |
| GET | `/settings/active` | none | 200 `ActiveSettings` |
| PUT | `/settings/active` | `ActiveSettings` JSON | 200 `ActiveSettings` |
| POST | `/chats` | none | 201 `Chat` |
| GET | `/chats` | none | 200 `Chat[]` (nuevo primero) |
| GET | `/chats/{id}` | none | 200 `ChatDetail` |
| DELETE | `/chats/{id}` | none | 204 (borra mensajes y documento) |
| GET | `/chats/{id}/messages` | none | 200 `Message[]` (cronológico) |
| POST | `/chats/{id}/documents` | multipart `file` | 202 `UploadResult` |
| GET | `/documents/{id}` | none | 200 `Document` |
| GET | `/documents/{id}/file` | none | 200 archivo original (`inline`); `Range` → 206 |
| GET | `/segments/{id}/thumb` | none | 200 webp 256 px del segmento multimedia (`immutable`); 404 si no tiene |
| GET | `/segments/{id}/asset` | none | 200 webp ≤1024 px (recorte/frame) del segmento; 404 si no tiene |
| POST | `/documents/{id}/reindex` | none | 202 `{document_id, status: "processing"}` |
| POST | `/documents/{id}/cancel` | none | 202 `{document_id, status: "processing"}`; 404 `document_not_found`; 409 `document_not_busy`. El documento termina `failed` con error "Ingesta cancelada" |
| POST | `/chats/{id}/messages` | `{"content": string (min 1)}` | 200 `text/event-stream` |

### Health
`GET /health` → `200`
```json
{"status":"ok","providers":{"chat":{"name":"DeepSeek","reachable":true},"embedding":{"name":"EmbeddingGemma 2 (local)","reachable":true}}}
```
`providers.chat`/`embedding` son `null` si no hay perfil activo de ese tipo. Si la base de datos está caída: `503 database_unavailable`.

### Providers
`POST /providers` body (`ProviderIn`):

| Campo | Tipo | Req | Notas |
|---|---|---|---|
| `kind` | `"chat"` \| `"embedding"` | sí | inmutable |
| `name`, `base_url`, `model` | string (min 1) | sí | |
| `api_key` | string \| null | no | nunca se devuelve |
| `dimensions` | int >= 1 | no | solo embedding; se autodetecta en `/test` si falta |
| `query_prefix`, `document_prefix` | string | no | solo embedding; `document_prefix` admite `{title}` |
| `supports_tools` | bool (false) | no | chat; lo actualiza `/test` |
| `default_k` (5), `max_k` (15), `max_tool_calls` (3) | int >= 1 | no | chat |

Respuesta real (`Provider`):
```json
{"id":"cb08c78f-811f-4853-b8c6-a61b49af2253","kind":"chat","name":"Mock","base_url":"http://x/v1","model":"mock","api_key_masked":"sk-****3456","dimensions":null,"query_prefix":null,"document_prefix":null,"supports_tools":false,"default_k":5,"max_k":15,"max_tool_calls":3}
```
`PATCH /providers/{id}`: todos los campos opcionales salvo `kind` (no se acepta). Campo ausente = sin cambios. `api_key: null` o `""` **borra** la key. En un perfil de **embeddings** con documentos indexados, cambiar `model`, `dimensions`, `query_prefix` o `document_prefix` → `409 provider_in_use` (invalidaría sus vectores; crear un perfil nuevo). Renombrar, `base_url`, `api_key`, etc. siguen permitidos. `DELETE`: `204`, o `409 provider_in_use` si hay documentos indexados con ese perfil.

`PUT /settings/active` body: `{"chat_profile_id"?: uuid|null, "embedding_profile_id"?: uuid|null}`. Si el perfil no es del tipo indicado → `422 kind_mismatch`; si no existe → `404 provider_not_found`.

`POST /providers/{id}/test` en un perfil de embeddings devuelve `{"ok":true,"detail":"embedding ok","dimensions":768}` (los campos que no aplican se omiten).

### Chats
```json
// POST /chats -> 201, GET /chats -> Chat[]
{"id":"1906e88a-08b0-493a-a5da-0f8476bfbb87","title":"","created_at":"2026-10-14T15:08:47.890319Z"}
// GET /chats/{id} -> ChatDetail
{"id":"...","title":"t.txt","created_at":"2026-10-14T03:23:29.794914Z","document":{"id":"9f77ed4f-2b47-4061-a3d0-83ccabfe1f3a","filename":"t.txt","status":"ready","error":null,"page_count":null,"created_at":"2026-10-14T03:23:40.066900Z"}}
```

### Mensajes (historial)
`GET /chats/{id}/messages` → ordenados por `created_at` ascendente (usuario antes que su respuesta):
```json
[
 {"id":"0c1775ab-...","role":"user","content":"¿Qué come el gato?","created_at":"2026-10-14T15:08:47.480869Z","citations":null,"searches":null},
 {"id":"a1e44d2c-...","role":"assistant","content":"El gato come pescado [1].","created_at":"2026-10-14T15:08:47.482027Z",
  "citations":[{"n":1,"document_id":"9f77ed4f-...","filename":"t.txt","page":null,"snippet":"Hola mundo. El gato come pescado.","chunk_id":"5d0c2b8e-...","page_end":null,"content":"Hola mundo. El gato come pescado.","section":null}],
  "searches":[{"query":"¿Qué come el gato?","k":5}]}
]
```
Los mensajes `user` tienen `citations` y `searches` en `null`. En los `assistant`, `citations` es **la lista de fuentes recuperadas que entraron al contexto** del modelo (no solo las citadas en el texto; si lo recuperado excede el presupuesto de contexto, las últimas se descartan y la numeración `[1..n]` se mantiene consecutiva) y `searches` son las búsquedas que hizo el backend (informativo; útil para debug, no hace falta mostrarlo).

`chunk_id`, `page_end`, `content` y `section` son **nullable**: los mensajes guardados antes de estos campos los devuelven en `null`. En ese caso, caé al comportamiento básico (abrir en `page`, sin resaltado).

### Archivo original

`GET /documents/{id}/file` devuelve el archivo tal como se subió, con el `Content-Type` guardado del documento (`application/pdf`, `text/plain`, `text/markdown`) y `Content-Disposition: inline; filename=...` (los nombres no ASCII van como `filename*=utf-8''...`). Soporta `Range`.

- Cargalo por URL en pdf.js / react-pdf (`file={`${API}/documents/${id}/file`}`); no hace falta bajarlo como blob.
- En TXT/MD la respuesta es texto plano (`text/plain` / `text/markdown`): mostralo con un visor de texto, sin paginado.
- 404 `document_not_found` si el documento no existe; 404 `file_not_found` si el archivo ya no está en disco.

## 5. Ingesta de documentos

`POST /chats/{id}/documents` con `multipart/form-data`, campo **`file`**:

```ts
const fd = new FormData();
fd.append("file", file); // File del <input type="file">; NO setees content-type a mano
const r = await fetch(`${API}/chats/${chatId}/documents`, { method: "POST", body: fd });
// 202 -> { document_id: string, status: "pending" }
```

- Tipos soportados (por extensión **y** content-type del archivo): `.pdf` (`application/pdf`), `.txt` (`text/plain`), `.md` (`text/markdown`, `text/x-markdown` o `text/plain`) imágenes `.jpg`/`.jpeg` (`image/jpeg`), `.png`, `.webp`, `.gif` (migración 012: `kind: "image"`) y audio `.mp3` (`audio/mpeg`), `.wav` (`audio/wav`, `audio/x-wav`), `.m4a` (`audio/mp4`, `audio/x-m4a`), `.ogg`/`.opus` (`audio/ogg`, `audio/opus`), `.webm` (`audio/webm`), `.flac` (`audio/flac`) (`kind: "audio"`; el MIME canónico guardado se sirve en `/file`). Otro → `415 unsupported_file_type`. Ojo: algunos navegadores mandan `.md` con content-type vacío u otro; si falla, forzá `new File([file], file.name, {type: "text/markdown"})`.
- **Un documento por chat.** Segundo upload con documento existente → `409 chat_has_document`. Excepción: si el documento está en `failed`, se **reemplaza** automáticamente al re-subir.
- Requiere embedding activo (`409 no_embedding_profile` si no).
- La indexación corre en segundo plano. Hacé **polling** a `GET /documents/{document_id}` cada 1–2 s y cortá en `ready` o `failed` (con tope razonable, ej. 5 min).

| `status` | Significado |
|---|---|
| `pending` | Recibido, en cola |
| `processing` | Extrayendo texto y generando embeddings |
| `ready` | Listo para preguntar. `page_count` se llena en PDFs (null en TXT/MD) |
| `failed` | Falló; `error` tiene el motivo (texto en español, hasta 500 chars) |

Ejemplos de `error` reales: `"requiere OCR — fase 2"` (PDF escaneado / sin texto extraíble; no se soporta OCR todavía), `"el archivo no tiene texto"` (TXT/MD vacío), el mensaje del proveedor de embeddings si estaba caído (ej. `ConnectError: ...`), o `"Ingesta interrumpida por un reinicio; volvé a subir el archivo."` (el backend se reinició mientras el documento estaba `pending`/`processing`; re-subir lo reemplaza). Sugerencia UI: mostrar `error` tal cual y ofrecer "Subir otro archivo".

### Re-indexar un documento

`POST /documents/{id}/reindex` vuelve a trocear y embeber el archivo ya guardado (mismo perfil de embeddings, sin re-subir). Sirve para documentos indexados con una versión anterior del pipeline (ej. los subidos antes de la búsqueda híbrida: fragmentos de ~3200 caracteres). Responde `202 {"document_id": "...", "status": "processing"}` y corre en segundo plano: hacé el mismo polling que en la subida hasta `ready` o `failed`. Mientras está `processing`, preguntar devuelve `409 document_not_ready`. Los mensajes viejos conservan sus citas (con el texto de los fragmentos anteriores). Errores: `404 document_not_found`, `404 file_not_found`, `409 document_busy` (ya está `pending`/`processing`). Sugerencia UI: botón "Re-indexar" en el detalle del documento cuando está `ready` o `failed`.

## 6. Chat por SSE

`POST /chats/{id}/messages` con body `{"content":"..."}` responde `text/event-stream`. Como es **POST**, `EventSource` **no sirve** (solo GET): usá `fetch` + `ReadableStream`.

Enviá el texto **crudo** del usuario. El backend se encarga de reescribir la consulta de búsqueda (vía tool interna) y de usar el historial reciente (últimos 6 mensajes).

### Eventos (en este orden)

| Evento | Payload | Cuándo |
|---|---|---|
| `search` | `{"query": string, "k": number, "hits": number}` | 1..`max_tool_calls` veces, **antes** de `citations`; una por cada búsqueda que el backend ejecuta realmente (también la del fallback sin tools: `query` = pregunta, `k` = `default_k`). Siempre hay al menos una |
| `citations` | `Citation[]` | Una vez, después de los `search` y antes de los tokens |
| `token` | `{"text": string}` | 0..N veces; concatenar `text` |
| `done` | `{"message_id": string, "cited": number[]}` | Fin OK. `cited` = números `[n]` realmente usados en la respuesta |
| `error` | `{"code": string, "message": string}` | Falla durante el stream; es el último evento |

Stream real:
```
event: search
data: {"query": "qué come el gato", "k": 5, "hits": 3}

event: citations
data: [{"n": 1, "document_id": "9f77ed4f-...", "filename": "t.txt", "page": null, "snippet": "Hola mundo. El gato come pescado.", "chunk_id": "5d0c2b8e-...", "page_end": null, "content": "Hola mundo. El gato come pescado.", "section": null}]

event: token
data: {"text": "El gato "}

event: token
data: {"text": "come pescado "}

event: token
data: {"text": "[1]."}

event: done
data: {"message_id": "a1e44d2c-2293-419e-b269-d7912320c92e", "cited": [1]}

```
Notas:
- Si el proveedor no soporta streaming, el backend hace fallback y la respuesta llega en **un solo evento `token`**. Tu UI no debe asumir muchos tokens.
- El texto puede traer `[1]`, `[1, 2]`. Los números fuera de rango no entran en `cited`.
- Si hay `error` en el stream, **no se guarda** nada en el historial (ni la pregunta). Ya vio `citations` quizá: descartá el mensaje parcial o marcalo como fallido con opción de reintentar.
- Si el usuario cancela, usá `AbortController`; el backend corta.

### Errores antes del stream
Las precondiciones se validan antes de abrir el stream y llegan como **JSON normal con status HTTP** (`409 no_chat_profile`, `409 no_document`, `409 document_not_ready`, `404 chat_not_found`, `422 invalid_input`...). Revisá `res.ok` y `content-type` antes de parsear SSE.

### Cliente completo (TypeScript)

```ts
export type SseHandlers = {
  onCitations?: (c: Citation[]) => void;
  onToken?: (text: string) => void;
  onDone?: (d: DoneEvent) => void;
  onError?: (e: ApiError) => void; // error en stream o previo al stream
};

export async function askStream(
  chatId: string,
  content: string,
  h: SseHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ content }),
    signal,
  });

  const type = res.headers.get("content-type") ?? "";
  if (!res.ok || !type.includes("text/event-stream")) {
    // error previo al stream: JSON {code, message}
    const err: ApiError = await res
      .json()
      .catch(() => ({ code: "http_error", message: `HTTP ${res.status}` }));
    h.onError?.(err);
    return;
  }

  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  const dispatch = (block: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const raw of block.split("\n")) {
      const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
      if (line.startsWith(":")) continue; // comentario/keep-alive
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    if (dataLines.length === 0) return;
    const data = JSON.parse(dataLines.join("\n"));
    switch (event) {
      case "search": h.onSearch?.(data as SearchEvent); break;
      case "citations": h.onCitations?.(data as Citation[]); break;
      case "token": h.onToken?.((data as TokenEvent).text); break;
      case "done": h.onDone?.(data as DoneEvent); break;
      case "error": h.onError?.(data as ApiError); break;
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // los eventos se separan con una línea en blanco; un chunk puede cortar a mitad de evento
    let idx: number;
    while ((idx = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const m = buffer.slice(idx).match(/^\r?\n\r?\n/)!;
      dispatch(buffer.slice(0, idx));
      buffer = buffer.slice(idx + m[0].length);
    }
  }
  if (buffer.trim()) dispatch(buffer); // resto sin separador final
}
```

### Guía de renderizado
1. Al enviar: agregá el mensaje del usuario y una burbuja `assistant` vacía en estado "streaming".
2. `search` (1..N, llegan en vivo apenas el backend termina cada búsqueda) → disparan la animación de "escaneo" del documento: una pasada por búsqueda, mostrando `query` como etiqueta, `k` como cantidad objetivo y `hits` como resultados encontrados. Ya tenés `hits` al llegar el evento; no esperes a `citations`. `hits` puede ser menor que `k`.
3. `citations` → guardalas en la burbuja y mostralas primero (chips con `[n]`, `filename`, `page`, tooltip con `snippet`). `page` es `null` en TXT/MD.
4. `token` → `text += token` (render incremental).
5. `done` → marcá como completa, guardá `message_id` y resaltá solo las citas en `cited`. En el texto, reemplazá `/\[(\d+(?:\s*,\s*\d+)*)\]/g` por chips que abren la cita `n` correspondiente (página + snippet). Las citas no incluidas en `cited` mostralas atenuadas o plegadas ("Fuentes consultadas").
6. Click en una cita (click-to-trace): abrí `GET /documents/{document_id}/file` en el visor central, saltá a `page` (y hasta `page_end` si el chunk cruza páginas) y resaltá el chunk buscando `content` dentro de la capa de texto del visor (normalizá espacios en blanco antes de comparar; el chunk puede empezar o terminar a mitad de oración). Los rectángulos exactos podrían llegar más adelante; por ahora el resaltado es por coincidencia de texto. Si `page` es `null` (TXT/MD), buscá `content` en el texto completo. Desde la búsqueda híbrida `content` es un fragmento de ~1000 caracteres que no cruza títulos de sección (casi siempre dentro de una página). Es texto **literal** de la página (el espaciado tipo `S E C C I Ó N 0 9` se conserva), salvo que se le quitan los encabezados/pies repetidos y el número de página: si el chunk cruza de página, ese encabezado no aparece en `content`, así que resaltá por página (o por líneas) en vez de exigir una sola coincidencia continua.
7. `error` → mostrar `message` y botón "Reintentar" (según `code`, ver §7).

## 7. Errores

Forma única, también en el evento SSE `error`:
```json
{"code": "chat_not_found", "message": "Chat no encontrado"}
```
`message` está en español pensado para usuarios finales, salvo los de proveedor (técnicos). Programá la UI contra `code`, no contra `message`.

| code | HTTP | Significado | Acción de UI sugerida |
|---|---|---|---|
| `invalid_input` | 422 | Body/params inválidos (el `message` lista `campo: motivo` separados por `;`), o UUID mal formado | Marcar campos / bug del cliente |
| `invalid_k` | 422 | `default_k > max_k` (en PATCH) | Error en el formulario de proveedor |
| `kind_mismatch` | 422 | `PUT /settings/active` con perfil de otro tipo | Filtrar selects por `kind` |
| `unsupported_file_type` | 415 | Extensión/content-type no soportado | Mensaje "solo pdf, txt, md" |
| `chat_not_found` | 404 | El chat no existe (o ya se borró) | Volver a la lista de chats |
| `document_not_found` | 404 | Documento inexistente | Refrescar el chat |
| `file_not_found` | 404 | El documento existe pero su archivo no está disponible (`GET /documents/{id}/file`) | Avisar que el original no está disponible |
| `provider_not_found` | 404 | Perfil inexistente | Refrescar la lista de proveedores |
| `chat_has_document` | 409 | El chat ya tiene documento | Ocultar el upload; ofrecer "chat nuevo" |
| `no_embedding_profile` | 409 | No hay embeddings activo (upload) | Ir a Configuración |
| `no_chat_profile` | 409 | No hay proveedor de chat activo (preguntar) | Ir a Configuración (§3) |
| `no_document` | 409 | El chat no tiene documento | Pedir que suba uno |
| `document_busy` | 409 | Re-index pedido con el documento `pending`/`processing` | Seguir el polling |
| `document_not_ready` | 409 | El documento está `pending/processing/failed` | Seguir el polling o ofrecer re-subir si `failed` |
| `embedding_profile_missing` | 409 | El perfil de embeddings del documento ya no existe | Re-subir en un chat nuevo |
| `provider_in_use` | 409 | Borrar un perfil, o cambiar `model`/`dimensions`/prefijos de un perfil de embeddings, con documentos indexados | Informar que no se puede; ofrecer crear un perfil nuevo |
| `provider_unreachable` | 502 | Falló la llamada al proveedor (red, HTTP 4xx/5xx, respuesta inválida). `message` trae detalle técnico. Código por defecto de `ProviderError` | "No se pudo contactar al modelo"; revisar key/URL en Configuración; reintentar |
| `database_unavailable` | 503 | Base de datos caída (solo `/health`) | Banner "servicio no disponible" |
| `http_error` | 404/405/… | Error HTTP genérico (ruta o método inexistente) | Bug del cliente |
| `internal_error` | 500 | Error inesperado (`message` = texto de la excepción) | Mensaje genérico + reintentar |

Dentro de un stream SSE, los `code` posibles son los de proveedor (`provider_unreachable`) o `internal_error`; el HTTP ya fue 200 y no hay status que mirar.

## 8. Tipos TypeScript

```ts
export type UUID = string;
export type ISODate = string; // "2026-10-14T15:08:47.480869Z"

export type ProviderKind = "chat" | "embedding";

export interface Provider {
  id: UUID;
  kind: ProviderKind;
  name: string;
  base_url: string;
  model: string;
  api_key_masked: string | null;
  dimensions: number | null;      // embedding
  query_prefix: string | null;    // embedding
  document_prefix: string | null; // embedding
  supports_tools: boolean;        // chat
  default_k: number;              // chat
  max_k: number;                  // chat
  max_tool_calls: number;         // chat
}

export interface ProviderIn {
  kind: ProviderKind;
  name: string;
  base_url: string;
  model: string;
  api_key?: string | null;
  dimensions?: number | null;
  query_prefix?: string | null;
  document_prefix?: string | null;
  supports_tools?: boolean;
  default_k?: number;
  max_k?: number;
  max_tool_calls?: number;
}

/** PATCH: todo opcional; `kind` no se puede cambiar. api_key null/"" borra la key. */
export type ProviderPatch = Partial<Omit<ProviderIn, "kind">>;

export interface ActiveSettings {
  chat_profile_id: UUID | null;
  embedding_profile_id: UUID | null;
}

/** Campos que no aplican al tipo probado vienen omitidos (no null). */
export interface ProbeResult {
  ok: boolean;
  detail: string;
  dimensions?: number;      // embedding
  modalities?: string[];    // embedding: "text" + "image"/"audio" si el servidor las acepta
  supports_tools?: boolean; // chat
}

export interface Health {
  status: "ok";
  providers: {
    chat: { name: string; reachable: boolean } | null;
    embedding: { name: string; reachable: boolean } | null;
  };
}

export interface Chat {
  id: UUID;
  title: string; // "" hasta el primer upload; luego el filename
  created_at: ISODate;
}

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface Document {
  id: UUID;
  filename: string;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null; // solo PDFs ready
  created_at: ISODate;
  // Migración 011 (fuentes multimedia); los documentos existentes son "pdf"/"text".
  kind?: "pdf" | "text" | "image" | "audio" | "video";
  duration_s?: number | null; // audio/video
  width?: number | null; // imagen/video
  height?: number | null;
  stage?: "extracting" | "embedding" | "describing" | "transcribing" | null; // qué mide `progress`
  enrich_status?: "pending" | "running" | "ready" | "partial" | "failed" | "skipped";
}

export interface ChatDetail extends Chat {
  document: Document | null;
}

export interface UploadResult {
  document_id: UUID;
  status: "pending";
}

export interface Citation {
  n: number;
  document_id: UUID;
  filename: string;
  page: number | null; // null en TXT/MD
  snippet: string;     // ~200 chars
  chunk_id: UUID | null;      // null en mensajes viejos
  page_end: number | null;    // última página del chunk; null en TXT/MD o mensajes viejos
  content: string | null;     // texto completo del chunk (para resaltar); null en mensajes viejos
  section: string | null;     // título de la sección del chunk (ej. "9. ÁREA GEOGRÁFICA DE MERCADO"); null fuera de sección o en mensajes viejos
  // aún no implementado; si llega, el front lo usa en vez de buscar `content` en la capa de texto
  // (puntos PDF, origen arriba-izquierda, como PyMuPDF)
  rects?: { page: number; x0: number; y0: number; x1: number; y1: number }[] | null;
  // Contrato multimedia (migración 011). Ausentes en citas viejas: `kind` entonces es "text".
  kind?: "text" | "image" | "audio" | "video"; // qué abre el visor
  bbox?: [number, number, number, number] | null; // [x0,y0,x1,y1] normalizado 0-1 sobre la página/imagen
  t_start?: number | null; // segundos (audio/video)
  t_end?: number | null;
  has_thumb?: boolean; // existe GET /segments/{chunk_id}/thumb
  source_kind?: "pdf" | "text" | "image" | "audio" | "video" | null; // tipo de la fuente
}

export interface Search {
  query: string;
  k: number;
}

export interface Message {
  id: UUID;
  role: "user" | "assistant";
  content: string;
  created_at: ISODate;
  citations: Citation[] | null; // null en user
  searches: Search[] | null;    // null en user
}

export interface ApiError {
  code: string;
  message: string;
}

// SSE
export type TokenEvent = { text: string };
export type SearchEvent = { query: string; k: number; hits: number };
export type DoneEvent = { message_id: UUID; cited: number[] };
export type SseEvent =
  | { event: "search"; data: SearchEvent }
  | { event: "citations"; data: Citation[] }
  | { event: "token"; data: TokenEvent }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: ApiError };
```

Helper sugerido para llamadas JSON normales:
```ts
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (res.status === 204) return undefined as T;
  const body = await res.json();
  if (!res.ok) throw body as ApiError;
  return body as T;
}
```

## 9. Gotchas

- Fechas ISO 8601 UTC con `Z` y microsegundos; `new Date(s)` las parsea.
- IDs UUID string. `DELETE` devuelve `204` sin body (no hagas `res.json()`).
- Historial: `GET /messages` viene cronológico; guardá el orden tal cual. No hay paginación.
- Mandá la pregunta cruda; no armes prompts ni reescribas consultas en el cliente. El campo `searches` del mensaje del asistente muestra qué consultó el backend (puede diferir de la pregunta si el modelo usa tools).
- `citations` del mensaje incluye las fuentes que entraron al contexto del modelo (puede ser menos que las recuperadas); `cited` (solo en el evento `done`, no se persiste) dice cuáles aparecen en el texto. Para mensajes del historial, derivá lo citado parseando `[n]` del `content`.
- Borrar un chat borra su documento y mensajes.
- Respuestas lentas: el timeout hacia el proveedor es 120 s; mostrá estado "pensando" entre el último `search`/`citations` y el primer `token`.
- Si el chat no tiene título aún, mostrá "Nuevo chat".
- Sin auth: la API solo escucha en `127.0.0.1`; no la expongas tal cual.
- Inconsistencias conocidas: (1) `default_k > max_k` devuelve `invalid_k` en `PATCH` pero `invalid_input` en `POST /providers`; (2) los `error` de documentos fallidos y los `message` de dominio están en español, los de proveedor en inglés/técnicos; (3) el `message` de `kind_mismatch` incluye el UUID.

### Progreso de ingesta

`DocumentOut` incluye `progress: {done, total} | null` (fragmentos embebidos / total, solo mientras `processing`). PDFs sin capa de texto se procesan con OCR (Tesseract spa+eng) si está instalado; si no, fallan con "El PDF no tiene texto (escaneado); OCR no disponible".

## Verificar respuesta, sin spoilers y persona (migración 008)

- `POST /chats/{chat_id}/messages/{message_id}/verify` -> `{sentences: [str], claims: [{text, sentence_index, status: "respaldada"|"parcial"|"sin_respaldo", source_ns: [n], quote?, reason}]}`. Idempotente (se guarda en `messages.verification`; también viene en `GET /messages` como `verification`). Un LLM-call; "respaldada" con casi cero palabras en común con la fuente baja a "parcial". 404 `message_not_found`, 409 `no_chat_profile`, 502 `invalid_provider_response`.
- `POST /chats/{chat_id}/messages` acepta `max_page` (int >= 1, opcional): recuperación (vector, FTS, panorama, índice) limitada a chunks con `page_start <= max_page` (los sin página no se filtran) y prompt "solo lo leído".
- `PATCH /chats/{id}` acepta `persona` (string <= 120, null o vacío la quita); `ChatOut.persona`. Cada respuesta guarda la persona con la que se escribió (`MessageOut.persona`).

## Imágenes y perfil de visión (migración 012)

- **Perfil `vision`** (como `stt`/`tts`): `kind: "vision"` en `POST /providers` (solo `name`, `base_url`, `model`, `api_key`; OpenAI-compatible `/chat/completions` con `image_url`). Activo con `PUT /settings/active {"vision_profile_id": uuid|null}` (también en `GET`). `POST /providers/{id}/test` manda una imagen generada y devuelve `{"ok":true,"detail":"vision ok","caption":"..."}`; una respuesta vacía del modelo es `ok:false`.
- **Ingesta en dos etapas.** Etapa 1 (local): PDFs extraen imágenes embebidas (con bbox), gráficos vectoriales (`figure`) y páginas escaneadas (`page` + 4 `tile` si casi no hay texto); las imágenes sueltas son `whole_image` (+ tiles si lado >= 1200 px). Se guardan recortes (`/segments/{id}/asset`, <= 1024 px) y miniaturas (`/thumb`, 256 px) y se embeben de forma nativa; el documento pasa a `ready` y ya es buscable. Requiere que el perfil de embeddings admita imágenes (`modalities` incluye `image`; se detecta con `POST /providers/{id}/test`); sin eso un PDF se indexa solo como texto y una imagen falla con "El perfil de embeddings activo no admite imágenes".
- **Etapa 2 (descripciones)**: con perfil de visión activo, cada segmento recibe una descripción (tipo, descripción, objetos, texto visible, datos de gráficos) que se guarda como `content` del segmento (FTS + embedding de texto propio). `enrich_status`: `pending` -> `running` (`stage: "describing"`, `progress` = segmentos descriptos/total) -> `ready` | `partial` | `failed`; `skipped` sin perfil de visión. `enrich_error` explica `partial`/`failed`. Nunca cambia `status` (el documento sigue `ready`).
- `POST /documents/{id}/enrich` -> `202 {document_id, enrich_status: "pending"}` reintenta/retoma las descripciones pendientes o fallidas (errores: `404 document_not_found`, `409 document_busy`, `409 no_vision_profile`). `POST /documents/{id}/cancel` también cancela la etapa 2 (queda `partial`).
- `GET /documents/{id}/chunks` suma por segmento: `modality`, `origin`, `bbox`, `text_status`, `has_thumb`.

### Audio (fase 5)

- **Documento**: `kind: "audio"`, `duration_s` (seg), `mime` canónico. Tope 3 h: más largo => `status: "failed"` con `error` claro ("El audio dura más de 3 h..."); archivo mudo => `error: "El audio no tiene sonido"`. El original se sirve en `GET /documents/{id}/file` con HTTP Range (seek nativo).
- **Segmentos**: ventanas de ~20 s con 4 s de solape, bordes corridos al silencio más cercano (±3 s); las ventanas totalmente silenciosas se omiten. `modality: "audio"`, `origin: "window"`, `t_start`/`t_end` (seg, float), `content` = transcripción (vacío hasta la etapa 2). `GET /documents/{id}/chunks` los trae con `t_start`/`t_end`.
- **Etapa 1** (local, sin STT): ventanas + embedding nativo de audio por ventana + forma de onda; estado `processing` con `stage: "extracting"` y luego `"embedding"` (`progress` = ventanas). Una ventana cuyo embedding falla queda como segmento sólo-texto; si el perfil de embeddings no admite audio y no hay perfil `stt` activo, la ingesta falla con mensaje claro.
- **Etapa 2** (transcripción por ventana con el perfil `stt` activo, protocolo ollama u openai): `enrich_status` `pending` -> `running` (`stage: "transcribing"`, `progress` = ventanas transcriptas/total) -> `ready` | `partial` | `failed`; `skipped` sin perfil `stt` (`enrich_error` = "No hay un proveedor de transcripción activo"). Reanudable por ventana (`text_status`); una ventana sin voz queda `skipped`. `POST /documents/{id}/enrich` reintenta (`409 no_stt_profile` si no hay STT). Resumen y preguntas sugeridas se generan después de las transcripciones.
- `GET /documents/{id}/waveform` -> `{peaks: number[] (<= 800, normalizados 0-1), duration_s}`; `404 waveform_not_found` si el documento no es audio.
- **Citas** de audio: `kind: "audio"`, `t_start`, `t_end`, `content` = transcripción; el modelo las ve como `(audio · 01:20–01:40) «transcripción»`. `find_all` sobre audio usa transcripciones (sin visión).
- La búsqueda de texto incluye las descripciones de las imágenes (vector de texto + FTS); la búsqueda visual nativa y `find_all` están descriptas abajo (fase 3).

## Recuperación y respuestas multimodales (migración 013, fase 3)

- **Rankings por modalidad.** Cada búsqueda del modelo corre 4 rankings, cada uno top-n *dentro de su modalidad*: vector de texto (texto + descripciones/transcripciones), FTS, vector nativo de imagen/frame y vector nativo de audio; se fusionan por RRF con pesos (1.0 / 0.3 / 0.7 / 0.7). Nunca se comparan cosenos entre modalidades. El tool `search_document` gana el argumento `modality` (`auto|texto|imagen|audio|video`; sólo se ofrece si el documento tiene multimedia): sube el peso del ranking nativo a 1.5 y filtra a esa modalidad.
- **Tool `find_all(query, modality?)`** (sólo en documentos con multimedia): revisa TODAS las imágenes/frames, verifica cada candidata con el perfil `vision` activo (sí/no, cacheado por segmento + consulta normalizada en `segment_verdicts`) y devuelve todas las ocurrencias con página+bbox o tiempo (tope 40). Sin perfil de visión devuelve las candidatas con etiqueta de confianza. Cuenta para `max_tool_calls`. El planner ve además un catálogo de fuentes (`S1 · informe.pdf · PDF 4 págs · 3 imágenes (3 descritas)`).
- **Evento SSE `search`** suma: `modality` (pista usada), `exhaustive` (true en `find_all`), `rankings` (en búsquedas normales: candidatos por ranking `{vector, text, image, audio}`; en `find_all`: `{checked, discarded, asked}`). Cada `found[]` suma `kind`, `bbox`, `t_start`, `t_end`, `has_thumb` (mismo significado que en `Citation`). Todo se persiste en `searches` de los mensajes; mensajes viejos traen los defaults (`modality: "auto"`, `exhaustive: false`, `rankings: {}`, `kind: "text"`).
- **Citas** (`citations` en SSE y en `GET /messages`) ya llevan `kind`, `bbox`, `t_start`, `t_end`, `has_thumb`, `source_kind`. `score` = relevancia calibrada por modalidad: `clamp((cos - floor) / (1 - floor))`; en multimedia, el máximo entre su coseno nativo y el de su descripción. Los scores de texto ahora también se normalizan con el `floor` de texto (0.50 por defecto), así que bajan respecto de antes (cos 0.8 → 0.6).
- **`weak_evidence`** se decide por modalidad: ninguna fuente supera el umbral de *su* modalidad (texto 0.60, imagen 0.66, audio 0.68 por defecto).
- **Calibración en el perfil de embeddings**: `calibration` en `GET/PATCH /providers` = `{"text"|"image"|"audio": {"floor": 0-1, "threshold": 0-1}}` (`null` = defaults del código; `floor <= threshold`, `422 invalid_calibration` si no). Sólo perfiles `embedding`. La calcula `backend/scripts/calibrate_modalities.py` desde pares etiquetados (ver su docstring): `floor` = mediana de los no relacionados, `threshold` = corte de máxima F1.
- **Respuesta**: las fuentes multimedia llegan al LLM como `[n] (imagen · pág. 3, arriba) descripción` / `[n] (audio · 01:20–01:40) «transcripción»`. Si el perfil de chat activo tiene `supports_vision`, se adjuntan hasta 3 imágenes (assets <= 1024 px) de las fuentes de imagen mejor rankeadas, sólo en el mensaje de usuario.

## Video (fase 6, backend)

- **Subida**: mp4, m4v, mov, mkv y webm (`video/webm`; un `.webm` con `audio/webm` sigue siendo audio). Tope 2 h (`error: "El video dura más de 2 h..."`). Documento: `kind: "video"`, `duration_s`, `width`, `height`. El perfil de embeddings debe admitir imágenes (si no, falla con mensaje claro).
- **Reproducción**: `GET /documents/{id}/file` (Range) sirve la variante reproducible: el original si es mp4 h264/aac (yuv420p) o webm vp8/vp9/av1; en otro caso (hevc, mpeg4, mov, mkv...) se transcodifica UNA vez en la ingesta a `play.mp4` (h264/aac, <=720p, faststart) y `/file` sirve ese mp4 (`video/mp4`). `GET /file?variant=original` devuelve lo subido.
- **Etapa 1 (local)**: keyframes por cambio de escena (`scene>0.3`), mín. 2 s entre frames, al menos uno cada 10 s, tope 900; frames casi idénticos (coseno nativo > 0.97 contra el último conservado) se descartan y el frame conservado extiende su `t_end`. Segmentos `modality: "frame"`, `origin: "frame"`, `t_start` = instante, `t_end` = siguiente frame conservado; `/segments/{id}/thumb|asset`. Luego la pista de audio por el pipeline de audio (ventanas `modality: "audio"` en el mismo documento, ordinales después de los frames; pista muda o ausente => sólo frames). `stage`: `extracting` (transcodificación/escenas) y `embedding` (progress = frames y luego ventanas).
- **Etapa 2 (remota)**: `enrich_status` `running` con `stage: "describing"` (frames, perfil `vision`; "Describiendo fotogramas 3/40") y luego `"transcribing"` (ventanas, perfil `stt`; "Transcribiendo audio"). Si falta un proveedor, sus segmentos quedan `pending` y termina `partial` con `enrich_error` = "No hay un proveedor de ...". `POST /documents/{id}/enrich` acepta video con visión o STT.
- **`GET /documents/{id}/storyboard`** -> `[{segment_id, t, t_end, thumb_url}]` en orden temporal (`404 storyboard_not_found` si no es video). La forma de onda del audio del video: `GET /documents/{id}/waveform`.
- **Citas**: frames `kind: "video"`, `t_start`, `t_end`, `has_thumb: true`, `source_kind: "video"`. Transcripciones de su audio: `kind: "audio"` con `source_kind: "video"` (el front debe hacer seek en el player de video). El LLM ve `(video · 12:05)` para frames y `(video · 01:20–01:40) «..»` para su voz. `find_all` acepta `modality: "video"` (verifica frames con visión y devuelve instantes).

## Multi-fuente (fase 7, migración 014)

Un chat tiene **N fuentes** (los documentos de su colección). Los chats viejos (1 documento) no cambian.

- `GET /chats/{id}` → `sources: SourceOut[]` en orden de alta + `active_source_ids` + `overview` (agregado, solo con 2+ fuentes). **`document` sigue siendo la primera fuente (S1)**: existe por compatibilidad durante la transición; el cliente nuevo debe leer `sources`.
- `SourceOut` = `DocumentOut` + `alias` (`S1..Sn`, por orden de alta; **cambia si se quita una fuente anterior**: persistí el `document_id`, no el alias) + `active` (el chat la lee).
- `POST /chats/{id}/sources` (multipart `file`; alias de `POST /chats/{id}/documents`) agrega una fuente a un chat con o sin fuentes → `202 {document_id}`. Una fuente nueva usa el perfil de embeddings de sus hermanas (los vectores deben coincidir). Máximo 12 (`409 too_many_sources`). Si la ÚNICA fuente falló, la nueva la reemplaza (reintento).
- `DELETE /chats/{id}/sources/{document_id}` → `204`; borra documento, chunks y archivos (`404 source_not_found` si no es del chat). Las citas viejas a esa fuente quedan en el historial (sin pestaña).
- `PATCH /chats/{id}` acepta `active_source_ids: uuid[]` (las demás quedan silenciadas; se guarda `chats.disabled_sources`, la respuesta trae `disabled_sources`). `POST /chats/{id}/messages` acepta `source_ids: uuid[]` para esa pregunta (pisa la selección del chat). Sin fuentes activas listas → `409 no_active_sources`.
- `ChatOut.source_count`: cantidad de fuentes (badge de la barra lateral). `title`: nombre de la fuente si hay una, `"N fuentes"` si hay varias.
- Citas (`CitationOut` y evento SSE `citations`): `source_id` (= `document_id`), `source_alias`, `source_name` (= `filename`). `found[]` de cada búsqueda trae `document_id` y `source_alias`; `SearchOut.source` es `"all"` o los alias leídos (`"S1,S3"`).
- Estudio / tutor: cada `page`-ref (`{page, chunk_id}`) suma `document_id` de la fuente del fragmento.
- El planificador ve un catálogo `S1 · nombre · tipo …` y las herramientas `search_document`/`find_all` ganan `source` (`all` | alias) cuando hay 2+ fuentes. `max_page` (sin spoilers) aplica a cada fuente.

## Proyectos, categorías y biblioteca (migración 016)

Un **proyecto** es una `collections` con `kind='project'`: muchas fuentes y muchos chats. Cada chat del proyecto lee la colección completa (alias `S1..Sn` por orden de alta, igual que un chat suelto) y silencia fuentes con su propio `PATCH /chats/{id}` `active_source_ids`. `Chat` / `ChatOut` ganan `project_id` (uuid | null): la UI lista en la barra lateral solo los chats sueltos (`project_id == null`); los del proyecto viven en su página. Borrar un chat de un proyecto conserva las fuentes. Todo es aditivo: los chats sueltos funcionan igual.

### Proyectos

| Método | Ruta | Qué hace |
|---|---|---|
| POST | `/projects` `{name, description?, color?, icon?}` | 201 `Project` |
| GET | `/projects` | `Project[]` (`source_count`, `ready_count`, `chat_count`) |
| GET | `/projects/{id}` | `Project` + `sources[]` (`Source`: alias, `active:true`), `chats[]` (`Chat`), `overview` (resumen agregado, sin LLM, 2+ fuentes) |
| PATCH | `/projects/{id}` | renombra / descripción / color / icono |
| DELETE | `/projects/{id}` | 204. Borra el proyecto, sus chats (con mensajes) y sus fuentes (archivos incluidos). La UI pide confirmación |
| POST | `/projects/{id}/sources` (multipart `file`) | 202 `{document_id}`; ingesta en background (igual que `/chats/{id}/sources`). 409 `too_many_sources` (máx. 12), `no_embedding_profile`. Una fuente nueva hereda el perfil de embeddings de sus hermanas |
| DELETE | `/projects/{id}/sources/{doc}` | 204; los chats del proyecto dejan de silenciarla |
| POST | `/projects/{id}/chats` | 201 `Chat` (título = nombre del proyecto) |
| POST | `/projects/{id}/adopt` `{chat_id}` | **Mueve** un chat suelto al proyecto: sus documentos pasan a ser fuentes del proyecto (se mueven, no se copian) y el chat lee todo el proyecto; sus fuentes silenciadas siguen silenciadas. 409 `chat_in_project`, `embedding_profile_mismatch`, `too_many_sources`. Devuelve el `ProjectDetail` |

Subir con `POST /chats/{chat_id}/sources` a un chat de proyecto también agrega la fuente al proyecto.

### Categorías

`Category {id, name, origin: 'user'|'auto', color, doc_count, created_at}`. `GET/POST /categories`, `PATCH/DELETE /categories/{id}` (409 `category_exists`: el nombre es único sin mayúsculas). `POST /categories/{id}/assign` y `/unassign` con `{document_ids: [...]}` (varios a la vez; devuelven la categoría). Tocar una categoría automática (renombrar, recolorear o asignarle documentos) la convierte en `origin='user'`.

`POST /library/categorize` (síncrono, unos segundos): agrupa los documentos `ready` por su vector (`documents.doc_embedding` = promedio de los vectores de texto de sus fragmentos) con k-means y `k` elegido por silhouette (2..8), y nombra TODOS los grupos con UNA llamada al modelo de chat activo (si falla o no hay perfil de chat: «Grupo N»). **Solo escribe/reemplaza filas `origin='auto'`** (categorías y asignaciones automáticas de corridas anteriores); jamás toca categorías ni asignaciones `origin='user'` (un nombre repetido recibe sufijo numérico). Cada perfil de embeddings es un espacio vectorial distinto: se agrupa dentro de cada perfil (≥ 3 documentos) y los demás se omiten (`skipped_incompatible`). Respuesta: `{categories:[{id,name,doc_count}], clustered, skipped_incompatible, skipped_without_vector, named_by_model, note}`.

### Biblioteca y búsqueda global por concepto

- `GET /library/documents?kind=&category_id=&project_id=&status=&uncategorized=&q=` → `LibraryDoc[]` (más nuevos primero): `{id, filename, kind, status, mime, created_at, page_count, duration_s, summary, project_id, project_name, chat_id, chat_title, has_vector, categories:[{id,name,origin,color}]}`. `chat_id` = el chat a abrir (el propio, o el primero del proyecto). Calcula perezosamente los vectores de documento que falten.
- `POST /library/search` `{query, limit?≤50, kind?, category_id?, project_id?, document_ids?}` → `{query, results[], searched, skipped_unready, skipped_profiles, notes[]}`. Cada resultado: `{document: LibraryDoc, score (0..1), relevant, doc_score, hits[]}`; `hits[]` (hasta 3 por documento): `{chunk_id, kind: text|image|audio|video, page, page_end, t_start, t_end, bbox, has_thumb, snippet, score}` (miniatura: `GET /segments/{chunk_id}/thumb`). Pipeline: la consulta se embebe con el prefijo de consulta **del perfil de cada documento** (vectores de modelos distintos nunca se comparan; si un perfil no responde sus documentos se omiten y `notes` lo dice), se reusa el `Retriever` (vector + texto + nativo, RRF) sobre todos los documentos y se agrupa por documento; el `score` mezcla el mejor fragmento (calibrado por perfil) con el coseno consulta↔vector del documento.
- **Abrir un resultado**: navegar a `/chats/{chat_id}?doc={document_id}&chunk={chunk_id}&page={n}&t={seg}&te={seg}&k={kind}&s={snippet}`; el chat salta a esa cita (como un click en una cita) y limpia los parámetros.
- `POST /library/ask` `{document_ids (≤12), mode: 'chat'|'project', name?, project_id?}` → 201 `{chat_id, project_id|null, document_ids}`: «Preguntar a estos documentos». Una fuente pertenece a una sola colección (compartirla entre colecciones sigue diferido) y mover rompería el chat original, así que **copia** los documentos (archivo original, mismo perfil de embeddings) a un chat nuevo, a un proyecto nuevo o a `project_id`, crea el chat y los reindexa en background (el chat se puede abrir ya; las fuentes pasan de `pending` a `ready`). 409 `embedding_profile_mismatch` si los documentos son de perfiles distintos, `document_not_ready`.
- `POST /library/backfill` → `{updated: n}`: calcula `doc_embedding` de los documentos `ready` que no lo tengan (o cuyo número de fragmentos cambió: re-indexado, descripciones nuevas). Las listas y la búsqueda lo hacen solas; el endpoint existe para forzarlo.

## Compatibilidad de fuentes

Informativa, nunca bloquea. `GET /chats/{id}/compatibility` (fuentes activas), `GET /projects/{id}/compatibility` y
`POST /library/compatibility {document_ids}` (selección antes de "Preguntar a estos documentos"; alias por orden de selección)
devuelven `{coherent, sources:[{document_id, alias, filename, status: ok|weak|unrelated|duplicate|unknown, best_match:{document_id, alias, similarity}|null, reason|null}]}`.
Coseno entre vectores de documento (`documents.doc_embedding`) del mismo perfil de embeddings; perfil/dimensión distinta o sin indexar = `unknown`
(con `reason`). Puntaje por pares de chunks en espacio texto (top 30 % simétrico); `ok` >= 0.72, `unrelated` < 0.70, `duplicate` = mismo hash de archivo o puntaje >= 0.95 ("parece la misma fuente que S2"). Se arma un grafo de fuentes relacionadas: si hay más de un componente, toda fuente fuera del mayor (empate: todas) se marca y `coherent` es false (domain/compatibility.py).
Para `weak`/`unrelated` una llamada de chat da una frase `reason` en español, cacheada en memoria por conjunto de (documento, nº de chunks).
Reconsultar al agregar, quitar o silenciar fuentes (la clave de la query incluye la firma de las fuentes).

## Búsqueda global

`GET /api/v1/search?q=&limit=` (limit 1-50, default 20). Búsqueda por palabras clave (no lenguaje natural) sobre todos los chats, incluidos los de proyectos. Insensible a mayúsculas y acentos; cada término es una subcadena (`presu` encuentra «presupuesto»); varias palabras = AND; `"frase entre comillas"` = exacta. Orden: calidad del match (palabra entera > inicio de palabra > interior) + recencia.

```
{ messages: [{ message_id, chat_id, chat_title, project_id|null, project_name|null, role, created_at,
               snippet, highlights: [[start, end], ...] }],   // offsets UTF-16 dentro de `snippet`
  chats:    [{ chat_id, title, project_id|null, project_name|null, created_at, highlights }],
  projects: [{ project_id, name, created_at, highlights }],
  sources:  [{ document_id, filename, chat_id|null, project_id|null, highlights }] }
```

`snippet` ronda los 160 caracteres alrededor del primer match, sin marcadores `[n]` ni markdown. `q` vacío devuelve todo vacío. Migración 018: `pg_trgm`, función inmutable `search_fold` (lower + unaccent) e índices GIN trigram.

## Mapa conceptual (`concept_map`)

- `POST /chats/{id}/artifacts {kind: "concept_map", options: {focus?, depth?: 2-4, max_nodes?: 8-60}}`; `data = {root_id, nodes: [{id, label, summary, level, kind: "concepto"|"ejemplo", aliases, refs}], links: [{id, source, target, label, kind: "jerarquia"|"cruce", refs}]}`. `refs` = `{quote, page, t_start, t_end, bbox, kind, chunk_id, document_id}` con la cita verificada en el fragmento; un enlace sin cita verificable se descarta. Hasta 3 llamadas al LLM (extraer por grupo de secciones, unificar sinónimos / elegir raíz); la jerarquía (niveles, `jerarquia` vs `cruce`) se calcula por BFS desde la raíz.
- `POST /chats/{id}/artifacts/{artifact_id}/expand {node_id}` -> `{artifact, added_nodes, added_links}`: 1 llamada, 2-6 subconceptos verificados colgados de ese nodo, mezclados y guardados en el mapa. 404 `artifact_not_found`, 409 `not_expandable`, 422 `node_not_found`, 502 si no hay subconceptos verificables.
- Los artefactos `mind_map` viejos siguen renderizando; el Estudio ya no ofrece crear `mind_map`.
