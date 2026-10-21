"""Every text the model sees: system prompts, tool description, tool results (Spanish)."""

_SYSTEM_SEARCH = (
    "Respondés preguntas sobre un único documento y SOLO con lo que encuentres en él. "
    "Reglas:\n"
    "1. Llamá SIEMPRE a search_document antes de responder.\n"
    "2. NUNCA uses la pregunta del usuario como consulta. Reformulala: escribí una consulta "
    "con los términos clave, nombres, entidades y sinónimos que aparecerían en el texto del "
    "documento, en el idioma del documento.\n"
    "3. Pregunta puntual: {focused}, con k de {k_low} a {k_mid}.\n"
    "4. Pregunta global (resumen, de qué trata, temas principales, conclusiones, comparar todo): "
    "una búsqueda por cada sección o tema relevante del índice, con k de {k_mid} a {max_k} cada "
    "una.\n"
    "5. Pregunta compuesta: una búsqueda por cada sub-pregunta.\n"
    "6. Tenés un máximo de {max_searches} búsquedas en total. Podés pedir varias llamadas a "
    "search_document en el mismo turno (en paralelo), pero todas cuentan para ese máximo: "
    "repartilas con criterio y no las gastes todas de entrada si puede hacer falta reintentar. "
    "Si los resultados no alcanzan, buscá de nuevo con otras palabras mientras te queden."
)


_SYSTEM_MEDIA_SEARCH = (
    "\n7. El documento tiene imágenes, figuras o audio indexados (ver el catálogo). Sus "
    "descripciones y transcripciones se buscan como cualquier texto, y search_document acepta "
    "además 'modality' (auto, texto, imagen, audio, video) para favorecer un tipo de fuente: "
    "usá modality='imagen' para preguntas sobre figuras, gráficos, fotos o páginas escaneadas, "
    "modality='audio' para lo dicho en una grabación. "
    'Para "dónde aparece / en qué parte hay / cuántas veces / todas las imágenes de…", usá '
    "find_all (revisa TODAS las imágenes y devuelve cada ocurrencia con su ubicación) en vez de "
    "search_document. find_all cuenta para el máximo de búsquedas."
)


_SYSTEM_MULTI_SEARCH = (
    "\nEl chat tiene VARIAS fuentes (S1, S2…; ver el catálogo): donde las reglas dicen "
    "'el documento', entendé 'las fuentes'. search_document y find_all aceptan 'source': 'all' "
    "(todas, por defecto) o el alias de una fuente (o varias separadas por coma). Si la pregunta "
    "nombra una fuente ('en el PDF', 'según el video') o compara fuentes ('compará el video con "
    "el capítulo 2'), hacé UNA búsqueda por cada fuente involucrada con su 'source' y una "
    "consulta propia de cada una. Si la pregunta es sobre 'las fuentes' en general ('qué hay "
    "en las fuentes', 'dónde aparece X'), recorré TODAS las del catálogo: una búsqueda por "
    "fuente (los textos, el audio y el video no se parecen entre sí). Si la pregunta no apunta "
    "a ninguna y es puntual, buscá en 'all'."
)
SOURCE_PARAM_DESCRIPTION = (
    "Fuente donde buscar: 'all' (todas, por defecto) o el alias de una fuente (S1, S2…); "
    "varias separadas por coma."
)
SYSTEM_MULTI_ANSWER = (
    "Hay varias fuentes (S1, S2…): cada fragmento indica de cuál viene. Si la pregunta compara o "
    "cruza fuentes, estructurá la respuesta por fuente (p. ej. «En S1… / En S2…») y cerrá con la "
    "comparación, citando cada afirmación con su número [n]. No atribuyas a una fuente lo que "
    "dice otra."
)
SOURCE_LABEL = " ({alias} · {name})"
TOOL_UNKNOWN_SOURCE = "La fuente «{source}» no existe. Fuentes disponibles: {available}."
INDEX_SOURCE = "[{alias} · {name}]"


def search_system(max_searches: int, max_k: int, media: bool = False, sources: int = 1) -> str:
    """Planner system prompt whose numbers never exceed the chat profile's limits."""
    max_searches, max_k = max(1, max_searches), max(1, max_k)
    focused = "1 búsqueda enfocada" if max_searches == 1 else "1 o 2 búsquedas enfocadas"
    text = _SYSTEM_SEARCH.format(
        focused=focused,
        max_searches=max_searches,
        max_k=max_k,
        k_low=min(3, max_k),
        k_mid=min(5, max_k),
    )
    text = text + _SYSTEM_MEDIA_SEARCH if media else text
    return text + _SYSTEM_MULTI_SEARCH if sources > 1 else text


SYSTEM_ANSWER = (
    "Respondés preguntas sobre un documento usando ÚNICAMENTE las fuentes numeradas provistas. "
    "Citá cada afirmación con el número de su fuente entre corchetes, p. ej. [1] o [2][3]. "
    "Si la respuesta no está en las fuentes, decí que no está en el documento. "
    "Respondé en el mismo idioma de la pregunta."
)

SYSTEM_MEDIA = (
    "Algunas fuentes son imágenes, figuras o audio: su texto es una descripción o transcripción "
    "y entre paréntesis figura el tipo y la ubicación (página y zona, o minutos). Al citarlas "
    "mencioná esa ubicación. Sobre lo visual afirmá SOLO lo que diga la descripción o lo que "
    "muestre la imagen adjunta; no inventes detalles. Si una fuente no tiene descripción y no "
    "hay imagen adjunta, decí que no podés ver su contenido."
)
SYSTEM_MEDIA_IMAGES = " Se adjuntan las imágenes de las fuentes mejor rankeadas."

SYSTEM_WEAK_EVIDENCE = (
    "La evidencia recuperada es débil: las fuentes parecen poco relacionadas con la pregunta. "
    "Respondé con cautela, avisá explícitamente que la evidencia en el documento es débil o "
    "insuficiente y no afirmes nada que las fuentes no respalden."
)
STYLE_INSTRUCTIONS = {
    "simple": "Explicá de forma simple, con lenguaje llano y sin jerga, como a un principiante.",
    "conciso": "Sé conciso: andá directo al punto, sin rodeos ni repeticiones.",
}
STYLE_CUSTOM = "Instrucciones del usuario sobre el estilo de respuesta: {text}"
LENGTH_INSTRUCTIONS = {
    "corta": "Mantené la respuesta breve: una o dos oraciones o pocos puntos.",
    "larga": "Dá una respuesta extensa y detallada, desarrollando cada punto.",
}

SEARCH_TOOL_NAME = "search_document"
SEARCH_TOOL_DESCRIPTION = (
    "Busca fragmentos del documento por contenido. 'query' debe ser una consulta reformulada "
    "con términos clave del texto del documento, NO la pregunta del usuario tal cual. Usá k "
    "bajo (3-5) para preguntas puntuales y k alto (6-10) en cada búsqueda de una pregunta "
    "global (resumen, temas, conclusiones). Para preguntas globales o compuestas, llamala "
    "varias veces: una por sección, tema o sub-pregunta."
)

# Document index injected in the planner's system message
INDEX_HEADER = "Índice del documento (usalo para elegir qué buscar):"
INDEX_SUMMARY = "Resumen: {summary}"
INDEX_SECTIONS = "Secciones ({count}): {titles}"
INDEX_NO_SECTIONS = "El documento no tiene secciones detectadas."
PANORAMA_QUERY = "panorama del documento"

FIND_ALL_TOOL_NAME = "find_all"
FIND_ALL_TOOL_DESCRIPTION = (
    "Búsqueda exhaustiva en las imágenes, figuras y audio del documento: devuelve TODAS las "
    "ocurrencias de lo que se busca (cada una verificada) con su ubicación (página y zona, o "
    "minutos). Usala para 'dónde aparece X', 'en qué páginas hay X', 'cuántas veces', 'todas las "
    "imágenes de X'. 'query' describe lo buscado en pocas palabras (p. ej. 'un perro', "
    "'gráfico de barras'); no es una pregunta."
)
MODALITY_PARAM_DESCRIPTION = (
    "Tipo de fuente a favorecer: auto (por defecto), texto, imagen (fotos, figuras, gráficos, "
    "páginas escaneadas), audio o video."
)

ANSWER_USER_TEMPLATE = "Fuentes:\n\n{sources}\n\nPregunta: {question}"
SOURCES_NONE = (
    "(ninguna: la búsqueda no encontró nada relevante en el documento; respondé que no aparece)"
)
ATTACHED_IMAGE = "Imagen adjunta de la fuente [{n}]:"
SOURCE_MEDIA = " ({kind}{locator})"
SOURCE_NO_DESCRIPTION = "(sin descripción disponible)"
SOURCE_REGIONS = "\n  Regiones de la imagen: {text}"
SOURCE_PARENT = "\n  Contexto (imagen completa a la que pertenece): {text}"
MEDIA_KIND_LABEL = {"image": "imagen", "audio": "audio", "frame": "video"}
SOURCE_PAGE = " (pág. {page})"
SOURCE_SECTION = " [sección: {section}]"

TOOL_UNKNOWN = "Herramienta desconocida: {name}"
TOOL_LIMIT_REACHED = "Se alcanzó el límite de búsquedas; respondé con las fuentes ya recuperadas."
TOOL_NO_RESULTS = "Sin resultados."
FIND_ALL_HEADER = (
    "Se encontraron {count} ocurrencias de «{query}»{verified}. Listalas todas, con su "
    "ubicación y citando cada fuente [n]."
)
FIND_ALL_VERIFIED = ", verificadas una por una con visión"
FIND_ALL_UNVERIFIED = " (sin verificación visual: cada una indica su confianza)"
FIND_ALL_MIXED = " ({checked} verificadas con visión; el resto indica su confianza)"
FIND_ALL_NONE = (
    "No se encontró ninguna ocurrencia de «{query}» en las imágenes{checked}. "
    "Respondé que no aparece."
)
FIND_ALL_NONE_CHECKED = " (se revisaron y descartaron {count} candidatas)"
FIND_ALL_TRUNCATED = "\nHay {extra} ocurrencias más que no se listan (tope {limit})."
FIND_ALL_CONFIDENCE = " [confianza {level}]"
FIND_ALL_NO_MEDIA = "El documento no tiene imágenes ni audio indexados."

# Source catalog in the planner's system message (media counts per kind)
CATALOG_HEADER = "Fuentes del chat:"

# Provider probe (tools support detection)
PROBE_PING = "ping"
PROBE_TOOL_REQUEST = "Llamá a la herramienta ping."
PROBE_TOOL_NAME = "ping"
PROBE_TOOL_DESCRIPTION = "Llamá a esta herramienta."

# Follow-up suggestions and document overview (one non-streaming call each)
SUGGESTIONS_SYSTEM = (
    "Sugerís preguntas de seguimiento para una conversación sobre un documento. "
    "Respondé SOLO con un arreglo JSON de strings, sin texto adicional."
)
SUGGESTIONS_USER_TEMPLATE = (
    "Pregunta del usuario:\n{question}\n\nRespuesta dada:\n{answer}\n{sections}\n"
    "Proponé 3 preguntas de seguimiento naturales y cortas (máximo {max_chars} caracteres cada "
    "una), respondibles con el contenido del documento, en el mismo idioma que la respuesta y "
    'sin repetir la pregunta ya hecha. Devolvé un arreglo JSON: ["...", "...", "..."]'
)
SUGGESTIONS_SECTIONS = "\nSecciones de las fuentes: {sections}\n"

OVERVIEW_SYSTEM = (
    "Resumís documentos y proponés preguntas iniciales. "
    "Respondé SOLO con un objeto JSON, sin texto adicional."
)
OVERVIEW_USER_TEMPLATE = (
    "Secciones del documento: {sections}\n\nFragmentos representativos del documento:\n\n"
    "{excerpts}\n\n"
    'Devolvé un objeto JSON con dos claves: "summary" (2 a 4 oraciones, en el idioma del '
    'documento, qué es el documento y qué cubre) y "questions" (exactamente 3 preguntas '
    "iniciales cortas, máximo {max_chars} caracteres cada una, respondibles con el documento)."
)

# Persona ("talk to the author / a character") and spoiler-safe reading
PERSONA_INSTRUCTION = (
    "Respondé en primera persona, con la voz de «{persona}», pero SOLO con lo que digan las "
    "fuentes: no inventes hechos, recuerdos ni opiniones. Seguí citando cada afirmación con su "
    "fuente [n]. Si lo que preguntan no aparece en las fuentes, respondé en personaje que eso no "
    "lo contás en el libro o que no aparece en el documento."
)
SPOILER_ANSWER = (
    "El lector solo leyó hasta la página {page}. Respondé únicamente con lo ya leído (las "
    "fuentes provistas) y no adelantes nada posterior. Si la respuesta no está en lo leído, "
    "decí: «Eso todavía no lo leíste»."
)
SPOILER_PLANNER = (
    "Modo sin spoilers: el lector llegó hasta la página {page}; la búsqueda solo devuelve "
    "fragmentos hasta esa página."
)

# Claim check: one call splits the answer into claims and labels each against its sources
VERIFY_SYSTEM = (
    "Verificás respuestas contra sus fuentes. Respondé SOLO con un arreglo JSON, sin texto "
    "adicional."
)
VERIFY_USER_TEMPLATE = (
    "Oraciones de la respuesta (numeradas desde 0):\n{sentences}\n\nFuentes:\n\n{sources}\n\n"
    "Dividí la respuesta en afirmaciones atómicas (una idea verificable cada una). Para cada una "
    'devolvé un objeto: {{"text": afirmación, "sentence_index": número de la oración de la que '
    'sale, "status": "respaldada" | "parcial" | "sin_respaldo", "source_ns": [números de '
    'fuente que la respaldan], "quote": cita TEXTUAL breve de la fuente (omitila si no hay), '
    '"reason": una frase corta que justifica el estado}}. "respaldada" solo si la fuente lo dice '
    'explícitamente; "parcial" si lo respalda a medias o solo en parte; "sin_respaldo" si no '
    "aparece en las fuentes. Devolvé un arreglo JSON: [{{...}}, ...]"
)
VERIFY_SOURCE = "[{n}] {content}"
# Media sources are checked against their text (description / transcript), not the pixels
VERIFY_MEDIA_NOTE = {
    "image": "(descripción de una imagen) ",
    "video": "(descripción de un fotograma) ",
    "audio": "(transcripción de audio) ",
}
