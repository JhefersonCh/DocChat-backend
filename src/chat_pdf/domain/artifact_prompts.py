"""Prompts for the study artifacts. One non-streaming call each; JSON only."""

ARTIFACT_SYSTEM = (
    "Generás material de estudio a partir de fragmentos de un documento. Cada fragmento viene "
    "marcado con su página como [pág. N]. Usá SOLO información de los fragmentos, en el idioma "
    'del documento. En cada ítem incluí "page": el número de página (entero) de donde sale. '
    "Respondé SOLO con JSON válido, sin texto adicional."
)

CONTEXT = "Secciones del documento: {sections}\n\nFragmentos:\n\n{excerpts}\n\n"

INSTRUCTIONS = {
    "faq": (
        'Devolvé {{"items": [{{"question": "...", "answer": "...", "page": N}}]}} con 6 a 10 '
        "preguntas frecuentes y sus respuestas (2 a 4 oraciones), cubriendo todo el documento."
    ),
    "study_guide": (
        'Devolvé {{"concepts": [{{"term": "...", "definition": "...", "page": N}}], '
        '"glossary": [{{"term": "...", "definition": "...", "page": N}}], '
        '"sections": [{{"title": "...", "summary": "...", "page": N}}]}} con 5 a 8 conceptos '
        "clave, 5 a 12 términos de glosario y un resumen breve por cada sección principal."
    ),
    "quiz": (
        'Devolvé {{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer_index": 0, "explanation": "...", "page": N}}]}} con 5 a 10 preguntas de '
        "opción múltiple con exactamente 4 opciones, una sola correcta (answer_index es su "
        "posición, 0 a 3) y una explicación breve. Variá la posición de la correcta."
    ),
    "flashcards": (
        'Devolvé {{"cards": [{{"front": "...", "back": "...", "page": N}}]}} con 10 a 20 '
        "tarjetas: el frente es un término o pregunta corta y el dorso la respuesta concisa."
    ),
    "mind_map": (
        'Devolvé {{"title": "...", "page": N, "children": [{{"title": "...", "page": N, '
        '"children": [...]}}]}}: un esquema jerárquico del documento (raíz = tema del '
        "documento, 4 a 8 ramas, hasta 3 niveles, títulos de 1 a 6 palabras)."
    ),
    "chart": (
        'Extraé datos NUMÉRICOS presentes en el texto o tablas y devolvé {{"chart": {{"type": '
        '"bar" | "line" | "pie", "title": "...", "labels": ["..."], "series": [{{"name": '
        '"...", "values": [números]}}], "page": N}}}}. Cada serie tiene un valor por etiqueta; '
        "no inventes ni estimes números. Si no hay datos numéricos graficables devolvé "
        '{{"chart": null}}.{request}'
    ),
    "extraction": (
        "Extraé datos estructurados del documento. Columnas pedidas: {columns}. Devolvé "
        '{{"rows": [{{"<columna>": {{"value": "...", "quote": "cita textual breve", '
        '"page": N}}}}]}} usando exactamente esos nombres de columna. Normalmente es UNA fila '
        "(el documento entero); devolvé varias solo si el documento lista varios ítems "
        "(p. ej. varias facturas o personas). Si no encontrás una columna usá "
        '{{"value": null}}.'
    ),
    "audit": (
        "Extraé HECHOS NUMÉRICOS (montos, porcentajes, cantidades, fechas) de los fragmentos. "
        'Devolvé {{"facts": [{{"label": "qué mide", "value": número, "unit": "COP | USD | % | '
        'unidades...", "kind": "amount" | "percent" | "count" | "date" | "other", "group": '
        '"conjunto al que pertenece (p. ej. presupuesto inicial) o null", "is_total": true | '
        'false, "quote": "cita textual EXACTA con la cifra", "page": N}}]}}. Si una cifra es el '
        "total de un conjunto de partes, marcala is_total=true y usá el MISMO group en sus "
        'partes. "value" es el número completo (44 millones = 44000000; 12,5 % = 12.5). '
        "No calcules ni inventes cifras: solo las que están escritas."
    ),
    "entity_graph": (
        "Extraé un grafo de conocimiento. Devolvé "
        '{{"entities": [{{"name": "...", "type": "persona" | "organización" | "lugar" | '
        '"concepto" | "objeto" | "evento", "description": "1 oración", "aliases": ["otros '
        'nombres con que aparece"], "page": N}}], "relations": [{{"source": "nombre de la '
        'entidad", "target": "nombre de la entidad", "label": "relación corta (p. ej. hijo de, '
        'compite con, financia)", "quote": "cita textual EXACTA que sostiene la relación", '
        '"page": N}}]}}. Hasta 25 entidades y 40 relaciones; solo las más importantes. Usá un '
        "único nombre canónico por entidad y ponelo igual en las relaciones."
    ),
    "rubric": (
        "Evaluá el documento contra esta lista de criterios (usá exactamente estos id):\n"
        "{criteria}\n"
        'Devolvé {{"criteria": [{{"id": "...", "status": "cumple" | "parcial" | '
        '"no_encontrado", "justification": "1 a 2 oraciones", "quote": "cita textual EXACTA '
        'del documento que lo respalda (obligatoria salvo no_encontrado)", "page": N, '
        '"suggestion": "cómo mejorarlo"}}]}} con un ítem por criterio. Sé estricto: '
        "no_encontrado si el documento no lo trata."
    ),
    "timeline": (
        "Extraé los eventos con fecha u orden cronológico. Devolvé "
        '{{"events": [{{"date_text": "la fecha tal como está escrita (p. ej. marzo de 2025)", '
        '"date_iso": "YYYY | YYYY-MM | YYYY-MM-DD o null", "title": "evento en pocas '
        'palabras", "description": "1 oración", "quote": "cita textual EXACTA que contiene la '
        'fecha", "page": N}}]}} con hasta 25 eventos. date_text DEBE aparecer en la cita. Si el '
        'documento no tiene contenido temporal devolvé {{"events": []}}.'
    ),
}
CHART_REQUEST = "\nPedido del usuario: {request}"
