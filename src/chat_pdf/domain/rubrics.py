"""Fixed evaluation checklists per document type."""

PRESETS = ("plan_negocios", "articulo_academico", "ensayo", "general")
DEFAULT_PRESET = "general"
STATUSES = ("cumple", "parcial", "no_encontrado")

PRESET_LABELS = {
    "plan_negocios": "Plan de negocios",
    "articulo_academico": "Artículo académico",
    "ensayo": "Ensayo",
    "general": "General",
}

_CRITERIA: dict[str, list[tuple[str, str]]] = {
    "plan_negocios": [
        ("resumen", "Resumen ejecutivo"),
        ("problema", "Problema u oportunidad"),
        ("mercado", "Mercado objetivo"),
        ("competencia", "Análisis de competencia"),
        ("propuesta", "Propuesta de valor"),
        ("ingresos", "Modelo de ingresos"),
        ("finanzas", "Plan financiero"),
        ("equipo", "Equipo"),
        ("riesgos", "Riesgos y mitigación"),
        ("cronograma", "Cronograma / hitos"),
    ],
    "articulo_academico": [
        ("resumen", "Resumen / abstract"),
        ("problema", "Planteamiento del problema"),
        ("objetivos", "Objetivos o preguntas de investigación"),
        ("marco", "Marco teórico / antecedentes"),
        ("metodologia", "Metodología"),
        ("resultados", "Resultados"),
        ("discusion", "Discusión"),
        ("conclusiones", "Conclusiones"),
        ("limitaciones", "Limitaciones"),
        ("referencias", "Referencias bibliográficas"),
    ],
    "ensayo": [
        ("tesis", "Tesis clara"),
        ("introduccion", "Introducción"),
        ("argumentos", "Argumentos que sustentan la tesis"),
        ("evidencia", "Evidencia o ejemplos"),
        ("contraargumentos", "Contraargumentos considerados"),
        ("estructura", "Estructura y coherencia"),
        ("conclusion", "Conclusión"),
        ("fuentes", "Fuentes citadas"),
    ],
    "general": [
        ("proposito", "Propósito claro"),
        ("estructura", "Estructura organizada"),
        ("contenido", "Contenido sustentado"),
        ("datos", "Datos o evidencia"),
        ("conclusiones", "Conclusiones o cierre"),
        ("claridad", "Claridad de redacción"),
    ],
}


def criteria(preset: str) -> list[dict[str, str]]:
    return [{"id": i, "label": label} for i, label in _CRITERIA.get(preset, _CRITERIA["general"])]
