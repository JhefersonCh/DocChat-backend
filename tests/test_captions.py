import json

from chat_pdf.domain.captions import caption_prompt, parse_caption


def test_parses_a_chart_caption_and_keeps_its_data_searchable():
    raw = json.dumps(
        {
            "tipo": "gráfico",
            "descripcion": "Gráfico de barras de ventas por año",
            "objetos": ["barras", "eje x"],
            "texto_visible": "Ventas 2024",
            "datos": {"series": [{"nombre": "ventas", "valores": {"2023": 10, "2024": 20}}]},
        }
    )

    caption = parse_caption(raw)

    assert caption.kind == "gráfico"
    assert "Datos:" in caption.content and '"2024": 20' in caption.content
    assert "Ventas 2024" in caption.index_text and "barras" in caption.index_text


def test_parses_json_wrapped_in_a_code_fence_and_prose():
    raw = 'Claro:\n```json\n{"tipo": "foto", "descripcion": "Un perro", "objetos": ["perro"]}\n```'

    caption = parse_caption(raw)

    assert (caption.kind, caption.description, caption.objects) == ("foto", "Un perro", ["perro"])


def test_plain_prose_becomes_the_description():
    caption = parse_caption("Un perro   marrón\ncorriendo en un parque.")

    assert (caption.kind, caption.description) == (
        "otro",
        "Un perro marrón corriendo en un parque.",
    )


def test_empty_answers_have_no_caption():
    assert parse_caption("") is None
    assert parse_caption("  \n") is None
    assert parse_caption('{"tipo": "foto", "descripcion": "", "objetos": []}') is None


def test_tolerates_wrong_field_types():
    caption = parse_caption('{"tipo": 3, "descripcion": "x", "objetos": "perro", "datos": []}')

    assert (caption.kind, caption.objects, caption.data) == ("otro", [], None)


def test_prompt_asks_for_charts_ocr_and_includes_nearby_text():
    prompt = caption_prompt("Figura 3: ventas   trimestrales")

    assert "gráfico" in prompt and "texto_visible" in prompt and "ejes" in prompt
    assert "Figura 3: ventas trimestrales" in prompt
    assert "Texto cercano" not in caption_prompt(None)


def test_regions_are_clamped_filtered_and_a_missing_list_asks_for_a_grid():
    from chat_pdf.domain.captions import parse_caption

    raw = (
        '{"descripcion": "x", "regiones": ['
        '{"etiqueta": "colibrí", "descripcion": "azul", "bbox": [0.5, 0.2, 0.1, 1.4]},'
        '{"etiqueta": "todo", "bbox": [0, 0, 1, 1]}, {"etiqueta": "?", "bbox": [1, 2]},'
        '{"etiqueta": "flor", "bbox": [100, 600, 400, 900]}]}'
    )
    caption = parse_caption(raw)

    assert [(r.label, r.bbox) for r in caption.regions] == [
        ("colibrí", [0.1, 0.2, 0.5, 1.0]),
        ("flor", [0.1, 0.6, 0.4, 0.9]),  # 0-1000 scale rescaled
    ]
    assert not caption.needs_grid and parse_caption('{"descripcion": "x"}').needs_grid
