from chat_pdf.domain.cleaning import normalize, strip_boilerplate
from chat_pdf.domain.models import Page

HEADER = "R A Í C E S D E L P U T U M A Y O \nPlan de Negocios \nDocumento · 2026 {n}\n{n} \n"


def test_strip_boilerplate_drops_running_header_and_page_number_but_keeps_body_numbers():
    bodies = {2: "Proveedores", 3: "Portafolio", 4: "Tienda"}
    pages = [
        Page(1, "P L A N D E N E G O C I O S \nPortada"),
        *(Page(n, HEADER.format(n=n) + f"1 \n{b}\n2 \nMás {b}") for n, b in bodies.items()),
    ]

    cleaned = strip_boilerplate(pages)

    assert cleaned[0] == pages[0]
    assert cleaned[1].text == "1 \nProveedores\n2 \nMás Proveedores"
    assert [p.number for p in cleaned] == [1, 2, 3, 4]


def test_normalize_collapses_letter_spacing_and_hyphenation_but_not_single_letter_words():
    text = "S E C C I Ó N 0 9 \n9. Área de  mercado\nR A Í C E S y café o té a la comer-\ncializar"

    assert normalize(text) == (
        "SECCIÓN 09 9. Área de mercado RAÍCES y café o té a la comercializar"
    )
