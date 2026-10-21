import pytest

from chat_pdf.domain.audit import audit_facts
from chat_pdf.domain.numbers import find_numbers, parse_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$44.000.000", 44_000_000),
        ("44 millones", 44_000_000),
        ("12,5 %", 12.5),
        ("1.2M", 1_200_000),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
        ("1,234,567", 1_234_567),
        ("3 mil millones", 3e9),
        ("(1.200)", -1200),
        ("-15", -15),
        ("2,5 millones", 2_500_000),
        ("1.500", 1500),
        ("0.5", 0.5),
        ("120k", 120_000),
        ("sin cifras", None),
        ("1 y 2", None),
    ],
)
def test_parse_number_spanish_formats(text, expected):
    assert parse_number(text) == expected


def test_find_numbers_reports_scaled_value_mantissa_and_percent():
    assert find_numbers("creció 12,5 % hasta 44 millones") == [
        (12.5, 12.5, "%"),
        (44e6, 44.0, ""),
    ]
    assert find_numbers("entre 2020-2021") == [(2020, 2020, ""), (2021, 2021, "")]


def fact(label, value, group=None, total=False, kind="amount", unit="COP", page=1):
    return {
        "label": label, "value": value, "unit": unit, "kind": kind, "group": group,
        "is_total": total, "quote": f"{label} {value}", "page": page, "chunk_id": "c",
    }  # fmt: skip


def types(result):
    return [f["type"] for f in result["findings"]]


def test_group_sum_mismatch_and_match():
    bad = [fact("Obra", 30, "Presupuesto"), fact("Equipos", 20, "Presupuesto"),
           fact("Total", 60, "Presupuesto", total=True)]  # fmt: skip
    result = audit_facts(bad)
    (finding,) = result["findings"]
    assert finding["type"] == "sum_mismatch" and finding["expected"] == 50
    assert finding["actual"] == 60 and len(finding["evidence"]) == 3
    assert "Presupuesto" in finding["message"] and result["checked"]["sum_groups"] == 1
    ok = [*bad[:2], fact("Total", 50.2, "Presupuesto", total=True)]  # within rounding
    assert audit_facts(ok)["findings"] == []


def test_percent_groups_must_add_to_100():
    over = audit_facts([fact("A", 60, "Mercado", kind="percent", unit="%"),
                        fact("B", 55, "Mercado", kind="percent", unit="%")])  # fmt: skip
    assert types(over) == ["percent_total"] and over["findings"][0]["severity"] == "high"
    short = audit_facts([fact("A", 40, "M", kind="percent"), fact("B", 40, "M", kind="percent")])
    assert short["findings"][0]["severity"] == "low"
    fine = audit_facts([fact("A", 33.3, "M", kind="percent"), fact("B", 66.5, "M", kind="percent")])
    assert fine["findings"] == [] and fine["checked"]["percent_groups"] == 1


def test_same_metric_with_different_values_fuzzy_label():
    result = audit_facts([
        fact("Ingresos anuales proyectados", 120_000_000, page=2),
        fact("Ingresos anuales proyectado", 150_000_000, page=7),
    ])  # fmt: skip
    (finding,) = result["findings"]
    assert finding["type"] == "conflicting_values"
    assert [e["page"] for e in finding["evidence"]] == [2, 7]
    same = audit_facts([fact("Inversión inicial", 5e6), fact("inversion inicial", 5e6)])
    assert same["findings"] == [] and same["checked"]["repeated_metrics"] == 1


def test_scale_and_unit_mismatch():
    scale = audit_facts([fact("Ventas 2024", 4_400_000), fact("Ventas 2024", 4_400)])
    assert types(scale) == ["scale_mismatch"]
    unit = audit_facts([fact("Capital", 10, unit="USD"), fact("Capital", 10, unit="COP")])
    assert types(unit) == ["unit_mismatch"]


def test_different_groups_and_generic_labels_do_not_collide():
    result = audit_facts([
        fact("Total", 10, "Año 1", total=True), fact("Total", 99, "Año 2", total=True),
        fact("Ingresos", 1, "Plan A"), fact("Ingresos", 2, "Plan B"),
    ])  # fmt: skip
    assert result["findings"] == []


def test_clean_report_states_what_was_checked():
    result = audit_facts([fact("X", 1), fact("Y", 2)])
    assert result["findings"] == [] and result["facts_count"] == 2
    assert result["message"].startswith("No se encontraron inconsistencias")
    assert "2 cifras" in result["message"]
