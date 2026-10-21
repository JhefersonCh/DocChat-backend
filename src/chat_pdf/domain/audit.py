"""Deterministic consistency checks over numeric facts extracted by the model (no LLM here).

A fact is a dict: label, value (float), unit, kind (amount|percent|count|date|other),
group (str|None), is_total (bool), quote, page, chunk_id.
"""

import math
from difflib import SequenceMatcher
from typing import Any

from chat_pdf.domain.text_match import norm

SUM_REL_TOL = 0.005  # a total may differ from its parts by 0.5 % (rounding)
SUM_ABS_TOL = 0.5
PERCENT_TOL = 1.0
SAME_REL_TOL = 0.005
LABEL_SIMILARITY = 0.88
SCALE_FACTORS = (10, 100, 1000, 1e6)
_GENERIC = {"total", "subtotal", "suma", "otros", "otro", "de", "del", "la", "el", "los", "las"}
_SUMMABLE = ("amount", "count")


def _evidence(fact: dict) -> dict:
    return {k: fact.get(k) for k in ("quote", "page", "chunk_id", "value")}


def _finding(type_: str, severity: str, message: str, facts: list[dict], **extra: Any) -> dict:
    return {
        "type": type_,
        "severity": severity,
        "message": message,
        "evidence": [_evidence(f) for f in facts],
        **extra,
    }


def fmt(value: float, unit: str = "") -> str:
    text = f"{value:,.2f}".rstrip("0").rstrip(".") if value != int(value) else f"{int(value):,}"
    text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"{text} %" if unit == "%" else text


def _group_key(fact: dict) -> str:
    return norm(fact.get("group") or "")


def _by_group(facts: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        key = _group_key(fact)
        if key:
            groups.setdefault(key, []).append(fact)
    return groups


def check_sums(facts: list[dict]) -> tuple[list[dict], int]:
    """(a) the components of a group add up to its stated total."""
    findings, checked = [], 0
    for members in _by_group(facts).values():
        parts = [f for f in members if not f["is_total"] and f["kind"] in _SUMMABLE]
        totals = [f for f in members if f["is_total"] and f["kind"] in _SUMMABLE]
        if len(parts) < 2 or not totals:
            continue
        checked += 1
        added = sum(f["value"] for f in parts)
        if any(abs(added - t["value"]) <= max(SUM_REL_TOL * abs(t["value"]), SUM_ABS_TOL)
               for t in totals):  # fmt: skip
            continue
        total = totals[0]
        gap = abs(added - total["value"]) / (abs(total["value"]) or 1)
        group = members[0]["group"]
        findings.append(_finding(
            "sum_mismatch",
            "high" if gap > 0.05 else "medium",
            f"En «{group}» las partes suman {fmt(added)} pero el total declarado es "
            f"{fmt(total['value'])} (diferencia de {fmt(abs(added - total['value']))}).",
            [*parts, total],
            expected=added,
            actual=total["value"],
        ))  # fmt: skip
    return findings, checked


def check_percentages(facts: list[dict]) -> tuple[list[dict], int]:
    """(b) the percentages of a group add up to ~100."""
    findings, checked = [], 0
    for members in _by_group(facts).values():
        parts = [f for f in members if f["kind"] == "percent" and not f["is_total"]]
        if len(parts) < 2:
            continue
        checked += 1
        added = sum(f["value"] for f in parts)
        if abs(added - 100) <= PERCENT_TOL:
            continue
        over = added > 100
        findings.append(_finding(
            "percent_total",
            "high" if over else "low",
            f"En «{members[0]['group']}» los porcentajes suman {fmt(added)} %"
            + (", más de 100 %." if over else " (podría faltar alguna categoría)."),
            parts,
            expected=100,
            actual=added,
        ))  # fmt: skip
    return findings, checked


def _label_tokens(label: str) -> list[str]:
    return [t for t in norm(label).split() if t not in _GENERIC]


def _same_metric(a: dict, b: dict) -> bool:
    ta, tb = _label_tokens(a["label"]), _label_tokens(b["label"])
    if not ta or not tb or _group_key(a) != _group_key(b):
        return False
    sa, sb = " ".join(sorted(ta)), " ".join(sorted(tb))
    return sa == sb or SequenceMatcher(None, sa, sb).ratio() >= LABEL_SIMILARITY


def _words(text: str) -> set[str]:
    return {t for t in norm(text).split() if len(t) > 2 and not t.isdigit()}


def _same_context(a: dict, b: dict) -> bool:
    """Quotes that talk about different things (other words around the figure) are not a
    repetition of one metric; quotes with no words at all (a bare "35%") are inconclusive."""
    wa, wb = _words(a.get("quote") or ""), _words(b.get("quote") or "")
    if not wa or not wb:
        return True
    return len(wa & wb) / len(wa | wb) >= 0.5


def _clusters(facts: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for fact in facts:
        if fact["kind"] == "date":
            continue
        for cluster in out:
            if _same_metric(cluster[0], fact):
                cluster.append(fact)
                break
        else:
            out.append([fact])
    return [c for c in out if len(c) > 1]


def _scale_ratio(a: float, b: float) -> float | None:
    if not a or not b:
        return None
    ratio = max(abs(a), abs(b)) / min(abs(a), abs(b))
    return next((f for f in SCALE_FACTORS if math.isclose(ratio, f, rel_tol=0.01)), None)


def check_repeated(facts: list[dict]) -> tuple[list[dict], int]:
    """(c) one metric stated with different values; (d) unit / scale mismatches among them."""
    findings, checked = [], 0
    for cluster in _clusters(facts):
        checked += 1
        label = cluster[0]["label"]
        first = cluster[0]
        for other in cluster[1:]:
            if not _same_context(first, other):
                continue
            if first["unit"] and other["unit"] and norm(first["unit"]) != norm(other["unit"]):
                findings.append(_finding(
                    "unit_mismatch", "medium",
                    f"«{label}» aparece en unidades distintas ({first['unit']} y "
                    f"{other['unit']}).", [first, other],
                ))  # fmt: skip
                continue
            if math.isclose(first["value"], other["value"], rel_tol=SAME_REL_TOL, abs_tol=1e-9):
                continue
            factor = _scale_ratio(first["value"], other["value"])
            if factor:
                findings.append(_finding(
                    "scale_mismatch", "high",
                    f"«{label}» difiere por un factor de {fmt(factor)} "
                    f"({fmt(first['value'])} vs {fmt(other['value'])}): posible error de escala.",
                    [first, other], expected=first["value"], actual=other["value"],
                ))  # fmt: skip
            else:
                findings.append(_finding(
                    "conflicting_values", "high",
                    f"«{label}» figura con valores distintos: {fmt(first['value'], first['unit'])}"
                    f" y {fmt(other['value'], other['unit'])}.",
                    [first, other], expected=first["value"], actual=other["value"],
                ))  # fmt: skip
    return findings, checked


def audit_facts(facts: list[dict]) -> dict:
    sums, n_sums = check_sums(facts)
    pcts, n_pcts = check_percentages(facts)
    repeated, n_rep = check_repeated(facts)
    findings = sorted([*sums, *pcts, *repeated], key=lambda f: f["severity"] != "high")
    checked = {"sum_groups": n_sums, "percent_groups": n_pcts, "repeated_metrics": n_rep}
    scope = (
        f"Se revisaron {len(facts)} cifras: {n_sums} grupos con total, "
        f"{n_pcts} grupos de porcentajes y {n_rep} métricas repetidas."
    )
    message = (
        f"Se encontraron {len(findings)} posibles inconsistencias. {scope}"
        if findings
        else f"No se encontraron inconsistencias. {scope}"
    )
    return {
        "findings": findings,
        "facts_count": len(facts),
        "checked_groups": n_sums + n_pcts + n_rep,
        "checked": checked,
        "message": message,
    }
