"""Per-modality score calibration (plan §3.2, §3.5).

Cosines are only comparable inside one modality: unrelated image/text pairs already score
0.50-0.65 (the modality gap), so the text threshold (0.60) is unusable for images. The defaults
come from the spike; an embedding profile's `calibration` (migration 013, written by
scripts/calibrate_modalities.py) overrides them: recalibrating is editing the profile, not code.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Any

from chat_pdf.domain.errors import InvalidInput

if TYPE_CHECKING:
    from chat_pdf.domain.models import Chunk


@dataclass(frozen=True)
class Calibration:
    floor: float  # median cosine of unrelated pairs: relevance 0 at or below it
    threshold: float  # cosine above which a hit counts as real evidence


CalibrationTable = Mapping[str, Calibration]
MODALITIES = ("text", "image", "audio")

DEFAULT_CALIBRATION: dict[str, Calibration] = {
    "text": Calibration(floor=0.50, threshold=0.60),
    "image": Calibration(floor=0.55, threshold=0.66),
    "audio": Calibration(floor=0.50, threshold=0.68),
}


def space_of(modality: str) -> str:
    return "image" if modality == "frame" else modality  # a video frame is embedded as an image


def calibration_table(
    overrides: Mapping[str, Any] | None = None, text_threshold: float | None = None
) -> dict[str, Calibration]:
    """Defaults, then the app-level text threshold, then the profile's own calibration."""
    table = dict(DEFAULT_CALIBRATION)
    if text_threshold is not None:
        table["text"] = replace(table["text"], threshold=text_threshold)
    for modality, entry in (overrides or {}).items():
        if modality in MODALITIES:
            table[modality] = Calibration(float(entry["floor"]), float(entry["threshold"]))
    return table


def parse_calibration(raw: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Validate a profile's calibration payload; returns it normalized (plain floats)."""
    clean: dict[str, dict[str, float]] = {}
    for modality, entry in raw.items():
        if modality not in MODALITIES:
            raise InvalidInput("invalid_calibration", f"Modalidad desconocida: {modality}")
        try:
            floor, threshold = float(entry["floor"]), float(entry["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInput(
                "invalid_calibration", f"{modality}: se necesitan floor y threshold numéricos"
            ) from exc
        if not (0 <= floor < 1 and 0 <= threshold <= 1 and floor <= threshold):
            raise InvalidInput(
                "invalid_calibration",
                f"{modality}: se requiere 0 <= floor <= threshold <= 1 (floor < 1)",
            )
        clean[modality] = {"floor": floor, "threshold": threshold}
    return clean


def modality_relevance(
    cosine: float, modality: str, table: CalibrationTable | None = None
) -> float:
    """`clamp((cos - floor) / (1 - floor))` with the modality's floor, in [0, 1]."""
    floor = (table or DEFAULT_CALIBRATION)[space_of(modality)].floor
    return max(0.0, min(1.0, (cosine - floor) / (1 - floor)))


def above_threshold(cosine: float, modality: str, table: CalibrationTable | None = None) -> bool:
    return cosine >= (table or DEFAULT_CALIBRATION)[space_of(modality)].threshold


def chunk_relevance(chunk: "Chunk", table: CalibrationTable | None = None) -> float | None:
    """Displayed score: a media segment takes the better of its native cosine (judged in its own
    modality) and the cosine of its description (judged as text). None when nothing was measured."""
    scores = []
    if chunk.similarity is not None:
        scores.append(modality_relevance(chunk.similarity, "text", table))
    if chunk.native_similarity is not None:
        scores.append(modality_relevance(chunk.native_similarity, chunk.modality, table))
    return round(max(scores), 3) if scores else None


def chunk_margin(chunk: "Chunk", table: CalibrationTable | None = None) -> float | None:
    """Best `cosine - threshold` over the spaces measured (>= 0: real evidence); None if none."""
    table = table or DEFAULT_CALIBRATION
    margins = []
    if chunk.similarity is not None:
        margins.append(chunk.similarity - table["text"].threshold)
    if chunk.native_similarity is not None:
        margins.append(chunk.native_similarity - table[space_of(chunk.modality)].threshold)
    return max(margins) if margins else None


def fit_calibration(scores: list[float], labels: list[bool]) -> Calibration:
    """Calibrate one modality from labeled cosines (True = the pair is a real match).

    `floor` is the median of the unrelated pairs; `threshold` the cutoff of maximum F1 (ties go
    to the stricter cutoff), placed halfway to the next lower score so it separates, not touches.
    """
    if len(scores) != len(labels):
        raise ValueError("scores y labels deben tener el mismo largo")
    positives = [s for s, ok in zip(scores, labels, strict=True) if ok]
    negatives = [s for s, ok in zip(scores, labels, strict=True) if not ok]
    if not positives or not negatives:
        raise InvalidInput("calibration_needs_both", "Hacen falta pares correctos e incorrectos")
    floor = median(negatives)
    best_cut, best_f1 = max(scores), -1.0
    for cut in sorted(set(scores), reverse=True):  # strict ">" keeps the highest cut on ties
        tp = sum(p >= cut for p in positives)
        fp = sum(n >= cut for n in negatives)
        f1 = 2 * tp / (2 * tp + fp + (len(positives) - tp))
        if f1 > best_f1 + 1e-12:
            best_cut, best_f1 = cut, f1
    lower = [s for s in scores if s < best_cut]
    threshold = (best_cut + max(lower)) / 2 if lower else best_cut
    return Calibration(round(floor, 3), round(max(threshold, floor), 3))
