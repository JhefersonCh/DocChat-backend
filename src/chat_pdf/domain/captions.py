"""Image captions (plan §2.6): the prompt a vision model gets and the parser of its answer.

The caption is stored as the segment's text (`content`), so the full-text index and the text
embedding see it and a text-only chat model can reason about what the picture shows."""

import json
import re
from dataclasses import dataclass, field

CAPTION_MAX_TOKENS = 10000  # ample: reasoning models think first and return "" on a short budget
CONTEXT_CHARS = 600  # nearby page text passed to the model to disambiguate figures

CAPTION_PROMPT = (
    "Describí esta imagen con el mayor detalle posible, en español.\n"
    "Respondé SOLO con un objeto JSON (sin texto antes ni después, sin bloques de código) "
    "con estas claves:\n"
    '{"tipo": "foto|gráfico|tabla|diagrama|captura|documento|otro",\n'
    ' "descripcion": "descripción exhaustiva: qué se ve, escena, personas, animales, objetos, '
    'colores, disposición",\n'
    ' "objetos": ["lista corta de objetos, seres y elementos visibles"],\n'
    ' "texto_visible": "todo el texto legible en la imagen, transcripto tal cual (OCR); '
    'cadena vacía si no hay",\n'
    ' "datos": null}\n'
    'Si es un gráfico (barras, líneas, torta, dispersión, etc.), en "descripcion" indicá el '
    'tipo de gráfico y su título, y en "datos" poné\n'
    '{"ejes": {"x": "nombre y unidades", "y": "nombre y unidades"}, '
    '"series": [{"nombre": "...", "valores": {"etiqueta": valor}}], '
    '"tendencia": "tendencia principal"}\n'
    "con los valores leídos lo más precisos posible (aproximados si hace falta). "
    'Si es una tabla, "datos" lleva sus filas y columnas.\n'
    'Si no es gráfico ni tabla, "datos" es null. No inventes lo que no se ve.\n'
)


MAX_REGIONS = 8
MIN_REGION_SIDE = 0.02  # a region narrower / shorter than 2 % of the image is noise
MAX_REGION_AREA = 0.85  # one covering (almost) everything is the whole image, already a segment
DUPLICATE_IOU = 0.7  # two regions this alike are one

REGIONS_PROMPT = (
    'Además agregá la clave "regiones": una lista (máx. 8) de los objetos, personas, gráficos '
    "o bloques de texto más relevantes de la imagen, cada uno como\n"
    '{"etiqueta": "nombre corto", "descripcion": "una frase", '
    '"bbox": [x0, y0, x1, y1]}\n'
    "con bbox normalizado entre 0 y 1 (x0,y0 = esquina superior izquierda; x1,y1 = inferior "
    "derecha) respecto de la imagen completa. Lista vacía si la imagen es un solo objeto.\n"
)


@dataclass(frozen=True)
class CaptionRegion:
    """A salient thing inside the image: what it is and where (bbox normalized to the image)."""

    label: str
    description: str
    bbox: list[float]

    @property
    def text(self) -> str:
        return f"{self.label}: {self.description}" if self.description else self.label


@dataclass(frozen=True)
class Caption:
    kind: str  # foto | gráfico | tabla | diagrama | captura | documento | otro
    description: str
    objects: list[str] = field(default_factory=list)
    visible_text: str = ""
    data: object | None = None  # chart axes / series / values, or table rows
    regions: list[CaptionRegion] = field(default_factory=list)
    regions_given: int = -1  # items the model listed; -1: it sent no valid `regiones` list

    @property
    def needs_grid(self) -> bool:
        """No usable regions, yet the model did not say "a single object": use a grid."""
        return not self.regions and self.regions_given != 0

    @property
    def content(self) -> str:
        """Readable text stored as the segment's `content` (what the chat model is shown)."""
        lines = [f"[{self.kind}] {self.description}".strip()]
        if self.objects:
            lines.append("Objetos: " + ", ".join(self.objects))
        if self.visible_text:
            lines.append("Texto visible: " + self.visible_text)
        if self.data:
            lines.append("Datos: " + json.dumps(self.data, ensure_ascii=False))
        return "\n".join(lines)

    @property
    def index_text(self) -> str:
        """What is full-text indexed and embedded: description + objects + visible text + data."""
        data = json.dumps(self.data, ensure_ascii=False) if self.data else ""
        parts = [self.kind, self.description, " ".join(self.objects), self.visible_text, data]
        return " ".join(p for p in parts if p).strip()


def caption_prompt(context: str | None = None, regions: bool = False) -> str:
    """The user prompt, with the nearby page text (figure legend, section) when there is any.
    `regions` also asks for the salient objects with their bbox (same call, no extra cost)."""
    prompt = CAPTION_PROMPT + (REGIONS_PROMPT if regions else "")
    snippet = " ".join((context or "").split())[:CONTEXT_CHARS]
    if not snippet:
        return prompt
    near = f'Texto cercano en el documento (contexto, puede mencionar la figura): "{snippet}"'
    return f"{prompt}\n{near}"


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _iou(a: list[float], b: list[float]) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_regions(raw: object) -> list[CaptionRegion]:
    """Valid regions of the model's `regiones`: bbox clamped to 0-1 (a model that answers in
    0-1000 or pixels-of-1000 is rescaled), corners ordered, tiny / whole-image / duplicate boxes
    and unlabeled ones dropped, at most 8. Anything malformed is skipped, never raised."""
    if not isinstance(raw, list):
        return []
    found: list[CaptionRegion] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), list):
            continue
        nums = [_number(v) for v in item["bbox"]]
        if len(nums) != 4 or any(v is None for v in nums):
            continue
        scale = 1000.0 if max(nums) > 1.5 else 1.0  # type: ignore[type-var]
        x0, y0, x1, y1 = (min(1.0, max(0.0, v / scale)) for v in nums)  # type: ignore[operator]
        box = [
            round(min(x0, x1), 4),
            round(min(y0, y1), 4),
            round(max(x0, x1), 4),
            round(max(y0, y1), 4),
        ]
        w, h = box[2] - box[0], box[3] - box[1]
        label = _text(item.get("etiqueta") or item.get("label"))
        if not label or w < MIN_REGION_SIDE or h < MIN_REGION_SIDE or w * h > MAX_REGION_AREA:
            continue
        if any(_iou(box, r.bbox) >= DUPLICATE_IOU for r in found):
            continue
        found.append(CaptionRegion(label, _text(item.get("descripcion")), box))
        if len(found) == MAX_REGIONS:
            break
    return found


def _json_object(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_caption(raw: str) -> Caption | None:
    """Caption from the model's answer. Tolerates code fences and plain prose (a model that
    ignores the JSON instruction still produced a usable description). None if it is empty."""
    if not raw or not raw.strip():
        return None
    obj = _json_object(raw)
    if obj is None:
        return Caption("otro", " ".join(raw.split()))
    description = _text(obj.get("descripcion"))
    objects = obj.get("objetos")
    names = [_text(o) for o in objects if _text(o)] if isinstance(objects, list) else []
    caption = Caption(
        _text(obj.get("tipo")).lower() or "otro",
        description,
        names,
        _text(obj.get("texto_visible")),
        obj.get("datos") or None,
        parse_regions(obj.get("regiones")),
        len(obj["regiones"]) if isinstance(obj.get("regiones"), list) else -1,
    )
    return caption if (description or names or caption.visible_text or caption.data) else None
