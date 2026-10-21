import json
from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import UUID

from chat_pdf.domain.models import Citation


def _encode(obj: Any) -> Any:
    """Domain objects -> JSON (the domain itself knows nothing about JSON)."""
    if isinstance(obj, Citation):  # the API names the source: id / alias / name
        return {**asdict(obj), "source_id": obj.document_id, "source_name": obj.filename}
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=_encode)}\n\n"
