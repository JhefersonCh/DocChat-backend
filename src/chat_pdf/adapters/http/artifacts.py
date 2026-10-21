import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

from chat_pdf.application.artifacts import (
    Artifact,
    DeleteArtifact,
    GenerateArtifact,
    ListArtifacts,
    RequestArtifact,
)
from chat_pdf.application.concept_map import ExpandConceptMap
from chat_pdf.application.slides import AssessSlides

router = APIRouter(prefix="/api/v1")


class ArtifactBody(BaseModel):
    kind: str
    options: dict[str, Any] | None = None


class ArtifactOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    options: dict[str, Any]
    data: dict[str, Any] | None
    error: str | None
    created_at: datetime


def _out(a: Artifact) -> ArtifactOut:
    return ArtifactOut(**{f: getattr(a, f) for f in ArtifactOut.model_fields})


def _use_cases(request: Request):
    return request.app.state.use_cases


@router.post("/chats/{chat_id}/artifacts", status_code=202, response_model=ArtifactOut)
async def create_artifact(
    chat_id: uuid.UUID, body: ArtifactBody, background: BackgroundTasks, request: Request
):
    uc = _use_cases(request)
    request_artifact: RequestArtifact = uc.request_artifact
    generate: GenerateArtifact = uc.generate_artifact
    artifact = await request_artifact.execute(chat_id, body.kind, body.options)
    background.add_task(generate.execute, artifact.id)
    return _out(artifact)


class SlidesAssessmentOut(BaseModel):
    suitable: bool
    score: float
    reasons: list[str]
    recommended_slides: int
    audience: str | None = None
    outline_hint: list[str]


@router.post("/chats/{chat_id}/artifacts/slides/assess", response_model=SlidesAssessmentOut)
async def assess_slides(chat_id: uuid.UUID, request: Request):
    """Is the material worth a deck? One cheap LLM call over ALL the active sources."""
    use_case: AssessSlides = _use_cases(request).assess_slides
    return await use_case.execute(chat_id)


class ExpandBody(BaseModel):
    node_id: str


class ExpandOut(BaseModel):
    artifact: ArtifactOut
    added_nodes: list[str]
    added_links: list[str]


@router.post("/chats/{chat_id}/artifacts/{artifact_id}/expand", response_model=ExpandOut)
async def expand_artifact(
    chat_id: uuid.UUID, artifact_id: uuid.UUID, body: ExpandBody, request: Request
):
    """Grow a concept map around a node: 3-6 verified subconcepts (1 LLM call)."""
    use_case: ExpandConceptMap = _use_cases(request).expand_concept_map
    artifact, added = await use_case.execute(chat_id, artifact_id, body.node_id)
    return ExpandOut(
        artifact=_out(artifact), added_nodes=added["nodes"], added_links=added["links"]
    )


@router.get("/chats/{chat_id}/artifacts", response_model=list[ArtifactOut])
async def list_artifacts(chat_id: uuid.UUID, request: Request):
    use_case: ListArtifacts = _use_cases(request).list_artifacts
    return [_out(a) for a in await use_case.execute(chat_id)]


@router.delete("/chats/{chat_id}/artifacts/{artifact_id}", status_code=204)
async def delete_artifact(chat_id: uuid.UUID, artifact_id: uuid.UUID, request: Request):
    use_case: DeleteArtifact = _use_cases(request).delete_artifact
    await use_case.execute(chat_id, artifact_id)
