from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response

from chat_pdf.adapters.http.deps import (
    get_manage_providers,
    get_probe_provider,
    get_set_active_profiles,
)
from chat_pdf.adapters.http.schemas import (
    ActiveSettings,
    ProbeOut,
    ProviderIn,
    ProviderOut,
    ProviderPatch,
)
from chat_pdf.application.providers import ManageProviders, ProbeProvider, SetActiveProfiles

router = APIRouter(prefix="/api/v1")
Manage = Annotated[ManageProviders, Depends(get_manage_providers)]


def _active(ids) -> ActiveSettings:
    return ActiveSettings(
        chat_profile_id=ids["chat"],
        embedding_profile_id=ids["embedding"],
        stt_profile_id=ids.get("stt"),
        tts_profile_id=ids.get("tts"),
        vision_profile_id=ids.get("vision"),
    )


@router.get("/providers", response_model=list[ProviderOut])
async def list_providers(manage: Manage):
    return [ProviderOut.of(p) for p in await manage.list()]


@router.post("/providers", response_model=ProviderOut, status_code=201)
async def create_provider(body: ProviderIn, manage: Manage):
    return ProviderOut.of(await manage.create(body.model_dump()))


@router.patch("/providers/{profile_id}", response_model=ProviderOut)
async def update_provider(profile_id: UUID, body: ProviderPatch, manage: Manage):
    return ProviderOut.of(await manage.update(profile_id, body.model_dump(exclude_unset=True)))


@router.delete("/providers/{profile_id}", status_code=204)
async def delete_provider(profile_id: UUID, manage: Manage):
    await manage.delete(profile_id)
    return Response(status_code=204)


@router.post(
    "/providers/{profile_id}/test", response_model=ProbeOut, response_model_exclude_none=True
)
async def test_provider_endpoint(
    profile_id: UUID, probe: Annotated[ProbeProvider, Depends(get_probe_provider)]
):
    return await probe.execute(profile_id)


@router.get("/settings/active", response_model=ActiveSettings)
async def get_active(manage: Manage):
    return _active(await manage.active_ids())


@router.put("/settings/active", response_model=ActiveSettings)
async def put_active(
    body: ActiveSettings, use_case: Annotated[SetActiveProfiles, Depends(get_set_active_profiles)]
):
    """Only fields present in the body change; an explicit null deactivates."""
    changes = {
        kind: getattr(body, f"{kind}_profile_id")
        for kind in ("chat", "embedding", "stt", "tts", "vision")
        if f"{kind}_profile_id" in body.model_fields_set
    }
    return _active(await use_case.execute(changes))
