from typing import Annotated

from fastapi import APIRouter, Depends

from chat_pdf.adapters.http.deps import get_check_health
from chat_pdf.adapters.http.schemas import HealthOut, ProviderHealthOut
from chat_pdf.application.providers import CheckHealth

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthOut)
async def health(use_case: Annotated[CheckHealth, Depends(get_check_health)]):
    providers = await use_case.execute()
    return HealthOut(
        providers={
            kind: ProviderHealthOut(name=h.name, reachable=h.reachable) if h else None
            for kind, h in providers.items()
        }
    )
