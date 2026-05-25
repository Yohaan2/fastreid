from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_fastreid_service
from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Comprueba si el servicio está activo y si los modelos están cargados.",
    tags=["Observabilidad"],
)
@limiter.limit(settings.rate_limit_health)
async def health(request: Request) -> HealthResponse:
    service = get_fastreid_service()
    import torch

    return HealthResponse(
        status="ok",
        model_person_loaded=service.person_model_loaded,
        model_vehicle_loaded=service.vehicle_model_loaded,
        gpu_available=torch.cuda.is_available(),
        device=str(service.device),
    )
