from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_fastreid_service, RequireAPIKey
from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import EmbeddingResponse
from app.services.fastreid_service import FastReIDService
from app.utils.image import bytes_to_pil
from app.utils.validation import validate_upload_image

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


@router.post(
    "/embed/person",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar embedding de persona",
    description=(
        "Recibe un crop de persona (multipart/form-data) y retorna su embedding visual. "
        "Requiere header `x-api-key`."
    ),
    tags=["Embeddings"],
    dependencies=[RequireAPIKey],
)
@limiter.limit(settings.rate_limit_embed)
async def embed_person(
    request: Request,
    image: UploadFile = File(..., description="Imagen recortada de la persona (JPEG/PNG/WebP, máx 5MB)"),
    service: FastReIDService = Depends(get_fastreid_service),
) -> EmbeddingResponse:
    if not service.person_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de persona no disponible. Verificar pesos en weights/",
        )

    raw = await validate_upload_image(image)

    try:
        pil_image = bytes_to_pil(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        embedding, elapsed_ms = service.embed_person(pil_image)
    except RuntimeError as exc:
        logger.error("embed_person_inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error durante la inferencia",
        ) from exc

    logger.info(
        "embed_person_ok",
        dimension=len(embedding),
        processing_ms=elapsed_ms,
        filename=image.filename,
    )

    return EmbeddingResponse(
        model=settings.model_name_person,
        dimension=len(embedding),
        embedding=embedding,
        processing_ms=elapsed_ms,
    )


@router.post(
    "/embed/vehicle",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar embedding de vehículo",
    description=(
        "Recibe un crop de vehículo (multipart/form-data) y retorna su embedding visual. "
        "Requiere header `x-api-key`."
    ),
    tags=["Embeddings"],
    dependencies=[RequireAPIKey],
)
@limiter.limit(settings.rate_limit_embed)
async def embed_vehicle(
    request: Request,
    image: UploadFile = File(..., description="Imagen recortada del vehículo (JPEG/PNG/WebP, máx 5MB)"),
    service: FastReIDService = Depends(get_fastreid_service),
) -> EmbeddingResponse:
    if not service.vehicle_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de vehículo no disponible. Verificar pesos en weights/",
        )

    raw = await validate_upload_image(image)

    try:
        pil_image = bytes_to_pil(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        embedding, elapsed_ms = service.embed_vehicle(pil_image)
    except RuntimeError as exc:
        logger.error("embed_vehicle_inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error durante la inferencia",
        ) from exc

    logger.info(
        "embed_vehicle_ok",
        dimension=len(embedding),
        processing_ms=elapsed_ms,
        filename=image.filename,
    )

    return EmbeddingResponse(
        model=settings.model_name_vehicle,
        dimension=len(embedding),
        embedding=embedding,
        processing_ms=elapsed_ms,
    )
