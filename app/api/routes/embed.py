from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_fastreid_service, RequireAPIKey
from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import (
    EmbeddingResponse,
    BoundingBox,
    DetectionEmbedding,
    MultiEmbeddingResponse,
)
from app.services.fastreid_service import DetectedEmbedding, FastReIDService
from app.utils.image import bytes_to_pil
from app.utils.validation import validate_upload_image

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)


def _to_detection_schema(item: DetectedEmbedding) -> DetectionEmbedding:
    """Convierte el resultado interno del servicio al schema de respuesta."""
    x1, y1, x2, y2 = item.bbox
    return DetectionEmbedding(
        bbox=BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1)),
        confidence=item.confidence,
        dimension=len(item.embedding),
        embedding=item.embedding,
    )


@router.post(
    "/embed/person",
    response_model=MultiEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Detectar personas y generar embeddings",
    description=(
        "Recibe una imagen completa (multipart/form-data), detecta todas las personas, "
        "realiza un crop temporal de cada una y retorna su embedding visual. "
        "Requiere header `x-api-key`."
    ),
    tags=["Embeddings"],
    dependencies=[RequireAPIKey],
)
@limiter.limit(settings.rate_limit_embed)
async def embed_person(
    request: Request,
    image: UploadFile = File(..., description="Imagen completa a analizar (JPEG/PNG/WebP, máx 5MB)"),
    service: FastReIDService = Depends(get_fastreid_service),
) -> MultiEmbeddingResponse:
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
        detected, elapsed_ms = service.embed_persons_from_image(pil_image)
    except RuntimeError as exc:
        logger.error("embed_person_inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error durante la detección o inferencia",
        ) from exc

    logger.info(
        "embed_person_ok",
        count=len(detected),
        processing_ms=elapsed_ms,
        filename=image.filename,
    )

    return MultiEmbeddingResponse(
        model=settings.model_name_person,
        count=len(detected),
        detections=[_to_detection_schema(d) for d in detected],
        processing_ms=elapsed_ms,
    )


@router.post(
    "/embed/vehicle",
    response_model=MultiEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Detectar vehículos y generar embeddings",
    description=(
        "Recibe una imagen completa (multipart/form-data), detecta todos los vehículos, "
        "realiza un crop temporal de cada uno y retorna su embedding visual. "
        "Requiere header `x-api-key`."
    ),
    tags=["Embeddings"],
    dependencies=[RequireAPIKey],
)
@limiter.limit(settings.rate_limit_embed)
async def embed_vehicle(
    request: Request,
    image: UploadFile = File(..., description="Imagen completa a analizar (JPEG/PNG/WebP, máx 5MB)"),
    service: FastReIDService = Depends(get_fastreid_service),
) -> MultiEmbeddingResponse:
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
        detected, elapsed_ms = service.embed_vehicles_from_image(pil_image)
    except RuntimeError as exc:
        logger.error("embed_vehicle_inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error durante la detección o inferencia",
        ) from exc

    logger.info(
        "embed_vehicle_ok",
        count=len(detected),
        processing_ms=elapsed_ms,
        filename=image.filename,
    )

    return MultiEmbeddingResponse(
        model=settings.model_name_vehicle,
        dimension=len(embedding),
        embedding=embedding,
        processing_ms=elapsed_ms,
    )


@router.post(
    "/embed/person/full",
    response_model=MultiEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Detectar personas y generar embeddings de imagen completa",
    description=(
        "Recibe una imagen completa (multipart/form-data), detecta todas las personas, "
        "realiza un crop temporal de cada una y retorna su embedding visual en batch. "
        "Requiere header `x-api-key`."
    ),
    tags=["Embeddings"],
    dependencies=[RequireAPIKey],
)
@limiter.limit(settings.rate_limit_embed)
async def embed_person_full(
    request: Request,
    image: UploadFile = File(..., description="Imagen completa a analizar (JPEG/PNG/WebP, máx 5MB)"),
    service: FastReIDService = Depends(get_fastreid_service),
) -> MultiEmbeddingResponse:
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
        detected, elapsed_ms = service.embed_persons_from_image(pil_image)
    except RuntimeError as exc:
        logger.error("embed_person_full_inference_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error durante la detección o inferencia",
        ) from exc

    logger.info(
        "embed_person_full_ok",
        count=len(detected),
        processing_ms=elapsed_ms,
        filename=image.filename,
    )

    return MultiEmbeddingResponse(
        model=settings.model_name_person,
        count=len(detected),
        detections=[_to_detection_schema(d) for d in detected],
        processing_ms=elapsed_ms,
    )
