import imghdr
import io
from PIL import Image
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

ALLOWED_MIME_TYPES = {"jpeg", "png", "webp"}


async def validate_upload_image(file: UploadFile) -> bytes:
    """
    Valida un UploadFile de imagen y retorna sus bytes si pasa todas las comprobaciones.

    Comprobaciones:
    - Tamaño máximo (MAX_IMAGE_MB)
    - MIME type real (magic bytes, no el Content-Type del cliente)
    - Dimensiones mínimas
    - Integridad de la imagen (no corrupta)
    """
    raw = await file.read()

    # 1. Tamaño máximo
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Imagen supera el límite de {settings.max_image_mb} MB",
        )

    # 2. MIME type real mediante magic bytes
    detected = imghdr.what(None, h=raw)
    if detected not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de imagen no soportado: {detected}. Usar JPEG, PNG o WebP",
        )

    # 3. Dimensiones mínimas + integridad
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw))
        width, height = img.size
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Imagen corrupta o no procesable: {exc}",
        ) from exc

    if width < settings.min_image_width or height < settings.min_image_height:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Dimensiones insuficientes: {width}x{height}px. "
                f"Mínimo requerido: {settings.min_image_width}x{settings.min_image_height}px"
            ),
        )

    return raw
