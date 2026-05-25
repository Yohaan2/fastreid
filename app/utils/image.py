import io
from PIL import Image


def bytes_to_pil(data: bytes) -> Image.Image:
    """Convierte bytes a imagen PIL. Lanza ValueError si la imagen está corrupta."""
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # Detecta corrupción sin cargar píxeles
    except Exception as exc:
        raise ValueError(f"Imagen corrupta o inválida: {exc}") from exc

    # Reabrir tras verify() porque verify() cierra el stream interno
    image = Image.open(io.BytesIO(data))
    return image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Garantiza que la imagen esté en modo RGB."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
