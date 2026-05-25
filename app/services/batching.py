"""
Módulo de batching — reservado para implementación futura.

Objetivo futuro:
    Agrupar múltiples crops recibidos en una ventana de tiempo
    y procesarlos en un único forward pass (batch de 8/16/32 imágenes).

Beneficio:
    Mejora significativa del throughput cuando hay GPU disponible,
    ya que el hardware subutilizado con inferencia 1-a-1.

Estado actual:
    NO implementado. El servicio procesa imagen por imagen.
    Esta interfaz está definida para facilitar la migración futura
    sin cambios en la capa de API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class BatchRequest:
    """Representa una solicitud pendiente de embedding dentro de un batch."""
    image: "Image.Image"
    object_type: str  # "person" | "vehicle"


@dataclass
class BatchResult:
    """Resultado de un embedding generado en batch."""
    embedding: list[float] = field(default_factory=list)
    processing_ms: int = 0


class BatchProcessor:
    """
    Placeholder para procesamiento en batch.
    Cuando se implemente, aceptará N imágenes y las procesará
    en un único forward pass para maximizar uso de GPU.
    """

    def __init__(self, max_batch_size: int = 16) -> None:
        self.max_batch_size = max_batch_size
        raise NotImplementedError(
            "BatchProcessor aún no está implementado. "
            "Usar FastReIDService.embed_person / embed_vehicle directamente."
        )
