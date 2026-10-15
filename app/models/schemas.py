from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Literal


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(examples=["ok"])
    model_person_loaded: bool
    model_vehicle_loaded: bool
    gpu_available: bool
    device: str = Field(examples=["cpu", "cuda:0"])


class EmbeddingResponse(BaseModel):
    object_type: Literal["person", "vehicle"] = Field(
        default="person",
        description="Tipo de objeto embebido (person | vehicle)",
    )
    model: str = Field(examples=["msmt17_sbs_R101-ibn"])
    dimension: int = Field(examples=[2048])
    embedding: Annotated[list[float], Field(min_length=1)]
    processing_ms: int = Field(examples=[38], description="Tiempo de inferencia en milisegundos")


class BoundingBox(BaseModel):
    """Bounding box en formato x, y, width, height (píxeles absolutos)."""

    x: float = Field(..., description="Coordenada x inicial (esquina superior izquierda)")
    y: float = Field(..., description="Coordenada y inicial (esquina superior izquierda)")
    width: float = Field(..., description="Ancho del bounding box")
    height: float = Field(..., description="Alto del bounding box")


class DetectionEmbedding(BaseModel):
    """Par de bounding box, tipo de objeto, confianza del detector y embedding visual."""

    object_type: Literal["person", "vehicle"] = Field(
        default="person",
        description="Tipo de objeto detectado y embebido (person | vehicle)",
    )
    model: str = Field(
        default="msmt17_sbs_R101-ibn",
        description="Modelo ReID utilizado para generar este embedding",
    )
    bbox: BoundingBox
    confidence: float = Field(..., description="Confianza de la detección entre 0 y 1")
    dimension: int = Field(examples=[2048], description="Dimensión del embedding visual")
    embedding: Annotated[list[float], Field(min_length=1)]


class MultiEmbeddingResponse(BaseModel):
    """Respuesta para endpoints que procesan imágenes completas o consultas con múltiples objetos."""

    model: str = Field(examples=["msmt17_sbs_R101-ibn+sbs_R50-ibn_vehicle"])
    count: int = Field(..., description="Cantidad de objetos detectados y procesados")
    detections: list[DetectionEmbedding] = Field(default_factory=list)
    processing_ms: int = Field(..., description="Tiempo total de procesamiento en milisegundos")


class ErrorResponse(BaseModel):
    detail: str

