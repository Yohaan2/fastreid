from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Optional


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(examples=["ok"])
    model_person_loaded: bool
    model_vehicle_loaded: bool
    gpu_available: bool
    device: str = Field(examples=["cpu", "cuda:0"])


class EmbeddingResponse(BaseModel):
    model: str = Field(examples=["fastreid_person"])
    dimension: int = Field(examples=[2048])
    embedding: Annotated[list[float], Field(min_length=1)]
    processing_ms: int = Field(examples=[38], description="Tiempo de inferencia en milisegundos")


class BoundingBox(BaseModel):
    """Caja delimitadora absoluta del objeto detectado (píxeles)."""

    x: float = Field(description="Coordenada X de la esquina superior izquierda")
    y: float = Field(description="Coordenada Y de la esquina superior izquierda")
    width: float
    height: float


class DetectionEmbedding(BaseModel):
    """Embedding de un único objeto detectado y recortado de la imagen."""

    bbox: BoundingBox
    confidence: float = Field(examples=[0.92], description="Confianza de la detección")
    dimension: int = Field(examples=[2048])
    embedding: Annotated[list[float], Field(min_length=1)]


class MultiEmbeddingResponse(BaseModel):
    """Respuesta para una imagen completa: N objetos detectados + embeddings."""

    model: str = Field(examples=["fastreid_person"])
    count: int = Field(examples=[2], description="Número de objetos detectados")
    detections: list[DetectionEmbedding]
    processing_ms: int = Field(examples=[120], description="Tiempo total (detección + embedding) en ms")


class ErrorResponse(BaseModel):
    detail: str
