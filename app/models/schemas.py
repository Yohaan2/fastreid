from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated


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


class ErrorResponse(BaseModel):
    detail: str
