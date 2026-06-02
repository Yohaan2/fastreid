from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # Application
    app_env: str = Field(default="production", description="Entorno: development | production")
    log_level: str = Field(default="INFO", description="Nivel de logging")

    # Security
    api_key: str = Field(..., description="API Key interna requerida")

    # Model paths
    model_path_person: str = Field(default="weights/person_model.pth")
    model_path_vehicle: str = Field(default="weights/vehicle_model.pth")
    fastreid_config_person: str = Field(default="configs/bagtricks_R50_person.yml")
    fastreid_config_vehicle: str = Field(default="configs/bagtricks_R50_vehicle.yml")
    model_name_person: str = Field(default="fastreid_person")
    model_name_vehicle: str = Field(default="fastreid_vehicle")
    embedding_dimension: int = Field(default=2048)

    # Detección de objetos (YOLO)
    detection_model_path: str = Field(
        default="weights/yolov8n.pt",
        description="Pesos del detector YOLO. Si no existe, se descarga automáticamente.",
    )
    detection_confidence: float = Field(
        default=0.4, description="Confianza mínima para aceptar una detección"
    )
    detection_iou: float = Field(default=0.45, description="Umbral IoU para NMS")
    detection_person_class: int = Field(default=0, description="Clase COCO de persona")
    detection_vehicle_classes: str = Field(
        default="2,3,5,7",
        description="Clases COCO de vehículo (car, motorcycle, bus, truck)",
    )

    # Image validation
    max_image_mb: float = Field(default=5.0, description="Tamaño máximo de imagen en MB")
    min_image_width: int = Field(default=32, description="Ancho mínimo en píxeles")
    min_image_height: int = Field(default=32, description="Alto mínimo en píxeles")

    # Performance
    enable_gpu: bool = Field(default=False, description="Activar inferencia GPU (CUDA)")
    request_timeout: int = Field(default=30, description="Timeout máximo por request en segundos")

    # Rate limiting
    rate_limit_embed: str = Field(default="60/minute", description="Rate limit para endpoints /embed/*")
    rate_limit_health: str = Field(default="120/minute", description="Rate limit para /health")

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_mb * 1024 * 1024)

    @property
    def vehicle_class_ids(self) -> set[int]:
        """Parsea la lista CSV de clases de vehículo a un set de enteros."""
        try:
            return {
                int(c.strip())
                for c in self.detection_vehicle_classes.split(",")
                if c.strip()
            }
        except Exception:
            return {2, 3, 5, 7}  # Clases COCO por defecto si falla el parseo


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
