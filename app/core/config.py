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


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
