from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import embed, health
from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.services.detection_service import DetectionService
from app.services.fastreid_service import FastReIDService

setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan handler:
    - Startup: carga modelos UNA SOLA VEZ en el singleton FastReIDService.
    - Shutdown: libera recursos GPU si aplica.
    """
    logger.info("service_startup", env=settings.app_env)
    FastReIDService.get_instance()
    DetectionService.get_instance()
    logger.info(
        "service_ready",
        person_model=FastReIDService.get_instance().person_model_loaded,
        vehicle_model=FastReIDService.get_instance().vehicle_model_loaded,
        detector=DetectionService.get_instance().model_loaded,
        device=str(FastReIDService.get_instance().device),
    )
    yield
    logger.info("service_shutdown")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="FastReID Embedding Service",
    description=(
        "Microservicio de generación de embeddings visuales para re-identificación "
        "de personas y vehículos. Parte del ecosistema OPTRAX."
    ),
    version="1.0.0",
    docs_url="/docs" if settings.app_env == "development" else None,
    redoc_url="/redoc" if settings.app_env == "development" else None,
    openapi_url="/openapi.json" if settings.app_env == "development" else None,
    lifespan=lifespan,
)

# Rate limiting global
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ------------------------------------------------------------------
# Manejo de excepciones genéricas — NO exponer stack traces
# ------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        method=request.method,
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"},
    )


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(health.router, prefix="", tags=["Observabilidad"])
app.include_router(embed.router, prefix="", tags=["Embeddings"])
