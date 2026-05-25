import secrets
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


def validate_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """
    Dependency de FastAPI que valida el header x-api-key.
    Usa comparación en tiempo constante para evitar timing attacks.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key requerida",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key inválida",
        )

    return api_key
