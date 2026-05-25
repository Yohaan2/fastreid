from fastapi import Depends

from app.core.security import validate_api_key
from app.services.fastreid_service import FastReIDService


def get_fastreid_service() -> FastReIDService:
    """
    Dependency que retorna el singleton del servicio FastReID.
    El modelo ya está cargado en memoria desde el startup de la app.
    """
    return FastReIDService.get_instance()


# Re-exporta para uso directo en rutas
RequireAPIKey = Depends(validate_api_key)
ReIDService = Depends(get_fastreid_service)
