"""
Singleton de inferencia FastReID.

Responsabilidades:
- Cargar modelos UNA SOLA VEZ al inicio del servicio.
- Exponer métodos de inferencia para persona y vehículo.
- Normalizar embeddings L2.
- Retornar el vector como list[float] + tiempo de procesamiento.

NO conoce: cámaras, eventos, usuarios, zonas, base de datos.
"""
import os
import time
import threading
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger
from app.services.preprocessing import (
    PERSON_TRANSFORM,
    VEHICLE_TRANSFORM,
    preprocess_image,
)

logger = get_logger(__name__)

_lock = threading.Lock()


class FastReIDService:
    """Wrapper singleton sobre los modelos FastReID de persona y vehículo."""

    _instance: Optional["FastReIDService"] = None

    def __init__(self) -> None:
        self._device = self._resolve_device()
        self._person_model: Optional[torch.nn.Module] = None
        self._vehicle_model: Optional[torch.nn.Module] = None
        self._load_models()

    # ------------------------------------------------------------------
    # Singleton thread-safe
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "FastReIDService":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if settings.enable_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("device_selected", device="cuda", gpu_name=torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            if settings.enable_gpu:
                logger.warning("gpu_requested_but_unavailable", fallback="cpu")
            else:
                logger.info("device_selected", device="cpu")
        return device

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Carga de modelos
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        self._person_model = self._load_single_model(
            config_path=settings.fastreid_config_person,
            weights_path=settings.model_path_person,
            label="person",
        )
        self._vehicle_model = self._load_single_model(
            config_path=settings.fastreid_config_vehicle,
            weights_path=settings.model_path_vehicle,
            label="vehicle",
        )

    def _load_single_model(
        self,
        config_path: str,
        weights_path: str,
        label: str,
    ) -> Optional[torch.nn.Module]:
        """
        Intenta cargar un modelo FastReID.
        Si los pesos no existen, registra warning y retorna None (el servicio
        seguirá levantando, pero ese endpoint devolverá 503).
        """
        if not os.path.exists(weights_path):
            logger.warning(
                "model_weights_not_found",
                label=label,
                weights_path=weights_path,
            )
            return None

        try:
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model
            from fastreid.utils.checkpoint import Checkpointer

            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.BACKBONE.PRETRAIN = False
            cfg.MODEL.DEVICE = str(self._device)
            cfg.CUDNN_BENCHMARK = False
            cfg.freeze()

            model = build_model(cfg)
            Checkpointer(model).load(weights_path)
            model.eval()
            model.to(self._device)

            logger.info("model_loaded", label=label, device=str(self._device))
            return model

        except ImportError:
            logger.warning(
                "fastreid_not_installed",
                label=label,
                hint="Instalar FastReID: pip install -e /opt/fast-reid",
            )
            return None
        except Exception as exc:
            logger.error("model_load_error", label=label, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Estado del servicio
    # ------------------------------------------------------------------

    @property
    def person_model_loaded(self) -> bool:
        return self._person_model is not None

    @property
    def vehicle_model_loaded(self) -> bool:
        return self._vehicle_model is not None

    @property
    def gpu_available(self) -> bool:
        return torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Inferencia pública
    # ------------------------------------------------------------------

    def embed_person(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding para un crop de persona."""
        return self._run_inference(
            image=image,
            model=self._person_model,
            transform=PERSON_TRANSFORM,
            label="person",
        )

    def embed_vehicle(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding para un crop de vehículo."""
        return self._run_inference(
            image=image,
            model=self._vehicle_model,
            transform=VEHICLE_TRANSFORM,
            label="vehicle",
        )

    # ------------------------------------------------------------------
    # Lógica de inferencia interna
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image: Image.Image,
        model: Optional[torch.nn.Module],
        transform,
        label: str,
    ) -> tuple[list[float], int]:
        if model is None:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        t_start = time.perf_counter()

        tensor = preprocess_image(image, transform, self._device)

        with torch.no_grad():
            output = model(tensor)

        # FastReID puede retornar dict o tensor directo según la config
        features: torch.Tensor = output["features"] if isinstance(output, dict) else output

        raw_norm = features.norm(p=2, dim=1).item()
        logger.debug(
            "embed_raw_features",
            label=label,
            raw_l2_norm=round(raw_norm, 4),
            first_5_raw=features.squeeze(0)[:5].cpu().tolist(),
        )

        # Normalización L2 — fundamental para cosine similarity en pgvector
        features = F.normalize(features, p=2, dim=1)

        embedding: list[float] = features.squeeze(0).cpu().tolist()
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        return embedding, elapsed_ms
