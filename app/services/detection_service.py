"""
Singleton de detección de objetos (YOLO / Ultralytics).

Responsabilidades:
- Cargar el detector YOLO UNA SOLA VEZ al inicio del servicio.
- Analizar una imagen completa y devolver bounding boxes de personas
  o vehículos (equivalente a la detección de rostros de InsightFace,
  pero para cuerpo completo y vehículos).
- NO genera embeddings: solo localiza objetos y los devuelve como cajas.

El recorte (crop) y el embedding los realiza FastReIDService a partir de
las cajas que entrega este servicio.
"""
import os
import threading
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


@dataclass(frozen=True)
class Detection:
    """Una detección individual en coordenadas absolutas de la imagen."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


class DetectionService:
    """Wrapper singleton sobre el detector YOLO de Ultralytics."""

    _instance: Optional["DetectionService"] = None

    def __init__(self) -> None:
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Singleton thread-safe
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "DetectionService":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Carga el detector YOLO de forma estrictamente local y manual.
        No realiza ninguna descarga automática por red.
        """
        # Configurar variable de entorno para evitar warnings de permisos de escritura en Docker/Linux
        if "YOLO_CONFIG_DIR" not in os.environ:
            os.environ["YOLO_CONFIG_DIR"] = "/tmp"

        weights = settings.detection_model_path

        if not os.path.exists(weights):
            logger.error(
                "detection_weights_missing",
                weights_path=weights,
                hint=(
                    f"Falta el archivo de pesos local. Por favor colócalo manualmente "
                    f"en la ruta: {os.path.abspath(weights)}"
                )
            )
            self._model = None
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(weights)
            self._model.to("cuda" if settings.enable_gpu else "cpu")
            logger.info("detection_model_loaded", weights=weights)

        except ImportError:
            logger.warning(
                "ultralytics_not_installed",
                hint="Instalar detector: pip install ultralytics",
            )
            self._model = None
        except Exception as exc:
            logger.error(
                "detection_model_load_error",
                error=str(exc),
                hint=f"Error al inicializar YOLO con los pesos en: {os.path.abspath(weights)}"
            )
            self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Detección pública
    # ------------------------------------------------------------------

    def detect_persons(self, image: Image.Image) -> list[Detection]:
        """Detecta personas (cuerpo completo) en la imagen."""
        return self._detect(image, allowed_classes={settings.detection_person_class})

    def detect_vehicles(self, image: Image.Image) -> list[Detection]:
        """Detecta vehículos (car, motorcycle, bus, truck) en la imagen."""
        return self._detect(image, allowed_classes=settings.vehicle_class_ids)

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _detect(
        self,
        image: Image.Image,
        allowed_classes: set[int],
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Detector YOLO no disponible — modelo no cargado")

        rgb = image.convert("RGB") if image.mode != "RGB" else image

        results = self._model.predict(
            source=rgb,
            conf=settings.detection_confidence,
            iou=settings.detection_iou,
            classes=list(allowed_classes),
            verbose=False,
        )

        detections: list[Detection] = []
        width, height = rgb.size

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in allowed_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Clamp a los límites de la imagen
                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(width, int(x2))
                iy2 = min(height, int(y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                detections.append(
                    Detection(
                        x1=ix1,
                        y1=iy1,
                        x2=ix2,
                        y2=iy2,
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

        return detections
