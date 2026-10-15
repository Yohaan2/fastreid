"""
Singleton de detección de objetos (YOLO / Ultralytics) con Lazy Loading.

Responsabilidades:
- Cargar el detector YOLO bajo demanda (solo si se usa el endpoint de imagen completa).
- Analizar una imagen completa y devolver bounding boxes de personas.
- NO genera embeddings: solo localiza objetos y los devuelve como cajas.
"""
import os
import threading
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps

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

    @property
    def object_type(self) -> str:
        """Retorna 'person' o 'vehicle' según la clase COCO detectada."""
        if self.class_id == settings.detection_person_class:
            return "person"
        return "vehicle"


class DetectionService:
    """Wrapper singleton sobre el detector YOLO de Ultralytics con carga perezosa."""

    _instance: Optional["DetectionService"] = None

    def __init__(self) -> None:
        self._model = None

    @classmethod
    def get_instance(cls) -> "DetectionService":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def model(self):
        """Carga perezosa del modelo YOLO la primera vez que se solicita."""
        if self._model is None:
            with _lock:
                if self._model is None:
                    self._load_model()
        return self._model

    @property
    def model_loaded(self) -> bool:
        """Indica si el modelo de detección está cargado en memoria o si los pesos existen en el disco."""
        if self._model is not None:
            return True
        return os.path.exists(settings.detection_model_path)

    def _load_model(self) -> None:
        """Carga el detector YOLO de forma estrictamente local y manual."""
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

            logger.info("lazy_loading_yolo_started", weights=weights)
            model = YOLO(weights)
            model.to("cuda" if settings.enable_gpu else "cpu")
            self._model = model
            logger.info("detection_model_loaded", weights=weights)

        except ImportError:
            logger.warning(
                "ultralytics_not_installed",
                hint="Instalar detector: pip install ultralytics",
            )
            self._model = None
        except Exception as exc:
            logger.error("yolo_load_error", error=str(exc))
            self._model = None

    # ------------------------------------------------------------------
    # Detección pública
    # ------------------------------------------------------------------

    def detect_persons(self, image: Image.Image) -> list[Detection]:
        """Detecta personas (cuerpo completo) en la imagen."""
        return self._detect(image, allowed_classes={settings.detection_person_class})

    def detect_vehicles(self, image: Image.Image) -> list[Detection]:
        """Detecta vehículos (car, motorcycle, bus, truck) en la imagen."""
        return self._detect(image, allowed_classes=settings.vehicle_class_ids)

    def detect_persons_and_vehicles(
        self,
        image: Image.Image,
        enable_crop_fallback: bool = True,
    ) -> list[Detection]:
        """
        Detecta tanto personas como vehículos en una sola pasada de YOLO.
        Si no detecta ningún objeto (p. ej. el usuario subió un crop muy cerrado
        donde el sujeto toca los bordes de la imagen), aplica un segundo intento
        con 20% de padding negro y umbral reducido (`detection_fallback_confidence`).
        """
        allowed_classes = {settings.detection_person_class} | settings.vehicle_class_ids
        detections = self._detect(
            image,
            allowed_classes=allowed_classes,
            conf=settings.detection_confidence,
            imgsz=640,
        )
        if detections or not enable_crop_fallback:
            return detections

        # Fallback para crops cerrados: añadir 20% de margen negro y bajar umbral
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        width, height = rgb.size
        pad_x = max(8, int(width * 0.20))
        pad_y = max(8, int(height * 0.20))
        padded = ImageOps.expand(rgb, border=(pad_x, pad_y, pad_x, pad_y), fill=(0, 0, 0))

        retry_detections = self._detect(
            padded,
            allowed_classes=allowed_classes,
            conf=settings.detection_fallback_confidence,
            imgsz=640,
        )
        if not retry_detections:
            return []

        best = max(retry_detections, key=lambda d: d.confidence)
        logger.info(
            "detection_crop_fallback_matched",
            object_type=best.object_type,
            confidence=round(best.confidence, 4),
            original_size=f"{width}x{height}",
        )
        return [
            Detection(
                x1=0,
                y1=0,
                x2=width,
                y2=height,
                confidence=best.confidence,
                class_id=best.class_id,
            )
        ]

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _detect(
        self,
        image: Image.Image,
        allowed_classes: set[int],
        conf: Optional[float] = None,
        imgsz: int = 320,
    ) -> list[Detection]:
        yolo_model = self.model
        if yolo_model is None:
            raise RuntimeError("Detector YOLO no disponible — modelo no cargado o pesos faltantes")

        rgb = image.convert("RGB") if image.mode != "RGB" else image
        effective_conf = settings.detection_confidence if conf is None else conf

        results = yolo_model.predict(
            source=rgb,
            conf=effective_conf,
            iou=settings.detection_iou,
            classes=list(allowed_classes),
            imgsz=imgsz,
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
