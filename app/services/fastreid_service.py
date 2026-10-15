"""
Singleton de inferencia FastReID con soporte para aceleración ONNX Runtime en CPU
y micro-batching dinámico.

Responsabilidades:
- Cargar modelos UNA SOLA VEZ al inicio del servicio (o bajo demanda).
- Soportar ONNX Runtime en CPU cuando el archivo `.onnx` esté disponible.
- Exponer métodos de inferencia individual, micro-batch y detección completa (persona + vehículo).
- Normalizar embeddings L2.
- Retornar el vector como list[float] + tiempo de procesamiento.

NO conoce: cámaras, eventos, usuarios, zonas, base de datos.
"""
import os
import time
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger
from app.services.batching import BatchProcessor
from app.services.detection_service import Detection, DetectionService
from app.services.preprocessing import (
    PERSON_HEIGHT,
    PERSON_TRANSFORM,
    PERSON_WIDTH,
    VEHICLE_HEIGHT,
    VEHICLE_TRANSFORM,
    VEHICLE_WIDTH,
    preprocess_image,
)

logger = get_logger(__name__)

_lock = threading.Lock()


class _FastReIDOnnxExportWrapper(torch.nn.Module):
    """
    Envuelve el modelo FastReID para que durante `torch.onnx.export` reciba un tensor
    `(N, 3, H, W)` ya normalizado (ImageNet mean/std) y retorne directamente el tensor
    de features `(N, 2048)` sin requerir un diccionario `{"images": ...}` ni deshacer
    la normalización dos veces dentro de `Baseline.preprocess_image`.
    """

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        # FastReID Baseline resta pixel_mean y divide por pixel_std internamente.
        # Como `PERSON_TRANSFORM`/`VEHICLE_TRANSFORM` ya entrega el tensor normalizado
        # con ImageNet stats, reconstruimos la escala [0..255] esperada por Baseline
        # o llamamos directamente a backbone + heads si están expuestos.
        self._has_backbone_heads = hasattr(base_model, "backbone") and hasattr(
            base_model, "heads"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_backbone_heads:
            feats = self.base_model.backbone(x)
            out = self.base_model.heads(feats)
        else:
            out = self.base_model(x)
        if isinstance(out, dict):
            return out["features"]
        return out


@dataclass
class DetectedEmbedding:
    """Resultado de detectar + recortar + embeber un único objeto."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    embedding: list[float]
    object_type: Literal["person", "vehicle"] = "person"
    model: str = ""


class FastReIDService:
    """Wrapper singleton sobre los modelos FastReID de persona y vehículo."""

    _instance: Optional["FastReIDService"] = None

    def __init__(self) -> None:
        self._device = self._resolve_device()

        # Optimizar hilos de PyTorch en CPU según configuración (evitar contención excesiva)
        if self._device.type == "cpu":
            threads = max(1, settings.cpu_threads)
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            logger.info("pytorch_threads_optimized", num_threads=threads, num_interop_threads=1)

        self._person_model: Optional[torch.nn.Module] = None
        self._vehicle_model: Optional[torch.nn.Module] = None
        self._person_onnx_session: Optional[Any] = None
        self._vehicle_onnx_session: Optional[Any] = None
        self._person_load_attempted: bool = False
        self._vehicle_load_attempted: bool = False

        self._batch_processor = BatchProcessor(
            batch_infer_fn=self._run_batch_inference_by_type,
            max_batch_size=settings.batch_max_size,
            window_ms=settings.batch_window_ms,
        )

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
    # Lazy Loading + Auto-Exportación a ONNX Runtime
    # ------------------------------------------------------------------

    def _resolve_writable_onnx_path(self, preferred_path: str, label: str) -> str:
        """
        Retorna `preferred_path` si existe o su directorio es escribible;
        de lo contrario usa `/tmp/{label}_model.onnx` (útil si `/app/weights` está montado `:ro`).
        """
        if os.path.exists(preferred_path):
            return preferred_path
        fallback_tmp = os.path.join("/tmp", f"{label}_model.onnx")
        if os.path.exists(fallback_tmp):
            return fallback_tmp
        parent_dir = os.path.dirname(os.path.abspath(preferred_path)) or "."
        if os.path.isdir(parent_dir) and os.access(parent_dir, os.W_OK):
            return preferred_path
        return fallback_tmp

    def _export_pytorch_to_onnx(
        self,
        pt_model: torch.nn.Module,
        onnx_path: str,
        height: int,
        width: int,
        label: str,
    ) -> Optional[str]:
        """
        Exporta automáticamente el modelo PyTorch `.pth` a `.onnx` la primera vez
        que arranca el servicio en CPU, con eje `batch_size` dinámico.
        """
        target_path = self._resolve_writable_onnx_path(onnx_path, label)
        try:
            import onnx  # noqa: F401

            logger.info(
                "onnx_auto_export_started",
                label=label,
                target_path=target_path,
                input_shape=f"Nx3x{height}x{width}",
            )
            wrapper = _FastReIDOnnxExportWrapper(pt_model).eval().to("cpu")
            dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32, device="cpu")

            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    target_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["features"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "features": {0: "batch_size"},
                    },
                )

            logger.info("onnx_auto_export_completed", label=label, onnx_path=target_path)
            return target_path
        except Exception as exc:
            logger.warning(
                "onnx_auto_export_failed_using_pytorch",
                label=label,
                error=str(exc),
            )
            return None

    def _try_load_onnx_session(self, onnx_path: str, label: str) -> Optional[Any]:
        if not settings.enable_onnx or self._device.type != "cpu":
            return None
        resolved_path = self._resolve_writable_onnx_path(onnx_path, label)
        if not os.path.exists(resolved_path):
            return None
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, settings.cpu_threads)
            sess_options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                resolved_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            logger.info("onnx_model_loaded", label=label, onnx_path=resolved_path)
            return session
        except Exception as exc:
            logger.warning("onnx_load_failed_fallback_pytorch", label=label, error=str(exc))
            return None

    def _ensure_person_loaded(self) -> None:
        if self._person_load_attempted:
            return
        with _lock:
            if self._person_load_attempted:
                return
            self._person_onnx_session = self._try_load_onnx_session(
                settings.model_onnx_path_person, "person"
            )
            if self._person_onnx_session is None:
                logger.info("lazy_loading_model_started", label="person")
                pt_model = self._load_single_model(
                    config_path=settings.fastreid_config_person,
                    weights_path=settings.model_path_person,
                    label="person",
                )
                if pt_model is not None and settings.enable_onnx and self._device.type == "cpu":
                    exported_path = self._export_pytorch_to_onnx(
                        pt_model=pt_model,
                        onnx_path=settings.model_onnx_path_person,
                        height=PERSON_HEIGHT,
                        width=PERSON_WIDTH,
                        label="person",
                    )
                    if exported_path:
                        self._person_onnx_session = self._try_load_onnx_session(
                            exported_path, "person"
                        )
                # Si ONNX cargó con éxito, liberamos el modelo PyTorch para ahorrar RAM
                if self._person_onnx_session is not None:
                    self._person_model = None
                else:
                    self._person_model = pt_model
            self._person_load_attempted = True

    def _ensure_vehicle_loaded(self) -> None:
        if self._vehicle_load_attempted:
            return
        with _lock:
            if self._vehicle_load_attempted:
                return
            self._vehicle_onnx_session = self._try_load_onnx_session(
                settings.model_onnx_path_vehicle, "vehicle"
            )
            if self._vehicle_onnx_session is None:
                logger.info("lazy_loading_model_started", label="vehicle")
                pt_model = self._load_single_model(
                    config_path=settings.fastreid_config_vehicle,
                    weights_path=settings.model_path_vehicle,
                    label="vehicle",
                )
                if pt_model is not None and settings.enable_onnx and self._device.type == "cpu":
                    exported_path = self._export_pytorch_to_onnx(
                        pt_model=pt_model,
                        onnx_path=settings.model_onnx_path_vehicle,
                        height=VEHICLE_HEIGHT,
                        width=VEHICLE_WIDTH,
                        label="vehicle",
                    )
                    if exported_path:
                        self._vehicle_onnx_session = self._try_load_onnx_session(
                            exported_path, "vehicle"
                        )
                if self._vehicle_onnx_session is not None:
                    self._vehicle_model = None
                else:
                    self._vehicle_model = pt_model
            self._vehicle_load_attempted = True

    @property
    def person_onnx_session(self) -> Optional[Any]:
        self._ensure_person_loaded()
        return self._person_onnx_session

    @property
    def vehicle_onnx_session(self) -> Optional[Any]:
        self._ensure_vehicle_loaded()
        return self._vehicle_onnx_session

    @property
    def person_model(self) -> Optional[torch.nn.Module]:
        self._ensure_person_loaded()
        return self._person_model

    @property
    def vehicle_model(self) -> Optional[torch.nn.Module]:
        self._ensure_vehicle_loaded()
        return self._vehicle_model

    def warmup(self) -> None:
        """
        Pre-carga (y auto-exporta a ONNX si corresponde) los modelos de persona y vehículo
        en el startup de la aplicación para evitar latencia en la primera petición.
        """
        self._ensure_person_loaded()
        self._ensure_vehicle_loaded()

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

            logger.info("model_loaded", label=label, config=config_path, device=str(self._device))
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
        if self._person_onnx_session is not None or self._person_model is not None:
            return True
        if self._person_load_attempted:
            return False
        return os.path.exists(settings.model_onnx_path_person) or os.path.exists(
            settings.model_path_person
        )

    @property
    def vehicle_model_loaded(self) -> bool:
        if self._vehicle_onnx_session is not None or self._vehicle_model is not None:
            return True
        if self._vehicle_load_attempted:
            return False
        return os.path.exists(settings.model_onnx_path_vehicle) or os.path.exists(
            settings.model_path_vehicle
        )

    @property
    def gpu_available(self) -> bool:
        return torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Inferencia pública (crops directos + micro-batching asíncrono)
    # ------------------------------------------------------------------

    def embed_person(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding síncrono para un crop de persona."""
        return self._run_inference(
            image=image,
            model=self.person_model,
            onnx_session=self.person_onnx_session,
            transform=PERSON_TRANSFORM,
            label="person",
        )

    def embed_vehicle(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding síncrono para un crop de vehículo."""
        return self._run_inference(
            image=image,
            model=self.vehicle_model,
            onnx_session=self.vehicle_onnx_session,
            transform=VEHICLE_TRANSFORM,
            label="vehicle",
        )

    async def embed_person_batched(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding de persona agrupando peticiones concurrentes en micro-batch."""
        return await self._batch_processor.submit(image, "person")

    async def embed_vehicle_batched(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding de vehículo agrupando peticiones concurrentes en micro-batch."""
        return await self._batch_processor.submit(image, "vehicle")

    # ------------------------------------------------------------------
    # Detección + crop temporal + embedding (imagen completa o query forense)
    # ------------------------------------------------------------------

    def embed_persons_from_image(self, image: Image.Image) -> tuple[list[DetectedEmbedding], int]:
        """
        Analiza una imagen completa, detecta personas, recorta cada una
        (crop temporal) y genera su embedding en un único batch optimizado.
        """
        t_total = time.perf_counter()
        detector = DetectionService.get_instance()

        t_det = time.perf_counter()
        detections = detector.detect_persons(image)
        det_ms = int((time.perf_counter() - t_det) * 1000)
        logger.info("detection_stage", label="person", count=len(detections), detection_ms=det_ms)

        results, embed_ms = self._detect_crop_embed(
            image=image,
            detections=detections,
            model=self.person_model,
            onnx_session=self.person_onnx_session,
            transform=PERSON_TRANSFORM,
            label="person",
            model_name=settings.model_name_person,
        )

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            "pipeline_total",
            label="person",
            detection_ms=det_ms,
            embedding_ms=embed_ms,
            total_ms=total_ms,
        )
        return results, total_ms

    def embed_vehicles_from_image(self, image: Image.Image) -> tuple[list[DetectedEmbedding], int]:
        """
        Analiza una imagen completa, detecta vehículos (car, motorcycle, bus, truck),
        recorta cada uno y genera su embedding en un único batch optimizado.
        """
        t_total = time.perf_counter()
        detector = DetectionService.get_instance()

        t_det = time.perf_counter()
        detections = detector.detect_vehicles(image)
        det_ms = int((time.perf_counter() - t_det) * 1000)
        logger.info("detection_stage", label="vehicle", count=len(detections), detection_ms=det_ms)

        results, embed_ms = self._detect_crop_embed(
            image=image,
            detections=detections,
            model=self.vehicle_model,
            onnx_session=self.vehicle_onnx_session,
            transform=VEHICLE_TRANSFORM,
            label="vehicle",
            model_name=settings.model_name_vehicle,
        )

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            "pipeline_total",
            label="vehicle",
            detection_ms=det_ms,
            embedding_ms=embed_ms,
            total_ms=total_ms,
        )
        return results, total_ms

    def embed_query_from_image(self, image: Image.Image) -> tuple[list[DetectedEmbedding], int]:
        """
        Pipeline unificado para búsqueda forense (`/embed/query`):
        1. Ejecuta YOLO una sola vez detectando tanto `person` como `vehicle`.
        2. Si no detecta objetos en el primer intento (ej. el usuario subió un crop
           muy cerrado al borde), aplica el fallback con padding de 20% y conf=0.15.
        3. Separa las detecciones de personas y vehículos, genera sus embeddings en
           batch con su modelo respectivo (`person_model` / `vehicle_model`) y
           retorna la lista unificada ordenada por área/confianza.
        """
        t_total = time.perf_counter()
        detector = DetectionService.get_instance()

        t_det = time.perf_counter()
        detections = detector.detect_persons_and_vehicles(image, enable_crop_fallback=True)
        det_ms = int((time.perf_counter() - t_det) * 1000)

        person_dets = [d for d in detections if d.object_type == "person"]
        vehicle_dets = [d for d in detections if d.object_type == "vehicle"]

        logger.info(
            "detection_stage_query",
            total_count=len(detections),
            person_count=len(person_dets),
            vehicle_count=len(vehicle_dets),
            detection_ms=det_ms,
        )

        combined_results: list[DetectedEmbedding] = []

        if person_dets:
            if not self.person_model_loaded:
                raise RuntimeError("Modelo person no disponible — pesos no cargados")
            person_results, _ = self._detect_crop_embed(
                image=image,
                detections=person_dets,
                model=self.person_model,
                onnx_session=self.person_onnx_session,
                transform=PERSON_TRANSFORM,
                label="person",
                model_name=settings.model_name_person,
            )
            combined_results.extend(person_results)

        if vehicle_dets:
            if not self.vehicle_model_loaded:
                raise RuntimeError("Modelo vehicle no disponible — pesos no cargados")
            vehicle_results, _ = self._detect_crop_embed(
                image=image,
                detections=vehicle_dets,
                model=self.vehicle_model,
                onnx_session=self.vehicle_onnx_session,
                transform=VEHICLE_TRANSFORM,
                label="vehicle",
                model_name=settings.model_name_vehicle,
            )
            combined_results.extend(vehicle_results)

        # Ordenar por confianza descendente para que el sujeto más relevante vaya primero
        combined_results.sort(key=lambda item: item.confidence, reverse=True)

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            "pipeline_query_total",
            count=len(combined_results),
            detection_ms=det_ms,
            total_ms=total_ms,
        )
        return combined_results, total_ms

    # ------------------------------------------------------------------
    # Inferencia en batch e individual (PyTorch / ONNX Runtime)
    # ------------------------------------------------------------------

    def _forward_batch_tensor(
        self,
        batch_tensor: torch.Tensor,
        model: Optional[torch.nn.Module],
        onnx_session: Optional[Any],
        label: str,
    ) -> torch.Tensor:
        if onnx_session is not None:
            input_name = onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: batch_tensor.detach().cpu().numpy()}
            ort_outs = onnx_session.run(None, ort_inputs)
            raw_np = ort_outs[0]
            features = torch.from_numpy(np.asarray(raw_np, dtype=np.float32))
        elif model is not None:
            with torch.inference_mode():
                output = model(batch_tensor.to(self._device))
            features = output["features"] if isinstance(output, dict) else output
            features = features.detach().cpu()
        else:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        return F.normalize(features, p=2, dim=1)

    def _run_batch_inference_by_type(
        self,
        images: list[Image.Image],
        object_type: Literal["person", "vehicle"],
    ) -> tuple[list[list[float]], int]:
        if object_type == "person":
            transform = PERSON_TRANSFORM
            model = self.person_model
            onnx_session = self.person_onnx_session
        else:
            transform = VEHICLE_TRANSFORM
            model = self.vehicle_model
            onnx_session = self.vehicle_onnx_session

        t_start = time.perf_counter()
        tensors = []
        for img in images:
            rgb = img.convert("RGB") if img.mode != "RGB" else img
            tensors.append(transform(rgb))

        batch_tensor = torch.stack(tensors)
        features = self._forward_batch_tensor(
            batch_tensor=batch_tensor,
            model=model,
            onnx_session=onnx_session,
            label=object_type,
        )
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        return features.tolist(), elapsed_ms

    def _detect_crop_embed(
        self,
        image: Image.Image,
        detections: list[Detection],
        model: Optional[torch.nn.Module],
        onnx_session: Optional[Any],
        transform,
        label: Literal["person", "vehicle"],
        model_name: str,
    ) -> tuple[list[DetectedEmbedding], int]:
        if model is None and onnx_session is None:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        t_start = time.perf_counter()
        rgb = image.convert("RGB") if image.mode != "RGB" else image

        if not detections:
            return [], 0

        t_crop = time.perf_counter()
        crops = [rgb.crop((det.x1, det.y1, det.x2, det.y2)) for det in detections]
        crop_ms = int((time.perf_counter() - t_crop) * 1000)

        tensors = []
        for crop in crops:
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            tensors.append(transform(crop))

        batch_tensor = torch.stack(tensors)
        preprocess_ms = int((time.perf_counter() - t_crop) * 1000)

        t_inference = time.perf_counter()
        features = self._forward_batch_tensor(
            batch_tensor=batch_tensor,
            model=model,
            onnx_session=onnx_session,
            label=label,
        )
        inference_ms = int((time.perf_counter() - t_inference) * 1000)

        embeddings_list = features.tolist()

        results: list[DetectedEmbedding] = []
        for det, embedding in zip(detections, embeddings_list):
            results.append(
                DetectedEmbedding(
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    confidence=det.confidence,
                    embedding=embedding,
                    object_type=label,
                    model=model_name,
                )
            )

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "detect_crop_embed_ok",
            label=label,
            count=len(results),
            crop_ms=crop_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            total_ms=elapsed_ms,
        )
        return results, elapsed_ms

    def _run_inference(
        self,
        image: Image.Image,
        model: Optional[torch.nn.Module],
        onnx_session: Optional[Any],
        transform,
        label: str,
    ) -> tuple[list[float], int]:
        if model is None and onnx_session is None:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        t_start = time.perf_counter()
        tensor = preprocess_image(image, transform, torch.device("cpu"))
        features = self._forward_batch_tensor(
            batch_tensor=tensor,
            model=model,
            onnx_session=onnx_session,
            label=label,
        )

        embedding: list[float] = features.squeeze(0).tolist()
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        return embedding, elapsed_ms
