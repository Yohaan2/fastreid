"""
Singleton de inferencia FastReID.

Responsabilidades:
- Cargar modelos UNA SOLA VEZ al inicio del servicio.
- Exponer métodos de inferencia para persona y vehículo.
- Normalizar embeddings L2.
- Retornar el vector como list[float] + tiempo de procesamiento.

NO conoce: cámaras, eventos, usuarios, zonas, base de datos.
"""
import base64
import io
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger
from app.services.detection_service import Detection, DetectionService
from app.services.preprocessing import (
    PERSON_TRANSFORM,
    VEHICLE_TRANSFORM,
    preprocess_image,
)

logger = get_logger(__name__)

_lock = threading.Lock()


@dataclass
class DetectedEmbedding:
    """Resultado de detectar + recortar + embeber un único objeto."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    embedding: list[float]


class FastReIDService:
    """Wrapper singleton sobre los modelos FastReID de persona y vehículo."""

    _instance: Optional["FastReIDService"] = None

    def __init__(self) -> None:
        self._device = self._resolve_device()
        
        # Limitar hilos de PyTorch en CPU para evitar thread thrashing en VPS/Droplets
        if self._device.type == "cpu":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            logger.info("pytorch_threads_optimized", num_threads=1, num_interop_threads=1)
            
        self._person_model: Optional[torch.nn.Module] = None
        self._vehicle_model: Optional[torch.nn.Module] = None

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
    # Lazy Loading de modelos
    # ------------------------------------------------------------------

    @property
    def person_model(self) -> Optional[torch.nn.Module]:
        if self._person_model is None:
            with _lock:
                if self._person_model is None:
                    logger.info("lazy_loading_model_started", label="person")
                    self._person_model = self._load_single_model(
                        config_path=settings.fastreid_config_person,
                        weights_path=settings.model_path_person,
                        label="person",
                    )
        return self._person_model

    @property
    def vehicle_model(self) -> Optional[torch.nn.Module]:
        if self._vehicle_model is None:
            with _lock:
                if self._vehicle_model is None:
                    logger.info("lazy_loading_model_started", label="vehicle")
                    self._vehicle_model = self._load_single_model(
                        config_path=settings.fastreid_config_vehicle,
                        weights_path=settings.model_path_vehicle,
                        label="vehicle",
                    )
        return self._vehicle_model

    def _load_single_model(
        self,
        config_path: str,
        weights_path: str,
        label: str,
    ) -> Optional[object]:
        """
        Intenta cargar un modelo FastReID en formato ONNX con ONNX Runtime para máximo rendimiento en CPU.
        Si no existe el archivo .onnx pero sí el .pth, lo convierte automáticamente de forma local.
        """
        onnx_path = weights_path.replace(".pth", ".onnx")

        # 1. Intentar cargar el modelo ONNX directamente con ONNX Runtime
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                logger.info("loading_onnx_model", label=label, path=onnx_path)

                # Optimización de hilos para CPU en ONNX Runtime
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1  # Forzar 1 hilo para evitar sobrecargas, se puede ampliar si vCPU > 1
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
                logger.info("onnx_model_loaded", label=label, path=onnx_path)
                return session
            except Exception as exc:
                logger.error("onnx_load_error_falling_back_to_pytorch", label=label, error=str(exc))

        # 2. Si no existe ONNX, verificar si al menos existe el .pth para convertirlo
        if not os.path.exists(weights_path):
            logger.warning(
                "model_weights_not_found",
                label=label,
                weights_path=weights_path,
            )
            return None

        # 3. Cargar con PyTorch y exportar a ONNX al vuelo
        try:
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model
            from fastreid.utils.checkpoint import Checkpointer

            logger.info("loading_pytorch_model_for_onnx_export", label=label, path=weights_path)
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.BACKBONE.PRETRAIN = False
            cfg.MODEL.DEVICE = "cpu"  # Exportar siempre en CPU
            cfg.CUDNN_BENCHMARK = False
            cfg.freeze()

            model = build_model(cfg)
            Checkpointer(model).load(weights_path)
            model.eval()

            # Determinar dimensiones correctas según el modelo
            from app.services.preprocessing import PERSON_HEIGHT, PERSON_WIDTH, VEHICLE_HEIGHT, VEHICLE_WIDTH
            height = PERSON_HEIGHT if label == "person" else VEHICLE_HEIGHT
            width = PERSON_WIDTH if label == "person" else VEHICLE_WIDTH

            # Exportar a ONNX
            success = self._export_to_onnx(model, onnx_path, height, width)

            # Liberar modelo PyTorch de la memoria RAM inmediatamente
            import gc
            del model
            gc.collect()

            if success and os.path.exists(onnx_path):
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
                logger.info("onnx_model_loaded_after_export", label=label, path=onnx_path)
                return session

            raise RuntimeError("La exportación a ONNX falló o no se generó el archivo")

        except ImportError:
            logger.warning(
                "fastreid_not_installed_for_export",
                label=label,
                hint="Se requiere FastReID instalado para exportar el .pth original a .onnx: pip install -e /opt/fast-reid",
            )
            return None
        except Exception as exc:
            logger.error("pytorch_load_or_export_error", label=label, error=str(exc))
            return None

    def _export_to_onnx(self, model: torch.nn.Module, onnx_path: str, height: int, width: int) -> bool:
        """
        Exporta de forma limpia un modelo PyTorch de FastReID a ONNX.
        Usa un wrapper para simplificar la salida eliminando diccionarios.
        """
        try:
            logger.info("exporting_to_onnx_started", path=onnx_path, height=height, width=width)

            # Wrapper para forzar que el grafo ONNX solo retorne el tensor plano de features
            class FastReIDONNXWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    output = self.m(x)
                    if isinstance(output, dict):
                        return output["features"]
                    return output

            wrapper = FastReIDONNXWrapper(model)
            wrapper.eval()

            # Tensor dummy para trazar la estructura
            dummy_input = torch.zeros(1, 3, height, width, device=torch.device("cpu"))

            # Asegurar directorio de destino
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            # Exportación robusta con batch size dinámico
            torch.onnx.export(
                wrapper,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            logger.info("exporting_to_onnx_completed", path=onnx_path)
            return True
        except Exception as exc:
            logger.error("exporting_to_onnx_failed", path=onnx_path, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Estado del servicio
    # ------------------------------------------------------------------

    @property
    def person_model_loaded(self) -> bool:
        if self._person_model is not None:
            return True
        return os.path.exists(settings.model_path_person) or os.path.exists(settings.model_path_person.replace(".pth", ".onnx"))

    @property
    def vehicle_model_loaded(self) -> bool:
        if self._vehicle_model is not None:
            return True
        return os.path.exists(settings.model_path_vehicle) or os.path.exists(settings.model_path_vehicle.replace(".pth", ".onnx"))

    @property
    def gpu_available(self) -> bool:
        return torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Inferencia pública (crops directos - tradicional)
    # ------------------------------------------------------------------

    def embed_person(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding para un crop de persona."""
        return self._run_inference(
            image=image,
            model=self.person_model,
            transform=PERSON_TRANSFORM,
            label="person",
        )

    def embed_vehicle(self, image: Image.Image) -> tuple[list[float], int]:
        """Genera embedding para un crop de vehículo."""
        return self._run_inference(
            image=image,
            model=self.vehicle_model,
            transform=VEHICLE_TRANSFORM,
            label="vehicle",
        )

    # ------------------------------------------------------------------
    # Detección + crop temporal + embedding (imagen completa - nuevo endpoint)
    # ------------------------------------------------------------------

    def embed_persons_from_image(self, image: Image.Image) -> tuple[list[DetectedEmbedding], int]:
        """
        Analiza una imagen completa, detecta personas, recorta cada una
        (crop temporal) y genera su embedding en un único batch optimizado.
        """
        t_total = time.perf_counter()
        detector = DetectionService.get_instance()

        # Etapa 1: Detección
        t_det = time.perf_counter()
        detections = detector.detect_persons(image)
        det_ms = int((time.perf_counter() - t_det) * 1000)
        logger.info("detection_stage", label="person", count=len(detections), detection_ms=det_ms)

        # Etapa 2: Crop + Embedding
        results, embed_ms = self._detect_crop_embed(
            image=image,
            detections=detections,
            model=self.person_model,
            transform=PERSON_TRANSFORM,
            label="person",
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

    def _detect_crop_embed(
        self,
        image: Image.Image,
        detections: list[Detection],
        model: Optional[object],
        transform,
        label: str,
    ) -> tuple[list[DetectedEmbedding], int]:
        if model is None:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        t_start = time.perf_counter()
        rgb = image.convert("RGB") if image.mode != "RGB" else image

        if not detections:
            return [], 0

        # Batch processing: recortar todos los crops
        t_crop = time.perf_counter()
        crops = [rgb.crop((det.x1, det.y1, det.x2, det.y2)) for det in detections]
        crop_ms = int((time.perf_counter() - t_crop) * 1000)

        # Preprocesar batch completo en un tensor
        tensors = []
        for crop in crops:
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            tensors.append(transform(crop))
        
        # Apilar en batch (N, C, H, W)
        batch_tensor = torch.stack(tensors)
        preprocess_ms = int((time.perf_counter() - t_crop) * 1000)

        # Inferencia batch única ultrarrápida
        t_inference = time.perf_counter()
        
        import onnxruntime as ort
        if isinstance(model, ort.InferenceSession):
            input_data = batch_tensor.numpy()
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            
            ort_outputs = model.run([output_name], {input_name: input_data})
            features = torch.from_numpy(ort_outputs[0])
        else:
            batch_tensor = batch_tensor.to(self._device)
            with torch.inference_mode():
                output = model(batch_tensor)
            features = output["features"] if isinstance(output, dict) else output
        
        features = F.normalize(features, p=2, dim=1)
        inference_ms = int((time.perf_counter() - t_inference) * 1000)

        # Convertir a lista de embeddings
        embeddings_list = features.cpu().tolist()

        # Construir resultados
        results: list[DetectedEmbedding] = []
        for det, embedding in zip(detections, embeddings_list):
            results.append(
                DetectedEmbedding(
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    confidence=det.confidence,
                    embedding=embedding,
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

    # ------------------------------------------------------------------
    # Lógica de inferencia interna (individual)
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image: Image.Image,
        model: Optional[object],
        transform,
        label: str,
    ) -> tuple[list[float], int]:
        if model is None:
            raise RuntimeError(f"Modelo {label} no disponible — pesos no cargados")

        t_start = time.perf_counter()

        import onnxruntime as ort
        if isinstance(model, ort.InferenceSession):
            # ONNX Runtime - no requiere device GPU/CPU, preprocesar en CPU
            tensor = preprocess_image(image, transform, torch.device("cpu"))
            input_data = tensor.numpy()
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            
            ort_outputs = model.run([output_name], {input_name: input_data})
            features = torch.from_numpy(ort_outputs[0])
        else:
            # PyTorch
            tensor = preprocess_image(image, transform, self._device)
            with torch.inference_mode():
                output = model(tensor)
            features = output["features"] if isinstance(output, dict) else output

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
