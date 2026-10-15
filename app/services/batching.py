"""
Módulo de micro-batching dinámico para inferencia ReID en CPU/GPU.

Agrupa múltiples crops recibidos de forma concurrente dentro de una ventana
corta (`batch_window_ms`, ej. 15 ms) y los ejecuta en un único forward pass
de hasta `max_batch_size` imágenes, multiplicando el throughput en escenarios
de alto volumen (1.000 cámaras con ByteTrack).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from PIL import Image


ObjectType = Literal["person", "vehicle"]


@dataclass
class BatchRequest:
    """Representa una solicitud pendiente de embedding dentro de un micro-batch."""

    image: "Image.Image"
    object_type: ObjectType
    future: asyncio.Future["BatchResult"]


@dataclass
class BatchResult:
    """Resultado de un embedding generado en batch."""

    embedding: list[float] = field(default_factory=list)
    processing_ms: int = 0


class BatchProcessor:
    """
    Micro-batcher asíncrono por tipo de objeto (`person` y `vehicle`).
    Agrupa solicitudes concurrentes durante `window_ms` milisegundos o hasta
    alcanzar `max_batch_size` y ejecuta una sola inferencia vectorizada.
    """

    def __init__(
        self,
        batch_infer_fn: Callable[[list["Image.Image"], ObjectType], tuple[list[list[float]], int]],
        max_batch_size: int = 12,
        window_ms: int = 15,
    ) -> None:
        self._batch_infer_fn = batch_infer_fn
        self.max_batch_size = max(1, max_batch_size)
        self.window_sec = max(0.001, window_ms / 1000.0)
        self._queues: dict[ObjectType, list[BatchRequest]] = {
            "person": [],
            "vehicle": [],
        }
        self._locks: dict[ObjectType, asyncio.Lock] = {
            "person": asyncio.Lock(),
            "vehicle": asyncio.Lock(),
        }
        self._FlushTasks: dict[ObjectType, asyncio.Task | None] = {
            "person": None,
            "vehicle": None,
        }

    async def submit(self, image: "Image.Image", object_type: ObjectType) -> tuple[list[float], int]:
        """Encola una imagen y espera el resultado del micro-batch."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[BatchResult] = loop.create_future()
        req = BatchRequest(image=image, object_type=object_type, future=fut)

        should_flush_now = False
        async with self._locks[object_type]:
            queue = self._queues[object_type]
            queue.append(req)
            if len(queue) >= self.max_batch_size:
                should_flush_now = True
            elif self._FlushTasks[object_type] is None or self._FlushTasks[object_type].done():
                self._FlushTasks[object_type] = loop.create_task(
                    self._delayed_flush(object_type)
                )

        if should_flush_now:
            await self._flush(object_type)

        result = await fut
        return result.embedding, result.processing_ms

    async def _delayed_flush(self, object_type: ObjectType) -> None:
        await asyncio.sleep(self.window_sec)
        await self._flush(object_type)

    async def _flush(self, object_type: ObjectType) -> None:
        async with self._locks[object_type]:
            batch = self._queues[object_type][: self.max_batch_size]
            self._queues[object_type] = self._queues[object_type][self.max_batch_size :]

        if not batch:
            return

        images = [item.image for item in batch]
        t0 = time.perf_counter()
        try:
            embeddings, _ = await asyncio.to_thread(
                self._batch_infer_fn,
                images,
                object_type,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            for item, emb in zip(batch, embeddings):
                if not item.future.done():
                    item.future.set_result(BatchResult(embedding=emb, processing_ms=elapsed_ms))
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

