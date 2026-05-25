import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formateador de logs en JSON estructurado para producción."""

    SKIP_FIELDS = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "id", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in self.SKIP_FIELDS:
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configura el sistema de logging global con salida JSON a stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Silenciar librerías ruidosas en producción
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


class StructuredLogger:
    """
    Wrapper sobre logging.Logger que acepta kwargs arbitrarios como campos
    estructurados en el JSON final.

    Uso: logger.info("evento", campo1=valor1, campo2=valor2)
    Equivale a: logger.info("evento", extra={"campo1": valor1, "campo2": valor2})
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    # Campos internos de LogRecord que no pueden sobreescribirse via extra={}
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    })

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        if self._logger.isEnabledFor(level):
            safe = {
                (f"_{k}" if k in self._RESERVED else k): v
                for k, v in kwargs.items()
            }
            self._logger.log(level, msg, extra=safe or None)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.exception(msg, extra=kwargs or None)


def get_logger(name: str) -> StructuredLogger:
    """Retorna un StructuredLogger nombrado listo para uso."""
    return StructuredLogger(name)
