import logging
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def _exporter_config() -> tuple[str, bool]:
    configured_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    parsed = urlparse(configured_endpoint)

    if parsed.scheme in {"http", "https"}:
        endpoint = parsed.netloc or parsed.path
        return endpoint, parsed.scheme == "http"

    return configured_endpoint, True


def configure_tracing(app: FastAPI) -> TracerProvider:
    endpoint, insecure = _exporter_config()
    service_name = os.getenv("OTEL_SERVICE_NAME", "fastreid")
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=insecure),
        )
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    logger.info("tracing_configured", extra={"service_name": service_name, "endpoint": endpoint})
    return provider


def shutdown_tracing(provider: TracerProvider) -> None:
    FastAPIInstrumentor.uninstrument()
    provider.shutdown()
