# =============================================================================
# Stage 1: builder — instala dependencias y FastReID desde fuente
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clonar FastReID e instalar sus dependencias primero
# fast-reid no tiene setup.py/pyproject.toml: se importa via PYTHONPATH
RUN git clone --depth=1 https://github.com/JDAI-CV/fast-reid.git /opt/fast-reid \
    && sed 's/faiss-gpu/faiss-cpu/g' /opt/fast-reid/docs/requirements.txt \
       | pip install --no-cache-dir -r /dev/stdin

# Instalar dependencias propias (sobreescriben versiones conflictivas de fast-reid)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: runtime — imagen final mínima para producción
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/opt/fast-reid \
    APP_ENV=production

# Dependencias de runtime para OpenCV/PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados del builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/fast-reid /opt/fast-reid

# Crear usuario no-root
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copiar código fuente
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup configs/ ./configs/

# Crear directorios con permisos correctos
RUN mkdir -p /app/weights /app/logs \
    && chown -R appuser:appgroup /app/weights /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--no-access-log"]
