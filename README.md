# FastReID Embedding Service

Microservicio productivo de generación de embeddings visuales para re-identificación de personas y vehículos. Parte del ecosistema **OPTRAX**.

## Responsabilidad

Este servicio **SOLO** genera embeddings. No guarda en base de datos, no conoce cámaras, eventos ni usuarios.

```
Perceiver -> Detection + ByteTrack -> Crop relevante
-> FastReID Service (este servicio) -> Embedding
-> Core guarda en pgvector
```

---

## Stack

| Componente | Tecnología |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Modelo | FastReID (ResNet50-IBN) |
| Inferencia | PyTorch (CPU/CUDA) |
| Seguridad | API Key + slowapi rate limiting |
| Contenedor | Docker + Docker Compose |
| Reverse proxy | NGINX (TLS, rate limit, security headers) |

---

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/health` | No | Estado del servicio y modelos |
| `POST` | `/embed/person` | `x-api-key` | Embedding de persona (2048-dim) |
| `POST` | `/embed/vehicle` | `x-api-key` | Embedding de vehículo (2048-dim) |

### POST /embed/person

```bash
curl -X POST https://tu-dominio/embed/person \
  -H "x-api-key: TU_API_KEY" \
  -F "image=@crop_persona.jpg"
```

Respuesta:
```json
{
  "model": "fastreid_person",
  "dimension": 2048,
  "embedding": [0.021, -0.134, ...],
  "processing_ms": 38
}
```

---

## Setup local

### 1. Variables de entorno

```bash
cp .env.example .env
# Editar .env con API_KEY y rutas de pesos
```

Generar API Key:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### 2. Pesos del modelo

Descargar pesos preentrenados de FastReID:
- [Market-1501 BagTricks R50](https://github.com/JDAI-CV/fast-reid/releases) → `weights/person_model.pth`
- [VeRi BagTricks R50](https://github.com/JDAI-CV/fast-reid/releases) → `weights/vehicle_model.pth`

### 3. Levantar con Docker Compose

```bash
# CPU-only
docker compose up --build -d

# Con GPU (descomentar bloque deploy en docker-compose.yml y ENABLE_GPU=true en .env)
docker compose up --build -d
```

### 4. Verificar

```bash
curl http://localhost/health
```

---

## TLS / HTTPS

Colocar los certificados en `nginx/certs/`:
```
nginx/certs/fullchain.pem
nginx/certs/privkey.pem
```

Para Let's Encrypt en DigitalOcean:
```bash
apt install certbot
certbot certonly --standalone -d tu-dominio.com
cp /etc/letsencrypt/live/tu-dominio.com/fullchain.pem nginx/certs/
cp /etc/letsencrypt/live/tu-dominio.com/privkey.pem nginx/certs/
```

---

## GPU (DigitalOcean GPU Droplet)

```bash
# En el droplet
apt install nvidia-container-toolkit
systemctl restart docker

# En .env
ENABLE_GPU=true

# En docker-compose.yml: descomentar bloque deploy.resources
docker compose up --build -d
```

---

## Integración con Core (pgvector)

Tabla sugerida en PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- halfvec(2048): usa float16 (2 bytes/dim) para soportar HNSW con >2000 dims.
-- El servicio genera embeddings de 2048 dims nativos de ResNet50.
CREATE TABLE embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id    BIGINT NOT NULL,
    camera_id   INTEGER,
    track_id    TEXT,
    object_type TEXT,
    image_path  TEXT,
    embedding   halfvec(2048),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON embeddings 
USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128);
```

Búsqueda forense:
```sql
SELECT id, event_id, camera_id, track_id, object_type,
       embedding <=> '[...]' AS distance
FROM embeddings
ORDER BY distance
LIMIT 20;
```

---

## Seguridad

- API Key validada con comparación en tiempo constante (no vulnerable a timing attacks)
- Rate limiting por IP a nivel NGINX y FastAPI (slowapi)
- MIME type validado por magic bytes (no por Content-Type del cliente)
- Tamaño máximo de upload: 5MB
- Timeout por request: 30s
- Stack traces nunca expuestos en producción
- Uvicorn no expuesto públicamente (solo accesible desde NGINX interno)
- Usuario no-root en contenedor

---

## Observabilidad

Logs en formato JSON estructurado a stdout:

```json
{"timestamp": "2024-01-15T10:30:00.123Z", "level": "INFO", "logger": "app.api.routes.embed", "message": "embed_person_ok", "dimension": 2048, "processing_ms": 42}
```

Preparado para integrarse con Prometheus + Grafana (métricas custom próxima iteración).
