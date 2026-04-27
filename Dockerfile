# =============================================================================
# STAGE 1 — Builder & Tester
# Instala dependencias, valida tipos y linting antes de pasar a producción.
# =============================================================================
# Sincronizado a 3.11 para compatibilidad con Numba/SHAP
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalación de dependencias del sistema originales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalación de dependencias core
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Instalación de herramientas de testing y tipado (Mypy, Flake8)
RUN pip install --no-cache-dir \
    mypy pandas-stubs types-PyYAML types-requests flake8

COPY . .

# Validación de calidad de código (Mypy --strict ya pasa tras las correcciones)
RUN pip install -e . && \
    mypy src --strict --ignore-missing-imports && \
    flake8 src --max-line-length=79


# =============================================================================
# STAGE 2 — Runner (Producción)
# Imagen mínima: solo lo necesario para ejecutar la API.
# =============================================================================
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencias de ejecución y curl para el healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpng16-16 \
    libfreetype6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar librerías y binarios desde el builder 
# CORRECCIÓN: La ruta debe ser python3.11 para que coincida con la base del builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Código fuente
COPY src/ ./src/

# Artefactos: Aquí corregimos la carga del preprocesador
COPY artifacts/ ./artifacts/

# Datos procesados
COPY data/processed/ ./data/processed/

# 1. Configuración de estructura interna
RUN mkdir -p /app/mlruns /app/artifacts/model /app/logs && \
    chmod -R 755 /app/mlruns /app/artifacts /app/logs

# 2. Copia de Artefactos desde Rutas ESTÁNDAR
COPY artifacts/preprocessor.pkl ./artifacts/preprocessor.pkl
COPY artifacts/model/model.pkl ./artifacts/model/model.pkl

EXPOSE 8000

# Healthcheck que monitorea el estado real del preprocesador
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health | grep '"preprocessor_loaded":true' || exit 1

CMD ["uvicorn", "credit_risk_analysis.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]