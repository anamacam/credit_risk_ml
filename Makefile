# Variables de entorno y rutas
PYTHON = python
PACKAGE_NAME = credit_risk_analysis
SRC_DIR = src
TEST_DIR = tests
DASHBOARD_PORT = 8501
MLFLOW_PORT = 5000
MIN_COVERAGE = 80

.PHONY: help install format lint test train promote dashboard ui up down logs pipeline clean

help:
	@echo "🛠️ MLOps Pipeline - Comandos disponibles [cite: 2026-02-27]:"
	@echo "  make install   : Instala dependencias y paquete"
	@echo "  make format    : Ejecuta Black e Isort"
	@echo "  make lint      : Análisis estático (Flake8 + Mypy --strict [cite: 2026-03-04])"
	@echo "  make test      : Ejecuta tests unitarios y verifica cobertura"
	@echo "  make train     : Lanza entrenamiento vía MLproject"
	@echo "  make promote   : Promociona el mejor modelo a PRODUCTION"
	@echo "  make up        : Levanta TODO el stack con Docker (API + UI + MLflow)"
	@echo "  make pipeline  : Ciclo Completo (Lint + Test + Train + Promote)"

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -e .
	@echo "✅ Entorno preparado."

format:
	isort $(SRC_DIR) $(TEST_DIR)
	black $(SRC_DIR) $(TEST_DIR)
	@echo "✨ Código formateado."

lint:
	@echo "🔍 Ejecutando análisis estático..."
	flake8 $(SRC_DIR) --count --show-source --statistics
	mypy $(SRC_DIR) --strict
	@echo "🛡️ Validaciones de tipos completadas [cite: 2026-03-04]."

test:
	@echo "🧪 Ejecutando tests con Pytest..."
	pytest --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=term-missing --cov-fail-under=$(MIN_COVERAGE)
	@echo "✅ Tests pasados y cobertura suficiente."

train:
	mlflow run . -e main
	@echo "📈 Entrenamiento finalizado en MLflow."

promote:
	$(PYTHON) $(SRC_DIR)/training/promotion_service.py
	@echo "🚀 Mejor modelo promocionado."

up:
	@echo "🐳 Levantando infraestructura Docker..."
	docker-compose up -d --build

down:
	docker-compose down

pipeline: lint test train promote
	@echo "🏁 Ciclo de MLOps completado con éxito sin tareas manuales [cite: 2026-02-27]."

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov mlruns
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Sweep completed."