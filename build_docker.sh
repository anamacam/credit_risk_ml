#!/bin/bash

echo "🚀 Iniciando proceso de build..."

# Paso 1: Preparar artefactos
echo "📦 Preparando artefactos..."
./prepare_artifacts.sh

if [ $? -ne 0 ]; then
    echo "❌ Falló la preparación de artefactos"
    exit 1
fi

# Paso 2: Construir imagen Docker
echo "🏗️  Construyendo imagen Docker..."
docker build -t credit-risk-api:latest .

if [ $? -ne 0 ]; then
    echo "❌ Falló la construcción de la imagen"
    exit 1
fi

echo "✅ Imagen construida exitosamente: credit-risk-api:latest"

# Paso 3: Opcional - Mostrar cómo ejecutar
echo ""
echo "Para ejecutar el contenedor:"
echo "docker run -p 8000:8000 credit-risk-api:latest"