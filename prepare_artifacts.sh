#!/bin/bash

echo "==================================="
echo "Preparando artefactos para Docker"
echo "==================================="

# Crear directorios necesarios
mkdir -p artifacts/model

# Buscar el modelo en mlruns
if [ -d "mlruns" ]; then
    echo "Buscando model.pkl en mlruns..."
    
    # Buscar archivos model.pkl en mlruns
    MODEL_FILE=$(find mlruns -name "model.pkl" -type f | head -n 1)
    
    if [ -f "$MODEL_FILE" ]; then
        echo "✅ Modelo encontrado en: $MODEL_FILE"
        cp "$MODEL_FILE" artifacts/model/model.pkl
        echo "✅ Modelo copiado a artifacts/model/model.pkl"
    else
        echo "❌ Error: No se encontró model.pkl en mlruns"
        echo "   Revisa que la ruta contenga el archivo model.pkl"
        exit 1
    fi
else
    echo "❌ Error: Directorio mlruns no encontrado"
    echo "   Asegúrate de estar en el directorio raíz del proyecto"
    exit 1
fi

# Verificar preprocesador
if [ -f "artifacts/preprocessor.pkl" ]; then
    echo "✅ Preprocesador encontrado en artifacts/preprocessor.pkl"
else
    echo "⚠️  Advertencia: No se encontró artifacts/preprocessor.pkl"
    echo "   Asegúrate de que exista en esa ubicación"
    echo "   Si no existe, el código podría fallar en tiempo de ejecución"
fi

# Mostrar estructura final
echo ""
echo "Estructura de artefactos preparada:"
ls -la artifacts/
ls -la artifacts/model/

echo ""
echo "==================================="
echo "✅ Artefactos listos para construir la imagen Docker"
echo "==================================="