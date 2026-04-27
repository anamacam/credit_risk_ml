import os
from pathlib import Path

RUN_ID = "f6ec06d592954560946dd1365a01d6dc"
FOUND = False

print(f"🔍 Buscando el run {RUN_ID} en el directorio actual...")

# Caminamos por todo el árbol de directorios
for root, _, files in os.walk("."):
    if RUN_ID in root and "meta.yaml" in files:
        full_path = Path(root).resolve()
        # Fix F541/S3457: Quitamos la 'f' del string sin placeholders
        print("\n🎯 ¡ENCONTRADO!")
        print(f"Ruta absoluta: {full_path}")
        print(f"Ruta relativa: {os.path.relpath(full_path)}")
        FOUND = True
        break

if not FOUND:
    # Fix E501: Cortamos el mensaje para no exceder los 79 caracteres
    msg = f"\n❌ No se encontró nada. Carpetas: {os.listdir('.')}"
    print(msg)
