import os
import re
from pathlib import Path


def update_imports() -> None:
    # Definimos el nuevo prefijo del paquete
    package_prefix = "credit_risk_analysis"
    # Módulos que ahora viven dentro del paquete [cite: 2026-02-27]
    modules = [
        "api", "config", "dashboard", "features",
        "governance", "modeling", "training", "utils"
        ]

    # Expresión regular para encontrar imports que empiecen por esos módulos
    pattern = re.compile(rf"from ({'|'.join(modules)})")
    replacement = rf"from {package_prefix}.\1"

    # Recorrer todos los archivos .py en la nueva estructura [cite: 2026-02-27]
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Aplicar el cambio de prefijo
                new_content = pattern.sub(replacement, content)

                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"✅ Imports actualizados en: {file_path}")


if __name__ == "__main__":
    update_imports()
