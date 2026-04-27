#!/usr/bin/env python
"""Verifica que los modelos existen en las rutas esperadas."""

from pathlib import Path

# Rutas a verificar
paths = [
    "experiments/exp_20260322_155648_v3_optimized/model.pkl",
    "experiments/exp_20260322_155648_v3_optimized/preprocessor.pkl",
]

print("🔍 Verificando modelos...")
print("-" * 50)

for path in paths:
    p = Path(path)
    if p.exists():
        size = p.stat().st_size / (1024 * 1024)  # Convertir a MB
        print(f"✅ {path}: {size:.2f} MB")
    else:
        print(f"❌ {path}: NO ENCONTRADO")

print("-" * 50)

# Verificar si existe artifacts/active_model
active_model = Path("artifacts/active_model")
if active_model.exists():
    if active_model.is_symlink():
        target = active_model.resolve()
        print(f"🔗 artifacts/active_model es symlink a: {target}")
    else:
        print("📁 artifacts/active_model es directorio normal")
else:
    print("⚠️ artifacts/active_model no existe")
