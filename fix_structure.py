import shutil
from pathlib import Path


def fix_project_layout() -> None:
    # 1. Definir nombres
    base_dir = Path("src")
    package_name = "credit_risk_analysis"
    package_path = base_dir / package_name

    # 2. Crear la carpeta del paquete si no existe [cite: 2026-02-27]
    if not package_path.exists():
        package_path.mkdir(parents=True)
        print(f"✅ Carpeta creada: {package_path}")

    # 3. Lista de elementos a mover (basado en tu captura) [cite: 2026-02-27]
    to_move = [
        "api", "config", "dashboard", "features",
        "governance", "modeling", "training", "utils",
        "__init__.py", "test_stress.py", "train_runner.py"
    ]

    for item in to_move:
        source = base_dir / item
        destination = package_path / item

        if source.exists():
            # Evitamos mover la carpeta destino dentro de sí misma
            if source == package_path:
                continue

            shutil.move(str(source), str(destination))
            print(f"➡️ Movido: {item} -> {package_name}/")

    # 4. Limpiar basura técnica de instalaciones fallidas
    egg_info = base_dir / f"{package_name}.egg-info"
    if egg_info.exists():
        shutil.rmtree(egg_info)
        print(f"🧹 Limpiado: {egg_info}")


if __name__ == "__main__":
    fix_project_layout()
