# Guía de Desarrollo - Credit Risk ML 🛠️

Este documento detalla los estándares técnicos y los procesos de automatización para el desarrollo del proyecto de análisis de riesgo crediticio.

## 🏗️ Estructura del Proyecto

El proyecto utiliza un **Source Layout** para garantizar que el código de producción esté aislado de los scripts de utilidad y pruebas.

- `src/`: Núcleo del paquete. Solo código que será distribuido.
- `data/`: Almacenamiento local de artefactos de datos (ignorado por Git si son pesados).
- `tests/` o `test_package.py`: Validación de integridad.

## 🛠️ Configuración del Entorno de Desarrollo

Para mantener la consistencia y evitar tareas manuales, seguimos estos pasos:

1. **Entorno Conda:**

```powershell
   conda activate .\.venv

2. **Instalación en Modo Editable:**
Esto permite que los cambios en src/ se reflejen instantáneamente sin reinstalar el paquete:

```powershell
   pip install -e .

   📏 Estándares de Calidad y Estilo
El proyecto tiene configuraciones automatizadas para los linters. No es necesario corregir manualmente si usas un formateador como Black.

1. Linting (Flake8)
Configurado en .flake8. El límite de línea es de 88 caracteres.
Ejecutar manualmente:

```powershell
   flake8 src
2. Tipado Estático (Mypy)
Configurado en mypy.ini. Es obligatorio anotar los tipos en las funciones de src/.

Ejecutar manualmente:

```powershell
   mypy src

📦 Ciclo de Vida del Paquete (Conda)
Para evitar errores de dependencia en producción, la construcción del paquete está automatizada mediante meta.yaml.

Construcción del artefacto:
```powershell
  conda build .

Este comando generará un archivo .conda en tu carpeta de conda-bld, el cual contiene todo lo necesario para desplegar el modelo.

🧪 Pruebas de Integración
Antes de realizar un commit o integrar MLflow, se debe validar que el paquete sea "importable" y que la configuración sea correcta:

```powershell
  python test_package.py

📈 Integración con MLflow (Próximamente)
El desarrollo futuro incluirá el tracking automático de experimentos. Cada ejecución deberá estar vinculada a la versión definida en meta.yaml para asegurar la trazabilidad total entre código y modelo.

Mantenedor: AI Collaborator / 
Última actualización: 2026-02-27
