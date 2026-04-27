from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configura un logger profesional con salida a consola y archivo."""

    # Definir formato de mensajes compatible con estándares industriales
    fmt_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicidad de handlers si el logger ya fue instanciado
    if not logger.handlers:
        # Handler para Consola (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler para Archivo (opcional)
        if log_file:
            log_path = Path("logs")
            log_path.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(
                log_path / log_file,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
