class RiskAppError(Exception):
    """Excepción base para la aplicación de Riesgo."""
    pass


class ModelNotFoundError(RiskAppError):
    """Lanzada cuando MLflow no encuentra el modelo especificado."""
    pass


class ConfigError(RiskAppError):
    """Lanzada cuando el schema.yaml tiene errores de formato."""
    pass
