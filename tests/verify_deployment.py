import sys
import json
from credit_risk_analysis.dashboard.services.model_service import ModelService
from credit_risk_analysis.dashboard.services.shap_service import ShapService
from credit_risk_analysis.utils.logging_config import setup_logger

logger = setup_logger(name="verify_deployment")


def run_e2e_test() -> bool:
    """Ejecuta una prueba de extremo a extremo de los servicios."""
    logger.info("🚀 Iniciando validación de servicios...")

    try:
        # 1. Validar ModelService
        service = ModelService()
        if not service.model:
            logger.error("❌ ModelService: No se pudo cargar el modelo.")
            return False
        logger.info("✅ ModelService: Modelo cargado.")

        # 2. Validar Predicción Sintética
        test_data = {
            "Age": 33,
            "Sex": "male",
            "Job": 2,
            "Housing": "own",
            "Saving accounts": "little",
            "Checking account": "quite rich",
            "Credit amount": 1500,
            "Duration": 24,
            "Purpose": "car"
        }

        prob = service.predict(test_data)
        msg_prob = f"✅ ModelService: Predicción exitosa. Prob: {prob:.4f}"
        logger.info(msg_prob)

        # 3. Validar ShapService
        shap_s = ShapService(model_service=service)
        explanation = shap_s.explain(test_data)

        if len(explanation["values"]) > 0:
            logger.info("✅ ShapService: Explicación generada.")
        else:
            logger.warning("⚠️ ShapService: Explicación vacía.")
            return False

        # 4. Validar tipos de datos para JSON
        try:
            json.dumps(explanation)
            logger.info("✅ Serialización: Datos SHAP compatibles con JSON.")
        except TypeError as e:
            logger.error(f"❌ Serialización: Error de tipos en SHAP: {e}")
            return False

        logger.info("🎊 ¡SISTEMA LISTO PARA PRODUCCIÓN!")
        return True

    except Exception as e:
        logger.critical(f"💥 Fallo crítico en validación: {e}")
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    if not success:
        sys.exit(1)
