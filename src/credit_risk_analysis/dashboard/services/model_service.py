from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelService")

COL_AGE = "Age"
COL_SEX = "Sex"
COL_JOB = "Job"
COL_HOUSING = "Housing"
COL_SAVING = "Saving accounts"
COL_CHECKING = "Checking account"
COL_AMOUNT = "Credit amount"
COL_DURATION = "Duration"
COL_PURPOSE = "Purpose"
COL_INST_RATIO = "inst_ratio"
COL_AGE_GROUP = "age_group"

BASE_COLUMNS: List[str] = [
    COL_AGE, COL_SEX, COL_JOB, COL_HOUSING, COL_SAVING,
    COL_CHECKING, COL_AMOUNT, COL_DURATION, COL_PURPOSE,
    COL_INST_RATIO, COL_AGE_GROUP,
]

DEFAULT_VALUES: Dict[str, Union[int, float, str]] = {
    COL_AGE: 0,
    COL_SEX: "male",
    COL_JOB: 0,
    COL_HOUSING: "own",
    COL_SAVING: "little",
    COL_CHECKING: "little",
    COL_AMOUNT: 0,
    COL_DURATION: 0,
    COL_PURPOSE: "car",
    COL_INST_RATIO: 0.0,
    COL_AGE_GROUP: "adult",
}

COLUMN_MAPPING: Dict[str, str] = {
    "age": COL_AGE,
    "sex": COL_SEX,
    "job": COL_JOB,
    "housing": COL_HOUSING,
    "saving_accounts": COL_SAVING,
    "checking_account": COL_CHECKING,
    "credit_amount": COL_AMOUNT,
    "duration": COL_DURATION,
    "purpose": COL_PURPOSE,
    "inst_ratio": COL_INST_RATIO,
    "age_group": COL_AGE_GROUP,
}


class ModelService:

    def __init__(self) -> None:
        base_path = Path(
            r"C:\projects\credit_risk_ml\experiments"
            r"\exp_20260322_155648_v3_optimized"
        )
        if Path("/app").exists():
            base_path = Path(
                "/app/experiments/exp_20260322_155648_v3_optimized"
            )
            logger.info("Docker environment detected")

        self.artifacts_dir: Path = base_path
        self.model_path: Path = base_path / "model.pkl"
        self.preprocessor_path: Path = base_path / "preprocessor.pkl"

        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = BASE_COLUMNS

        env_threshold = os.getenv("DEFAULT_THRESHOLD")
        self.threshold: float = float(env_threshold or 0.387)
        self.is_ready: bool = False

    def initialize(self) -> None:
        start_time = time.time()
        try:
            self._validate_assets()
            logger.info("Loading model and preprocessor")

            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            if self.preprocessor and hasattr(
                self.preprocessor, "get_feature_names_out"
            ):
                self.feature_names = list(
                    self.preprocessor.get_feature_names_out()
                )
                logger.info("One-Hot features detected: %d",
                            len(self.feature_names))
            else:
                self.feature_names = self._detect_features()

            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP explainer initialized")
            except Exception as exc:
                logger.warning("SHAP disabled: %s", exc)
                self.explainer = None

            self.is_ready = True
            latency = round(time.time() - start_time, 2)
            logger.info("ModelService ready in %s seconds", latency)

        except Exception as exc:
            self.is_ready = False
            logger.critical("Startup failure", exc_info=True)
            raise RuntimeError(f"Startup error: {exc}") from exc

    def _validate_assets(self) -> None:
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(self.artifacts_dir)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(self.preprocessor_path)

    def _detect_features(self) -> List[str]:
        if self.model and hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        return BASE_COLUMNS

    def _build_dataframe(self, model_input: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([model_input])
        df = df.rename(columns=COLUMN_MAPPING)
        for column in BASE_COLUMNS:
            if column not in df.columns:
                df[column] = DEFAULT_VALUES.get(column)
        return df[BASE_COLUMNS]

    def _transform(self, df: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")
        data = self.preprocessor.transform(df)
        if hasattr(data, "toarray"):
            data = data.toarray()
        return np.asarray(data)

    def _predict_probability(
        self, x: np.ndarray[Any, np.dtype[Any]]
    ) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        probs = self.model.predict_proba(x)
        return float(probs[0][1])

    def _compute_shap(
        self, x: np.ndarray[Any, np.dtype[Any]]
    ) -> Dict[str, Any]:
        if self.explainer is None:
            return {"base_value": 0.5, "shap_values": []}
        try:
            shap_values = self.explainer.shap_values(x)
            if shap_values is None:
                return {"base_value": 0.5, "shap_values": []}

            if isinstance(shap_values, list):
                shap_values = (
                    shap_values[1] if len(shap_values) > 1 else shap_values[0]
                )

            arr = np.asarray(shap_values)
            shap_list = [float(v) for v in arr[0]] if arr.ndim == 2 else [
                float(v) for v in arr]

            base = self.explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[1] if len(base) > 1 else base[0]

            return {"base_value": float(base), "shap_values": shap_list}
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)
            return {"base_value": 0.5, "shap_values": []}

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_ready:
            return {"status": "error", "message": "Service not ready"}

        try:
            start = time.time()
            df = self._build_dataframe(model_input)
            transformed = self._transform(df)

            feature_values = [float(v) for v in transformed[0].tolist()]

            probability = self._predict_probability(transformed)
            shap_data = self._compute_shap(transformed)
            shap_list = shap_data["shap_values"]

            if shap_list and len(shap_list) != len(self.feature_names):
                logger.warning("Dimension mismatch: SHAP %d vs Features %d",
                               len(shap_list), len(self.feature_names))
                shap_list = []

            decision = (
                "high_risk" if probability >= self.threshold
                else "low_risk"
            )
            latency = round(time.time() - start, 4)

            return {
                "status": "success",
                "probability": float(round(probability, 4)),
                "decision": str(decision),
                "threshold": float(self.threshold),
                "latency": float(latency),
                "base_value": float(shap_data["base_value"]),
                "shap_values": [float(v) for v in shap_list],
                "feature_values": feature_values,
                "feature_names": [str(n) for n in self.feature_names],
            }

        except Exception as exc:
            logger.error("Prediction error", exc_info=True)
            return {"status": "error", "message": str(exc)}
