from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk_analysis.utils.logging_config import setup_logger

logger = setup_logger(name="feature_engineering")


# [cite: 2026-03-04] Movemos el ignore a la definición de la clase
class CreditFeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore
    """Transformador profesional de ingeniería de características."""

    def __init__(self, ordinal_mapping: Dict[str, int] | None = None) -> None:
        super().__init__()
        self.ordinal_map = ordinal_mapping or {
            "unknown": 0, "little": 1, "moderate": 2,
            "quite rich": 3, "rich": 4,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series[Any] | None = None
    ) -> CreditFeatureEngineer:
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Ejecutando ingeniería de características...")
        df = X.copy()

        df["credit_per_month"] = df["credit_amount"] / df["duration"]
        df["age_duration_ratio"] = df["age"] / df["duration"]
        df["credit_amount"] = np.log1p(df["credit_amount"])

        for col in ["saving_accounts", "checking_account"]:
            if col in df.columns:
                df[col] = df[col].map(self.ordinal_map).fillna(0)

        df = pd.get_dummies(
            df,
            columns=["sex", "housing", "purpose"],
            drop_first=True
        )

        return df
