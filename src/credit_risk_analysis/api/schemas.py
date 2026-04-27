from __future__ import annotations

from typing import List, Set, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class CreditFeatures(BaseModel):
    """
    Schema Pydantic con Aliases para compatibilidad con el modelo ML.
    """
    age: int = Field(..., alias="Age", ge=18, le=100)
    sex: str = Field(..., alias="Sex")
    job: int = Field(..., alias="Job", ge=0, le=3)
    housing: str = Field(..., alias="Housing")
    saving_accounts: Optional[str] = Field(
        None, alias="Saving accounts"
    )
    checking_account: Optional[str] = Field(
        None, alias="Checking account"
    )
    credit_amount: float = Field(..., alias="Credit amount", gt=0)
    duration: int = Field(..., alias="Duration", ge=1)
    purpose: str = Field(..., alias="Purpose")
    inst_ratio: float = Field(default=0.0, alias="inst_ratio")
    age_group: str = Field(default="adult", alias="age_group")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        allowed: Set[str] = {"male", "female"}
        if v not in allowed:
            raise ValueError(f"sex debe ser uno de {allowed}")
        return v

    @field_validator("housing")
    @classmethod
    def validate_housing(cls, v: str) -> str:
        allowed: Set[str] = {"own", "rent", "free"}
        if v not in allowed:
            raise ValueError(f"housing debe ser uno de {allowed}")
        return v

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 35,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        }
    )


class BatchPredictRequest(BaseModel):
    """Request para predicción en batch limitado a 100 registros."""
    profiles: List[CreditFeatures] = Field(
        ..., min_length=1, max_length=100
    )
