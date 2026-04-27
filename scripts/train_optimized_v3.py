from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constantes de Negocio
SAVING_ACC = 'saving_accounts'
CHECKING_ACC = 'checking_account'
CREDIT_AMT = 'credit_amount'


def calculate_ks_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calcula la estadística Kolmogorov-Smirnov."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(max(tpr - fpr))


def find_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """Encuentra el threshold que maximiza el F1-Score."""
    _, _, thresholds = roc_curve(y_true, y_pred_proba)
    best_threshold, best_f1 = 0.5, 0.0
    for thresh in thresholds:
        y_p = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_p)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    return float(best_threshold)


def main() -> None:
    """Flujo v3: Optimización con búsqueda de hiperparámetros."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit_risk_production")

    # 1. CARGA Y FEATURE ENGINEERING
    df = pd.read_csv("data/raw/german_credit_data.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    df[SAVING_ACC] = df[SAVING_ACC].fillna('none')
    df[CHECKING_ACC] = df[CHECKING_ACC].fillna('none')

    # Ingeniería de Variables
    df['credit_per_month'] = df[CREDIT_AMT] / df['duration'].replace(0, 1)
    df['age_duration_ratio'] = df['age'] / df['duration'].replace(0, 1)

    # Selección de features
    num_feats = [
        'age', CREDIT_AMT, 'duration',
        'credit_per_month', 'age_duration_ratio'
    ]
    cat_feats = ['sex', 'housing', SAVING_ACC, CHECKING_ACC, 'purpose']

    x_input = df[num_feats + cat_feats]
    y_target = (df['risk'] == 'bad').astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_input, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    # 2. PREPROCESAMIENTO
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False
        ), cat_feats)
    ])

    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc = preprocessor.transform(x_test)

    # 3. BÚSQUEDA DE HIPERPARÁMETROS
    param_dist = {
        'n_estimators': [300, 500],
        'max_depth': [10, 15, None],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt']
    }

    base_rf = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        base_rf, param_distributions=param_dist, n_iter=5,
        cv=3, scoring='f1', random_state=42, n_jobs=-1
    )
    search.fit(x_train_proc, y_train)
    best_model = search.best_estimator_

    # 4. EVALUACIÓN Y THRESHOLD
    probs = best_model.predict_proba(x_test_proc)[:, 1]
    opt_thresh = find_optimal_threshold(y_test, probs)
    y_pred_opt = (probs >= opt_thresh).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "f1_score": float(f1_score(y_test, y_pred_opt)),
        "ks_score": calculate_ks_score(y_test, probs),
        "recall": float(recall_score(y_test, y_pred_opt))
    }

    # 5. LOGGING Y PERSISTENCIA
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp_{now_str}_v3_optimized"
    exp_folder = Path(f"experiments/{exp_name}")
    exp_folder.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"RF_V3_Final_{now_str}"):
        mlflow.log_params(best_model.get_params())
        mlflow.log_param("best_threshold", opt_thresh)
        mlflow.log_metrics(metrics)

        joblib.dump(best_model, exp_folder / "model.pkl")
        joblib.dump(preprocessor, exp_folder / "preprocessor.pkl")

        mlflow.log_artifacts(str(exp_folder), artifact_path="model_assets")
        mlflow.sklearn.log_model(best_model, "model")

    print(f"\n✅ Entrenamiento exitoso: {exp_name}")
    print(f"🎯 Threshold: {opt_thresh:.4f} | AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
