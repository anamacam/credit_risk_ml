import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, recall_score
from sklearn.compose import ColumnTransformer

# Constantes para evitar errores de linting
SAVING_ACC = 'Saving accounts'
CHECKING_ACC = 'Checking account'


def calculate_ks_score(y_true, y_pred_proba):
    """Calcula la estadística Kolmogorov-Smirnov."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(max(tpr - fpr))


def find_optimal_threshold(y_true, y_pred_proba):
    """Maximiza el F1-Score para encontrar el punto de corte ideal."""
    _, _, thresholds = roc_curve(y_true, y_pred_proba)
    best_threshold, best_f1 = 0.5, 0.0
    for thresh in thresholds:
        y_p = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_p)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    return float(best_threshold)


def main():
    """Flujo principal de entrenamiento."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("credit_risk_production")

    # 1. CARGA Y FEATURE ENGINEERING
    df = pd.read_csv("data/raw/german_credit_data.csv")
    df[SAVING_ACC] = df[SAVING_ACC].fillna('none')
    df[CHECKING_ACC] = df[CHECKING_ACC].fillna('none')

    # Variables para mejorar el AUC
    df['inst_ratio'] = df['Credit amount'] / df['Duration']
    age_bins = [0, 25, 45, 65, 100]
    age_labels = ['Young', 'Adult', 'Senior', 'Elder']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    num_feats = ['Age', 'Credit amount', 'Duration', 'inst_ratio']
    cat_feats = ['Sex', 'Housing', SAVING_ACC, CHECKING_ACC, 'age_group']

    x_input = df[num_feats + cat_feats]
    y_target = (df['Risk'] == 'bad').astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_input, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    # 2. PREPROCESAMIENTO
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])

    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc = preprocessor.transform(x_test)

    # 3. ENTRENAMIENTO
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train_proc, y_train)

    # 4. EVALUACIÓN Y THRESHOLD
    probs = model.predict_proba(x_test_proc)[:, 1]
    opt_thresh = find_optimal_threshold(y_test, probs)
    y_pred_opt = (probs >= opt_thresh).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "f1_score": float(f1_score(y_test, y_pred_opt)),
        "ks_score": calculate_ks_score(y_test, probs),
        "recall": float(recall_score(y_test, y_pred_opt))
    }

    # 5. LOGGING
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder = Path(f"experiments/exp_{now_str}_enhanced")
    exp_folder.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"RF_Enhanced_{now_str}"):
        mlflow.log_params(model.get_params())
        mlflow.log_param("optimal_threshold", opt_thresh)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="credit_risk_classifier"
        )

        joblib.dump(model, exp_folder / "model.pkl")
        joblib.dump(preprocessor, exp_folder / "preprocessor.pkl")

        with open(exp_folder / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(
            str(exp_folder / "preprocessor.pkl"),
            artifact_path="model"
        )

    print("\n" + "=" * 80)
    print(f"📊 AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
    print(f"🎯 Threshold: {opt_thresh:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
