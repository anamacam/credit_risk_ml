import pandas as pd
from sklearn.datasets import make_classification
from pathlib import Path


def create_data() -> None:
    """Genera datos sintéticos para validar el pipeline."""
    # Mypy: Ahora el retorno está tipado como None
    X, y = make_classification(
        n_samples=1000, n_features=4, n_informative=2, random_state=42
    )

    cols = ["age", "income", "loan_amount", "credit_score"]
    df = pd.DataFrame(X, columns=cols)
    df["default"] = y

    output_path = Path("data/raw/german_credit_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Datos dummy creados en {output_path}")


if __name__ == "__main__":
    create_data()
