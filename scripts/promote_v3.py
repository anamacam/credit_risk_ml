import os
from mlflow.tracking import MlflowClient


def promote():
    client = MlflowClient()
    os.makedirs("artifacts", exist_ok=True)

    # Buscamos el mejor modelo del experimento 7
    runs = client.search_runs(
        experiment_ids=["7"],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )

    if runs:
        run = runs[0]
        run_id = run.info.run_id
        auc = run.data.metrics.get("roc_auc", 0)
        # Usamos el nombre exacto que grabó tu script V3
        thresh = run.data.params.get("best_threshold", "N/A")

        print("-" * 30)
        print(f"🆔 Run ID: {run_id}")
        print(f"📊 ROC AUC: {auc:.4f}")
        print(f"🎯 Threshold: {thresh}")

        # Guardar el ID para que la API lo reconozca como el activo
        with open("artifacts/ACTIVE_MODEL.txt", "w") as f:
            f.write(run_id)

        print("\n✅ ACTIVE_MODEL.txt actualizado con éxito.")
        print("-" * 30)
    else:
        print("❌ No se encontró ningún run en el experimento 7.")


if __name__ == "__main__":
    promote()
