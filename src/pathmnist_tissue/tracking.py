"""Registro opcional de corridas en un servidor MLflow."""

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def tracking_run(
    uri: str | None, experiment: str, run_dir: Path, config: dict
) -> Generator[object | None, None, None]:
    """Abre una corrida MLflow si se configuró un servidor.

    Args:
        uri: Dirección del servidor; None desactiva el registro remoto.
        experiment: Nombre del experimento MLflow.
        run_dir: Carpeta local de la corrida.
        config: Configuración que recibe el identificador remoto.

    Yields:
        Módulo mlflow o None si no se configuró el servidor.
    """
    if uri is None:
        yield None
        return

    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_dir.name) as run:
        # Guardamos el ID para añadir después la validación o el test a esta misma corrida.
        config["mlflow_uri"] = uri
        config["mlflow_run_id"] = run.info.run_id
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        params = {
            key: value for key, value in config.items()
            if key not in ("mlflow_uri", "mlflow_run_id")
        }
        mlflow.log_params(params)
        # Al salir del bloque que usa tracking_run, MLflow cierra la corrida.
        yield mlflow


def log_test_to_mlflow(run_dir: Path, config: dict, uri: str | None = None) -> None:
    """Añade la evaluación final a la corrida MLflow original, si existe.

    Args:
        run_dir: Carpeta local que contiene la evaluación.
        config: Configuración guardada de la corrida.
        uri: Dirección alternativa del mismo servidor MLflow.
    """
    run_id = config.get("mlflow_run_id")
    if not run_id:
        return

    import mlflow

    mlflow.set_tracking_uri(uri or config["mlflow_uri"])
    report = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    with mlflow.start_run(run_id=run_id):
        _log_test_files(mlflow, run_dir, report)


def log_validation_to_mlflow(run_dir: Path, config: dict, report: dict) -> None:
    """Añade el análisis de validación de una corrida anterior a MLflow.

    Args:
        run_dir: Carpeta local de la corrida.
        config: Configuración con el identificador MLflow.
        report: Métricas de validación por clase.
    """
    run_id = config.get("mlflow_run_id")
    if not run_id:
        return

    import mlflow

    mlflow.set_tracking_uri(config["mlflow_uri"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_macro_f1", report["macro avg"]["f1-score"])
        for name in (
            "validation_metrics.json", "validation_confusion_matrix.csv",
            "validation_predictions.csv",
        ):
            mlflow.log_artifact(str(run_dir / name), artifact_path="validation")


def sync_saved_run(run_dir: Path, uri: str, experiment: str) -> None:
    """Importa a MLflow una corrida local completa.

    Args:
        run_dir: Carpeta de una corrida terminada.
        uri: Dirección del servidor MLflow.
        experiment: Nombre del experimento remoto.
    """
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("mlflow_run_id"):
        raise RuntimeError("Esta corrida ya está registrada en MLflow.")
    if not (run_dir / "duration.json").exists():
        raise RuntimeError("Solo se pueden importar corridas completas.")

    with tracking_run(uri, experiment, run_dir, config) as mlflow:
        _log_history(mlflow, run_dir / "history.csv")
        best = json.loads((run_dir / "best_validation.json").read_text(encoding="utf-8"))
        mlflow.log_metric("best_val_loss", best["loss"])
        resources = json.loads((run_dir / "duration.json").read_text(encoding="utf-8"))
        mlflow.log_metrics(resources)

        validation_file = run_dir / "validation_metrics.json"
        if validation_file.exists():
            validation = json.loads(validation_file.read_text(encoding="utf-8"))
            mlflow.log_metric("val_macro_f1", validation["macro avg"]["f1-score"])

        # La carpeta local sigue siendo la copia completa de cada experimento.
        mlflow.log_artifacts(str(run_dir), artifact_path="run_files")
        test_file = run_dir / "test_metrics.json"
        if test_file.exists():
            _log_test_files(mlflow, run_dir, json.loads(test_file.read_text(encoding="utf-8")))


def _log_history(mlflow, history_file: Path) -> None:
    """Registra las métricas de cada época de una corrida ya guardada.

    Args:
        mlflow: Módulo MLflow con una corrida activa.
        history_file: Archivo CSV producido durante el entrenamiento.
    """
    with history_file.open(newline="") as file:
        for row in csv.DictReader(file):
            metrics = {
                name: float(row[name])
                for name in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")
            }
            mlflow.log_metrics(metrics, step=int(row["epoch"]))


def _log_test_files(mlflow, run_dir: Path, report: dict) -> None:
    """Registra métricas y archivos de test en la corrida activa.

    Args:
        mlflow: Módulo MLflow con una corrida activa.
        run_dir: Carpeta local con resultados.
        report: Métricas de test por clase.
    """
    mlflow.log_metrics({
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"],
    })
    for name in ("test_metrics.json", "confusion_matrix.csv", "test_predictions.csv"):
        mlflow.log_artifact(str(run_dir / name), artifact_path="test")
