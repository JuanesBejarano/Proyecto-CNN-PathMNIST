"""Ejecuta secuencialmente las comparaciones configuradas y conserva su progreso"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

from mlflow.tracking import MlflowClient
import torch


ROOT = Path(__file__).resolve().parents[1]


def experiment_plan() -> list[dict]:
    """Define las doce variantes base y cuatro comparaciones de memoria

    Returns:
        Configuraciones con lote efectivo nominal de 128 y diez épocas
    """
    plan = []
    for size, batch_norm, adaptive_pool in product((28, 64, 128), (True, False), (True, False)):  # Combina resolución, BatchNorm y pooling para construir las doce variantes base
        plan.append(dict(size=size, batch_norm=batch_norm, adaptive_pool=adaptive_pool,
                         batch_size=128, accumulation_steps=1, gradient_checkpointing=False))
    for batch_norm, checkpointing in product((False, True), (False, True)):  # Añade cuatro controles de memoria con lote físico 32 y acumulación de cuatro pasos
        plan.append(dict(size=128, batch_norm=batch_norm, adaptive_pool=True,
                         batch_size=32, accumulation_steps=4,
                         gradient_checkpointing=checkpointing))
    return plan


def completed_run(settings: dict) -> Path | None:
    """Busca una corrida terminada que coincida con el experimento

    Args:
        settings: Configuración solicitada, sin el nombre de corrida

    Returns:
        Carpeta de una corrida completa compatible, o None
    """
    defaults = dict(batch_norm=True, adaptive_pool=True, accumulation_steps=1,
                    gradient_checkpointing=False)
    for folder in sorted((ROOT / "runs").glob("*"), reverse=True):  # Busca primero las corridas más recientes según sus nombres con fecha
        if not all((folder / name).exists() for name in (
            "config.json", "best_model.pt", "duration.json", "validation_metrics.json"
        )):
            continue
        config = json.loads((folder / "config.json").read_text())
        if config.get("study") != "amp_comparison" and config.get("normalization") != "group" and config.get("model", "simplecnn") == "simplecnn" and config.get("augmentation", "none") == "none" and all(config.get(key, defaults.get(key)) == value for key, value in settings.items()):
            return folder
    return None


def main() -> None:
    """Valida GPU y MLflow, y ejecuta o muestra la cola de experimentos"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Ejecuta la cola; si se omite, solo la muestra")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    plan = [dict(**item, epochs=10, seed=42, learning_rate=1e-3, weight_decay=1e-4)
            for item in experiment_plan()]
    for item in plan:
        existing = completed_run(item)
        print(f"{'LISTA' if existing else 'PENDIENTE'}: {item}", flush=True)
    if not args.execute:  # Sin esta opción solo muestra el plan, no inicia entrenamientos
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; esta cola requiere la GPU.")
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    client.search_experiments()  # Fallar antes de entrenar si MLflow no responde
    batch_dir = ROOT / "runs" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    batch_dir.mkdir(parents=True)
    progress = []
    print(f"COLA: {batch_dir}", flush=True)
    for index, settings in enumerate(plan, start=1):
        name = (f"simplecnn_{settings['size']}_bn{int(settings['batch_norm'])}"
                f"_gap{int(settings['adaptive_pool'])}_b{settings['batch_size']}"
                f"_acc{settings['accumulation_steps']}_cp{int(settings['gradient_checkpointing'])}")
        existing = completed_run(settings)
        row = dict(name=name, settings=settings, status="running")
        progress.append(row)
        progress_file = batch_dir / "progress.json"
        progress_file.write_text(json.dumps(progress, indent=2))
        print(f"[{index}/{len(plan)}] {name}: {'reutilizando' if existing else 'entrenando'}", flush=True)
        try:
            if existing:
                config = json.loads((existing / "config.json").read_text())
                if config.get("mlflow_run_id"):
                    # Verificamos que el ID existe en el servidor que usa esta cola
                    client.get_run(config["mlflow_run_id"])
                else:
                    subprocess.run([sys.executable, "-m", "src.cli", "sync-mlflow",
                                    str(existing), "--mlflow-uri", args.mlflow_uri], cwd=ROOT, check=True)
                row.update(status="reused", run_dir=str(existing))
            else:
                command = [sys.executable, "-u", "-m", "src.cli", "train",
                           "--name", name, "--device", "cuda", "--num-workers", "2",
                           "--mlflow-uri", args.mlflow_uri]
                for key, value in settings.items():
                    option = key.replace("_", "-")
                    if isinstance(value, bool):  # Convierte los booleanos en opciones --nombre o --no-nombre del comando de entrenamiento
                        command.append(f"--{option}" if value else f"--no-{option}")
                    else:
                        command.extend([f"--{option}", str(value)])
                # El log se actualiza durante el entrenamiento y sobrevive al cierre del notebook
                with (batch_dir / f"{name}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                saved = completed_run(settings)
                if saved is None:
                    raise RuntimeError("El proceso terminó sin generar todos los resultados.")
                config = json.loads((saved / "config.json").read_text())
                run = client.get_run(config["mlflow_run_id"])
                if run.info.status != "FINISHED":
                    raise RuntimeError(f"MLflow terminó con estado {run.info.status}.")
                row.update(status="finished", run_dir=str(saved))
            print(f"[{index}/{len(plan)}] {name}: {row['status']}", flush=True)
        except Exception as error:
            row.update(status="failed", error=str(error))
            raise
        finally:  # Conserva el progreso incluso si el entrenamiento termina con un error
            progress_file.write_text(json.dumps(progress, indent=2))
    print("COLA COMPLETA: todos los experimentos están guardados en MLflow.", flush=True)


if __name__ == "__main__":
    main()
