"""Entrenamiento reproducible y evaluación final desde la terminal"""

import argparse
import csv
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .compatibility import model_code_hash, model_source_compatible
from .data import make_loader
from .engine import run_epoch
from .evaluation import evaluate
from .model import SimpleCNN
from .resnet import TransferResNet18
from .tracking import log_validation_to_mlflow, log_test_to_mlflow, sync_saved_run, tracking_run


def choose_device(requested: str) -> torch.device:
    """Selecciona el dispositivo solicitado y comprueba que CUDA exista"""
    if requested == "cuda" and not torch.cuda.is_available():  # Evita pasar a CPU silenciosamente cuando se pidió entrenar con GPU
        raise RuntimeError(
            "CUDA no está disponible. Revisa el controlador NVIDIA y la instalación de PyTorch."
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def seed_all(seed: int) -> None:
    """Fija las semillas usadas por Python, NumPy y PyTorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Fija la semilla de todas las GPU, además de la semilla general de PyTorch


def create_run(args: argparse.Namespace, device: torch.device) -> tuple[Path, dict]:
    """Crea una carpeta única y guarda la configuración y arquitectura"""
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs y batch-size deben ser positivos.")
    if args.head_epochs < 0 or args.finetune_lr <= 0:
        raise ValueError("head-epochs debe ser no negativo y finetune-lr positivo.")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps debe ser al menos 1.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("El nombre solo puede contener letras, números, _ y -.")
    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)  # Impide reutilizar una carpeta existente y sobrescribir otra corrida
    model_source = Path(__file__).with_name("resnet.py" if args.model == "resnet18" else "model.py").read_text(encoding="utf-8")
    config = {
        "name": args.name, "study": args.study, "size": args.size, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "num_workers": args.num_workers, "device": str(device),
        "amp": args.amp and device.type == "cuda",
        "precision": "amp_fp16" if args.amp and device.type == "cuda" else "fp32",
        "validation_precision": "fp32",  # La selección del checkpoint usa validación FP32 aunque el entrenamiento utilice AMP
        "model": args.model,
        "architecture": "ResNet18" if args.model == "resnet18" else ("SimpleCNN-3blocks-GAP" if args.adaptive_pool else "SimpleCNN-3blocks-Flatten"),
        "pretrained": args.model == "resnet18",
        "head_epochs": args.head_epochs,
        "finetune_lr": args.finetune_lr,
        "batch_norm": True if args.model == "resnet18" else (args.normalization == "batch" if args.normalization else args.batch_norm),
        "normalization": "batch" if args.model == "resnet18" else (args.normalization or ("batch" if args.batch_norm else "none")),  # ResNet conserva BatchNorm y en SimpleCNN la opción normalization tiene prioridad sobre batch_norm
        "group_norm_groups": 8 if args.model == "simplecnn" and args.normalization == "group" else 0,
        "augmentation": args.augmentation,
        "adaptive_pool": True if args.model == "resnet18" else args.adaptive_pool,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,  # Lote efectivo nominal, el último grupo puede contener menos imágenes
        "gradient_checkpointing": args.gradient_checkpointing,
        "model_sha256": hashlib.sha256(model_source.encode()).hexdigest(),
        "model_code_sha256": model_code_hash(model_source),  # Permite reconocer el mismo código ejecutable aunque cambien comentarios o formato
        "data_dir": str(args.data_dir.resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "model_source.py").write_text(model_source, encoding="utf-8")  # Conserva la arquitectura utilizada para poder revisar corridas antiguas
    return run_dir, config


def save_epoch(run_dir: Path, history: list[dict], model: SimpleCNN,
               epoch: int, val_metrics: dict[str, float], best_loss: float) -> float:
    """Guarda el historial y actualiza el mejor modelo si mejora la validación"""
    with (run_dir / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    if val_metrics["loss"] < best_loss:  # Selecciona por menor pérdida de validación, no por accuracy ni por el resultado en test
        best_loss = val_metrics["loss"]
        torch.save(model.state_dict(), run_dir / "best_model.pt")
        (run_dir / "best_validation.json").write_text(
            json.dumps({"epoch": epoch, **val_metrics}, indent=2), encoding="utf-8"
        )
    return best_loss


def train(args: argparse.Namespace) -> None:
    """Entrena, selecciona por validación y guarda los resultados de la corrida"""
    if args.model == "resnet18" and args.normalization not in (None, "batch"):
        raise ValueError("ResNet preentrenada conserva BatchNorm; normalization es para SimpleCNN.")
    device = choose_device(args.device)
    seed_all(args.seed)
    run_dir, config = create_run(args, device)
    print(f"Dispositivo: {device} | Resultados: {run_dir}", flush=True)

    with tracking_run(args.mlflow_uri, args.mlflow_experiment, run_dir, config) as tracker:
        train_loader = make_loader(
            split="train", size=args.size, batch_size=args.batch_size,
            augmentation=args.augmentation,
            num_workers=args.num_workers, seed=args.seed, data_dir=args.data_dir,
        )
        val_loader = make_loader(
            split="val", size=args.size, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed, data_dir=args.data_dir,
        )
        if args.model == "resnet18":
            model = TransferResNet18(
                gradient_checkpointing=args.gradient_checkpointing,
            ).to(device)
        else:
            model = SimpleCNN(
                batch_norm=args.batch_norm, adaptive_pool=args.adaptive_pool, size=args.size,
                normalization=args.normalization,
                gradient_checkpointing=args.gradient_checkpointing,
            ).to(device)
        criterion = nn.CrossEntropyLoss()  # Recibe logits sin softmax y etiquetas de clase para calcular la pérdida multiclase
        optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay  # Incluye solo parámetros entrenables para excluir los bloques congelados de ResNet
        )
        scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None  # Escala la pérdida con AMP para reducir el riesgo de que gradientes pequeños se pierdan en FP16
        best_val_loss = float("inf")  # Permite que la primera pérdida finita de validación establezca el checkpoint inicial
        history: list[dict[str, float | int]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)  # Reinicia la medición para excluir el pico anterior al inicio del entrenamiento
            torch.cuda.synchronize(device)  # Espera a que termine el trabajo pendiente de la GPU antes de iniciar el cronómetro
        started = time.monotonic()

        for epoch in range(1, args.epochs + 1):
            if args.model == "resnet18" and epoch == args.head_epochs + 1:
                model.unfreeze_last_block()  # Tras entrenar la cabeza, habilita el ajuste de layer4 manteniendo congelados los bloques anteriores
                optimizer.add_param_group({  # Añade layer4 con su propia tasa de aprendizaje sin reiniciar el estado AdamW de la capa final
                    "params": model.network.layer4.parameters(), "lr": args.finetune_lr,
                })

            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler,
                accumulation_steps=args.accumulation_steps, amp_enabled=args.amp,
            )
            val_metrics = run_epoch(model, val_loader, criterion, device, amp_enabled=False)  # Evalúa en FP32 para seleccionar checkpoints con la misma precisión entre corridas AMP y FP32
            row = {
                "epoch": epoch, "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)
            if tracker is not None:
                tracker.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
            best_val_loss = save_epoch(run_dir, history, model, epoch, val_metrics, best_val_loss)
            print(
                f"Época {epoch:02d}/{args.epochs} | "
                f"train acc {train_metrics['accuracy']:.3f} | "
                f"val acc {val_metrics['accuracy']:.3f} | "
                f"val loss {val_metrics['loss']:.4f}",
                flush=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # Espera a que termine la GPU antes de cerrar la medición de tiempo
        resources = {"train_seconds": time.monotonic() - started}
        if device.type == "cuda":
            resources["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2  # Convierte a MiB el pico de memoria de tensores de PyTorch, no toda la memoria ocupada por el controlador
        (run_dir / "duration.json").write_text(
            json.dumps(resources, indent=2), encoding="utf-8"
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))  # Recupera la mejor época antes del análisis final de validación, que puede diferir de la última
        val_report = evaluate(model, val_loader, device, run_dir, split="validation")
        if tracker is not None:
            tracker.log_metrics(resources)
            tracker.log_metric("best_val_loss", best_val_loss)
            tracker.log_metric("val_macro_f1", val_report["macro avg"]["f1-score"])
            tracker.log_artifacts(str(run_dir), artifact_path="run_files")
        print(f"Modelo seleccionado por menor pérdida de validación: {run_dir / 'best_model.pt'}")


def load_saved_model(run_dir: Path, device: torch.device) -> tuple[dict, nn.Module]:
    """Carga el mejor checkpoint de una corrida compatible"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model") == "resnet18":
        model = TransferResNet18(pretrained=False).to(device)  # Reconstruye la arquitectura sin descargar ImageNet porque los pesos completos vienen del checkpoint
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))  # Recupera la mejor época antes del análisis final de validación, que puede diferir de la última
        return config, model
    current_source = Path(__file__).with_name("model.py").read_text(encoding="utf-8")
    if not model_source_compatible(  # Comprueba la compatibilidad de la arquitectura actual antes de cargar pesos de una corrida anterior
        config, hashlib.sha256(current_source.encode()).hexdigest(), model_code_hash(current_source)
    ):
        raise RuntimeError(
            "La arquitectura actual cambió desde el entrenamiento. "
            "Consulta model_source.py en la carpeta de esa corrida."
        )
    model = SimpleCNN(
        batch_norm=config.get("batch_norm", True),  # Las corridas antiguas sin esta opción se interpretan con BatchNorm activado
        normalization=config.get("normalization"),
        adaptive_pool=config.get("adaptive_pool", True), size=config["size"],
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    return config, model


def analyze_validation(args: argparse.Namespace) -> None:
    """Genera métricas por clase para una corrida ya entrenada"""
    if (args.run_dir / "validation_metrics.json").exists():
        print(f"Análisis de validación ya guardado en {args.run_dir}")
        return
    device = choose_device(args.device)
    config, model = load_saved_model(args.run_dir, device)
    loader = make_loader(
        split="val", size=config["size"], batch_size=config["batch_size"],
        num_workers=0, seed=config["seed"],
        data_dir=Path(config["data_dir"]),
    )
    report = evaluate(model, loader, device, args.run_dir, split="validation")
    log_validation_to_mlflow(args.run_dir, config, report)
    print(f"Análisis de validación guardado en {args.run_dir}")


def test(args: argparse.Namespace) -> None:
    """Evalúa una única vez la partición test de una corrida elegida"""
    if (args.run_dir / "test_metrics.json").exists():  # Evita sobrescribir una evaluación final ya guardada o repetirla accidentalmente
        raise RuntimeError("Esta corrida ya tiene una evaluación de test guardada.")
    device = choose_device(args.device)
    config, model = load_saved_model(args.run_dir, device)
    loader = make_loader(
        split="test", size=config["size"], batch_size=config["batch_size"],
        num_workers=config["num_workers"], seed=config["seed"],
        data_dir=Path(config["data_dir"]),
    )
    evaluate(model, loader, device, args.run_dir, split="test")
    log_test_to_mlflow(args.run_dir, config, args.mlflow_uri)  # Adjunta el resultado de test a la corrida original sin crear un entrenamiento nuevo
    print(f"Evaluación final guardada en {args.run_dir}")


def sync_mlflow(args: argparse.Namespace) -> None:
    """Registra en MLflow una corrida local existente"""
    sync_saved_run(args.run_dir, args.mlflow_uri, args.mlflow_experiment)
    print(f"Corrida importada a MLflow: {args.run_dir}")


def main() -> None:
    """Define los subcomandos de la interfaz de terminal"""
    parser = argparse.ArgumentParser(description="Experimentos CNN con PathMNIST")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Entrena y valida sin usar test")
    train_parser.add_argument("--model", choices=("simplecnn", "resnet18"), default="simplecnn")
    train_parser.add_argument("--head-epochs", type=int, default=2,
                              help="Épocas iniciales solo con la capa final de ResNet")
    train_parser.add_argument("--finetune-lr", type=float, default=1e-4,
                              help="Learning rate del último bloque de ResNet")
    train_parser.add_argument("--study", default="main", help="Identificador de la comparación")
    train_parser.add_argument("--name", default="baseline")
    train_parser.add_argument("--size", type=int, choices=(28, 64, 128, 224), default=28)
    train_parser.add_argument(
        "--batch-norm", action=argparse.BooleanOptionalAction, default=True,  # Genera las opciones --batch-norm y --no-batch-norm para activar o desactivar el mismo ajuste
        help="Usa BatchNorm después de cada convolución (por defecto: activado)",
    )
    train_parser.add_argument(
        "--adaptive-pool", action=argparse.BooleanOptionalAction, default=True,
        help="Promedia cada mapa a 1×1 antes de flatten (por defecto: activado)",
    )
    train_parser.add_argument(
        "--augmentation", choices=("none", "geometric", "geometric_color"),
        default="none", help="Aumentos solo en train: orientación o orientación y color suave",
    )
    train_parser.add_argument(
        "--normalization", choices=("batch", "group", "none"), default=None,
        help="SimpleCNN: sustituye la opción batch-norm; group usa ocho grupos",
    )
    train_parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Precisión mixta FP16 en CUDA; --no-amp usa FP32",
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument(
        "--accumulation-steps", type=int, default=1,
        help="Lotes antes de actualizar pesos; lote efectivo = batch-size × accumulation-steps",
    )
    train_parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="Recalcula activaciones durante backward para ahorrar memoria",
    )
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    train_parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    train_parser.add_argument("--mlflow-uri", default=None,
                              help="Dirección del servidor MLflow, por ejemplo http://127.0.0.1:5000")
    train_parser.add_argument("--mlflow-experiment", default="pathmnist-resolution")
    train_parser.set_defaults(func=train)
    val_parser = commands.add_parser(
        "analyze-validation", help="Genera análisis de validación de una corrida guardada"
    )
    val_parser.add_argument("run_dir", type=Path)
    val_parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    val_parser.set_defaults(func=analyze_validation)
    test_parser = commands.add_parser("test", help="Evalúa una corrida elegida una sola vez")
    test_parser.add_argument("run_dir", type=Path)
    test_parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    test_parser.add_argument("--mlflow-uri", default=None,
                             help="Permite usar otra dirección para la corrida MLflow original")
    test_parser.set_defaults(func=test)
    sync_parser = commands.add_parser("sync-mlflow", help="Importa una corrida existente sin reentrenar")
    sync_parser.add_argument("run_dir", type=Path)
    sync_parser.add_argument("--mlflow-uri", required=True)
    sync_parser.add_argument("--mlflow-experiment", default="pathmnist-resolution")
    sync_parser.set_defaults(func=sync_mlflow)
    args = parser.parse_args()
    args.func(args)  # Ejecuta la función asociada al subcomando mediante set_defaults


if __name__ == "__main__":
    main()
