"""Ejecuta las comparaciones pendientes de transferencia, aumentos y GroupNorm"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MLFLOW = "http://127.0.0.1:5000"
LOG_DIR = ROOT / "runs" / "augmentation_resnet"
PROGRESS = LOG_DIR / "progress.json"


def wait_for_gpu() -> None:
    """Espera a que termine el proceso que ahora ocupa la GPU

    La RTX tiene alrededor de 1,2 GiB en uso por otro entrenamiento
    Iniciamos la cola cuando la carga externa baje y permanezca baja un minuto
    """
    idle_since = None
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        utilization, memory = map(int, result.stdout.strip().split(","))  # Interpreta utilización en porcentaje y memoria ocupada en MiB para una sola GPU
        if utilization < 10 and memory < 1500:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= 60:  # Exige un minuto continuo de baja carga antes de iniciar la cola
                return
        else:
            idle_since = None  # Una subida de carga reinicia el periodo de espera
        time.sleep(30)


def main() -> int:
    """Entrena las cinco configuraciones pendientes, una a la vez"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{MLFLOW}/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("MLflow no está disponible.")

    configs = [
        {"name": "resnet18_pretrained", "model": "resnet18", "augmentation": "none"},
        {"name": "resnet18_geometric", "model": "resnet18", "augmentation": "geometric"},
        {"name": "simplecnn_geometric", "model": "simplecnn", "augmentation": "geometric"},
        {"name": "simplecnn_geometric_color", "model": "simplecnn", "augmentation": "geometric_color"},
        {"name": "simplecnn_128_groupnorm", "model": "simplecnn", "augmentation": "none", "normalization": "group"},
    ]
    PROGRESS.write_text(json.dumps({"status": "waiting_for_amp_comparison", "total": len(configs),
                                    "plan": configs}, indent=2), encoding="utf-8")
    # La comparación AMP/FP32 tiene prioridad sobre esta cola
    while subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "pathmnist-amp-comparison.service"],
    ).returncode == 0:
        time.sleep(30)
    wait_for_gpu()
    for index, config in enumerate(configs, start=1):
        command = [
            sys.executable, "-m", "pathmnist_tissue.cli", "train",
            "--name", config["name"], "--model", config["model"],
            "--size", "128", "--epochs", "10", "--batch-size", "32",
            "--accumulation-steps", "4", "--augmentation", config["augmentation"],
            "--seed", "42", "--amp", "--adaptive-pool",
            "--mlflow-uri", MLFLOW,
        ]
        if config.get("normalization") == "group":
            command.extend(["--normalization", "group"])
        if config["model"] == "resnet18":
            command.extend(["--head-epochs", "2", "--finetune-lr", "0.0001",
                            "--gradient-checkpointing"])  # Activa el ahorro de memoria para ResNet al ajustar su último bloque
        log_path = LOG_DIR / f"{index:02}_{config['name']}.log"
        state = {**config, "index": index, "total": len(configs),
                 "status": "running", "started_at": datetime.now().isoformat()}
        PROGRESS.write_text(json.dumps(state, indent=2), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["status"] = "complete" if result.returncode == 0 else "failed"
        state["return_code"] = result.returncode
        state["finished_at"] = datetime.now().isoformat()
        state["log"] = str(log_path)
        PROGRESS.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if result.returncode:  # Detiene la cola ante el primer fallo para evitar encadenar entrenamientos inválidos
            return result.returncode
    PROGRESS.write_text(json.dumps({"status": "complete", "completed": len(configs),
                                    "total": len(configs), "finished_at": datetime.now().isoformat()},
                                   indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
