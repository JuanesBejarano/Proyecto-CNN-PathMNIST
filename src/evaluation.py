"""Métricas y predicciones finales para análisis por clase."""

import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import CLASS_NAMES


@torch.inference_mode() # Este decorador hace que el contenido de la función se ejecute solo en modo inferencia, evitando así crear el grafo computacional para backward
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    output: Path, *, split: str,
) -> dict:
    """Calcula y guarda métricas por clase y predicciones individuales.

    Args:
        model: Modelo entrenado que produce logits.
        loader: Partición sin aleatorización para conservar los índices.
        device: Dispositivo de inferencia.
        output: Carpeta donde se guardan los archivos.
        split: "validation" o "test".

    Returns:
        Reporte de clasificación de scikit-learn.
    """
    if split not in ("validation", "test"):
        raise ValueError("split debe ser 'validation' o 'test'.")

    model.eval() # Setea el modelo en modo evaluación, batch_norm deja de usar las estadísticas de los batches en entrenamiento, y usa las acumuladas del mismo (con la media móvil) y dropout deja de apagar neuronas 
    truth: list[int] = []
    predictions: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True)) # non_blocking permite que el hilo de Python se ejecute sin esperar a que la transferencia de los tensores al acelerador
        truth.extend(targets.view(-1).tolist()) # view(-1) aplana el tensor, y tolist() pasa el tensor a una lista de Python, y extend es como append pero preserva la dimensión original de la lista (e.g, tengo [1, 2, 3].extend([4]), obtengo [1, 2, 3, 4], con append tendría [1, 2, 3, [4]])
        predictions.extend(logits.argmax(dim=1).cpu().tolist()) # argmax devuelve el índice del valor máximo a lo largo de las columnas (dim=1) de cada fila

    class_ids = list(range(len(CLASS_NAMES))) # list(0...8)
    report = classification_report(  # Calcula precision, recall y F1 por clase, además de accuracy y promedios
    truth, predictions, labels=class_ids, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0,  # Devuelve un diccionario y usa 0 cuando una división no está definida
    )
    _save_results(output, split, truth, predictions, report, class_ids)  # Guarda métricas, matriz de confusión y predicciones
    return report


def _save_results(
    output: Path, split: str, truth: list[int], predictions: list[int],
    report: dict, class_ids: list[int],
) -> None:
    """Guarda el reporte, la matriz de confusión y cada predicción.

    Args:
        output: Carpeta de la corrida.
        split: "validation" o "test".
        truth: Clases reales en el orden del dataset.
        predictions: Clases predichas en ese mismo orden.
        report: Métricas calculadas por clase.
        class_ids: Identificadores de clase en el orden de CLASS_NAMES.
    """
    (output / f"{split}_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    matrix = confusion_matrix(truth, predictions, labels=class_ids)
    # Se conserva el nombre histórico de test para los notebooks ya existentes
    matrix_name = "confusion_matrix.csv" if split == "test" else "validation_confusion_matrix.csv"
    with (output / matrix_name).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *CLASS_NAMES])
        writer.writerows([name, *row] for name, row in zip(CLASS_NAMES, matrix))

    # El índice permite recuperar la imagen original al revisar un error
    with (output / f"{split}_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_class", "predicted_class"])
        writer.writerows(
            (index, true, pred)
            for index, (true, pred) in enumerate(zip(truth, predictions))
        )
