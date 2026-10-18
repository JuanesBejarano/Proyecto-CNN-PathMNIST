"""Fija la selección por validación y ejecuta test sin volver a entrenar"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mlflow.tracking import MlflowClient
from pathmnist_tissue.cli import choose_device, test
from pathmnist_tissue.tracking import log_test_to_mlflow

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--mlflow-uri', default='http://127.0.0.1:5000')
    args = parser.parse_args()
    selection_file = ROOT / 'results' / 'final_selection.json'
    selection_file.parent.mkdir(exist_ok=True)
    if selection_file.exists():  # Respeta la elección fijada antes de consultar test aunque aparezcan corridas nuevas
        selection = json.loads(selection_file.read_text())
    else:
        candidates = []
        for folder in sorted((ROOT / 'runs').iterdir()):
            if not folder.is_dir() or not all((folder / name).exists() for name in (
                'config.json', 'best_model.pt', 'best_validation.json',
                'duration.json', 'validation_metrics.json',
            )):
                continue
            report = json.loads((folder / 'validation_metrics.json').read_text())
            candidates.append({'run': folder.name, 'validation_macro_f1': report['macro avg']['f1-score']})
        if not candidates:
            raise RuntimeError('No hay corridas completas para seleccionar.')
        winner = max(candidates, key=lambda row: row['validation_macro_f1'])  # Selecciona solo con validación y conserva el orden de ruta para resolver empates
        selection = dict(**winner, selected_at=datetime.now(timezone.utc).isoformat(),
                         criterion='Mayor F1 macro de validación; empate por orden de ruta.',
                         checkpoint_criterion='Menor pérdida de validación dentro de la corrida.',
                         candidates=candidates,
                         checkpoint_sha256=hashlib.sha256((ROOT / 'runs' / winner['run'] / 'best_model.pt').read_bytes()).hexdigest())
        # Se registra antes de cargar o consultar la partición test
        with selection_file.open('x') as file:  # La creación exclusiva impide sobrescribir una selección que ya exista
            json.dump(selection, file, indent=2)
            file.write('\n')
    folder = ROOT / 'runs' / selection['run']
    if hashlib.sha256((folder / 'best_model.pt').read_bytes()).hexdigest() != selection['checkpoint_sha256']:  # Comprueba que los pesos siguen siendo exactamente los que se seleccionaron
        raise RuntimeError('El checkpoint seleccionado cambió; se cancela la evaluación.')
    config = json.loads((folder / 'config.json').read_text())
    # Comprobar MLflow antes de inferencia permite reintentar sin perder resultados
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    if config.get('mlflow_run_id'):
        client.get_run(config['mlflow_run_id'])
    files = ('test_metrics.json', 'confusion_matrix.csv', 'test_predictions.csv')
    available = [(folder / name).exists() for name in files]
    if any(available) and not all(available):  # No reutiliza una evaluación parcial porque sus archivos podrían no corresponderse
        raise RuntimeError('La evaluación guardada está incompleta; revisar antes de continuar.')
    if all(available):
        print('Se reutiliza test guardado; no se repite inferencia.')
        log_test_to_mlflow(folder, config, args.mlflow_uri)  # Permite reintentar el registro en MLflow sin volver a consultar test
    else:
        choose_device(args.device)
        test(argparse.Namespace(run_dir=folder, device=args.device, mlflow_uri=args.mlflow_uri))
    report = json.loads((folder / 'test_metrics.json').read_text())
    print(json.dumps({'run': folder.name, 'test_accuracy': report['accuracy'],
                      'test_macro_f1': report['macro avg']['f1-score'],
                      'test_samples': report['macro avg']['support']}, indent=2))


if __name__ == '__main__':
    main()
