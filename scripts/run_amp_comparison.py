"""Compara AMP y FP32 con tres semillas emparejadas, sin usar test"""

import json
import subprocess
import sys
import time
from pathlib import Path

from run_augmentation_resnet import wait_for_gpu  # Comparte la espera de GPU con la cola de aumentos sin ejecutar esa cola al importar

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs' / 'amp_comparison'


def main():
    """Espera la cola anterior y ejecuta seis entrenamientos secuenciales"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    progress = OUTPUT / 'progress.json'
    progress.write_text(json.dumps({'status': 'waiting', 'total': 6}))
    # Esta comparación tiene prioridad. La GPU debe estar libre al iniciarla
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True,
    ).strip()
    if active:
        wait_for_gpu()
    completed = 0
    for seed in (42, 43, 44):
        # Alternamos el orden para que AMP no sea siempre la primera corrida
        for amp in ((True, False) if seed % 2 == 0 else (False, True)):  # Alterna el orden AMP/FP32 entre semillas para reducir el sesgo por orden de ejecución
            name = f'amp_comparison_seed{seed}_{"amp" if amp else "fp32"}'
            # Al reanudar, conservamos las corridas completas de este estudio
            saved = []
            for folder in (ROOT / 'runs').glob(f'*_{name}'):  # Busca resultados anteriores para reanudar sin repetir entrenamientos completos
                if not all((folder / file).exists() for file in
                           ('config.json', 'duration.json', 'validation_metrics.json', 'best_model.pt')):
                    continue
                config = json.loads((folder / 'config.json').read_text())
                if config.get('study') == 'amp_comparison' and config.get('seed') == seed and config.get('amp') == amp:
                    saved.append(folder)
            if saved:
                completed += 1
                continue
            progress.write_text(json.dumps(dict(status='running', completed=completed,
                                                total=6, seed=seed, amp=amp, name=name), indent=2))
            command = [sys.executable, '-m', 'pathmnist_tissue.cli', 'train',
                       '--name', name, '--study', 'amp_comparison', '--model', 'simplecnn',
                       '--size', '128', '--normalization', 'batch', '--adaptive-pool',
                       '--batch-size', '32', '--accumulation-steps', '4',
                       '--epochs', '10', '--seed', str(seed), '--augmentation', 'none',
                       '--no-gradient-checkpointing', '--device', 'cuda',
                       '--amp' if amp else '--no-amp',  # Mantiene los demás parámetros del par y cambia la precisión de entrenamiento
                       '--mlflow-uri', 'http://127.0.0.1:5000']
            with (OUTPUT / f'{name}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                progress.write_text(json.dumps(dict(status='failed', name=name,
                                                    completed=completed, total=6), indent=2))
                return result.returncode
            completed += 1
    progress.write_text(json.dumps(dict(status='complete', completed=completed, total=6), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
