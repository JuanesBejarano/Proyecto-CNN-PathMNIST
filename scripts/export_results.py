"""Conserva evidencia ligera y verificable sin modificar corridas ni MLflow"""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    rows, manifest = [], []
    for folder in sorted((ROOT / 'runs').iterdir()):
        if not folder.is_dir() or not (folder / 'config.json').exists():
            continue
        dest = output / 'runs' / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for source in sorted(folder.iterdir()):
            if source.suffix not in ('.json', '.csv') and source.name != 'model_source.py':  # Copia evidencia y arquitectura, dejando fuera checkpoints pesados y logs
                continue
            raw = source.read_bytes()
            exported = raw
            if source.name == 'config.json':
                portable_config = json.loads(raw)
                portable_config['data_dir'] = 'data'  # Publica una ruta relativa sin revelar la ubicación local del proyecto
                exported = (json.dumps(portable_config, indent=2) + '\n').encode()
            target = dest / source.name
            if source.name.endswith('_predictions.csv'):
                target = target.with_suffix('.csv.gz')
                target.write_bytes(gzip.compress(raw, mtime=0))  # Comprime sin pérdida y fija la fecha interna para obtener copias reproducibles
            else:
                target.write_bytes(exported)
            manifest.append({'source': source.relative_to(ROOT).as_posix(),
                             'copy': target.relative_to(output).as_posix(),
                             'source_sha256': hashlib.sha256(raw).hexdigest(),  # Registra la huella original para comprobar después la integridad de la evidencia
                             'copy_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                             'transformation': 'relative_data_dir' if source.name == 'config.json' else 'none'})
        cfg = json.loads((folder / 'config.json').read_text())
        metrics_path = folder / 'validation_metrics.json'
        row = dict(run=folder.name, name=cfg['name'], size=cfg['size'],
                   model=cfg.get('model', 'simplecnn'), seed=cfg['seed'],
                   normalization=cfg.get('normalization', 'batch' if cfg.get('batch_norm', True) else 'none'),
                   adaptive_pool=cfg.get('adaptive_pool', True),
                   batch_size=cfg['batch_size'], accumulation_steps=cfg.get('accumulation_steps', 1),
                   gradient_checkpointing=cfg.get('gradient_checkpointing', False),
                   augmentation=cfg.get('augmentation', 'none'), precision=cfg.get('precision', 'unspecified'),
                   complete=all((folder / n).exists() for n in ('duration.json', 'validation_metrics.json', 'best_model.pt')),
                   accuracy=None, macro_f1=None, tumor_recall=None, tumor_to_debris=None, train_seconds=None)
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            row.update(accuracy=m['accuracy'], macro_f1=m['macro avg']['f1-score'], tumor_recall=m['tumor_epithelium']['recall'])
            matrix = list(csv.reader(io.StringIO((folder / 'validation_confusion_matrix.csv').read_text())))
            tumor = [int(v) for v in matrix[9][1:]]  # Lee la fila de tumor, contando la cabecera y excluyendo la columna con el nombre de clase
            row['tumor_to_debris'] = tumor[2] / sum(tumor)  # Calcula qué proporción de imágenes reales de tumor fueron clasificadas como residuos
        if (folder / 'duration.json').exists():
            row['train_seconds'] = json.loads((folder / 'duration.json').read_text())['train_seconds']
        rows.append(row)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    complete = [r for r in rows if r['complete']]
    ranking = sorted(complete, key=lambda r: r['macro_f1'], reverse=True)  # Ordena únicamente corridas completas usando F1 macro de validación
    report = '''# Resultados conservados de PathMNIST

## Alcance y entrega

El proyecto documenta la trazabilidad de los experimentos y sus resultados. Este resumen conserva la evidencia en GitHub y sirve de base para la argumentación final. No sustituye el EDA ni el análisis visual de errores.

La pregunta es cómo cambian las confusiones de tejidos al conservar más detalle espacial. La hipótesis inicial espera menos errores tumor_epithelium → debris y mayor recall de tumor al aumentar resolución, manteniendo fija la normalización y el clasificador.

Los checkpoints se seleccionaron por menor pérdida de validación. La tabla usa la evaluación de validación guardada; los experimentos recientes indican evaluación FP32, aunque las curvas pueden proceder de AMP. Esta exportación no entrena ni evalúa modelos.

## Evidencia disponible

'''
    report += f'Se encontraron {len(rows)} carpetas de experimentos, {len(complete)} completas con métricas y checkpoint. Las carpetas sin evaluación no se incluyen en el ranking.\n\n'
    report += '| Corrida | F1 macro | Accuracy | Recall tumor | Tumor → debris | Tiempo (s) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for r in ranking:
        report += f"| [{r['name']}](runs/{r['run']}/validation_metrics.json) | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['tumor_recall']:.4f} | {100*r['tumor_to_debris']:.2f}% | {r['train_seconds']:.1f} |\n"
    report += '''
## Lectura de la hipótesis principal

Las siguientes comparaciones mantienen GAP, lote físico 128, sin acumulación ni checkpointing, SimpleCNN sin aumentos y semilla 42. Se usan únicamente corridas completas y se separa el estado de BatchNorm.

| BatchNorm | Resolución | F1 macro | Recall tumor | Tumor → debris |
| --- | ---: | ---: | ---: | ---: |
'''
    for norm in ('batch', 'none'):
        for size in (28, 64, 128):
            candidates = [r for r in complete if r['model'] == 'simplecnn' and r['normalization'] == norm and r['size'] == size and r['adaptive_pool'] and r['batch_size'] == 128 and r['accumulation_steps'] == 1 and not r['gradient_checkpointing'] and r['augmentation'] == 'none' and r['seed'] == 42 and 'amp_comparison' not in r['name']]
            if candidates:
                r = candidates[-1]
                report += f"| {'Sí' if norm == 'batch' else 'No'} | {size} | {r['macro_f1']:.4f} | {r['tumor_recall']:.4f} | {r['tumor_to_debris']*100:.2f}% |\n"
    report += '''
La evidencia debe interpretarse por comparación controlada: el ranking mezcla arquitectura, normalización, aumentos y resolución, de modo que sus diferencias no identifican por sí solas una causa. Mayor resolución no garantiza una mejora uniforme. Con BatchNorm, pasar de 28 a 64 mejora el recall de tumor de 0,5649 a 0,9811 y reduce tumor → debris de 24,30% a 0%; a 128 el recall cae a 0,7053 y la confusión vuelve a 10,96%. Sin BatchNorm, 64 y 128 mejoran frente a 28, pero la confusión aumenta ligeramente de 0,35% a 0,42% entre 64 y 128. Se observa respaldo parcial a la hipótesis, no una mejora monótona con la resolución. Estas cifras describen las corridas conservadas y deben leerse junto con las matrices completas.

Las extensiones de AMP tienen tres semillas; la mayoría de las demás comparaciones solo una. No se afirma significancia estadística ni equivalencia entre métodos. Tampoco se trata este ranking de validación como evaluación final de generalización.

## Qué se preserva

- `summary.csv`: configuración resumida y métricas de todas las carpetas encontradas.
- `runs/`: configuraciones originales, copia del modelo fuente, curvas (`history.csv`), selección del checkpoint, duración, métricas por clase y matrices de confusión disponibles.
- Predicciones por imagen comprimidas sin pérdida (`validation_predictions.csv.gz`), para conservar índices, etiquetas y errores sin cargar los pesos.
- `manifest.json`: rutas de origen y hashes SHA-256 para verificar cada copia. Las configuraciones publicadas normalizan `data_dir` a `data`; los originales locales no se modifican.
- `examples/`: imágenes de muestra y activaciones conservadas del baseline. Son ilustraciones conservadas, no una nueva selección de errores.

Los pesos permanecen en `../runs/`; MLflow conserva `../mlflow.db` y `../mlartifacts/`. No se incluyen en Git. Un clon del repositorio dispone de las métricas y predicciones, pero necesita entrenar o recuperar los pesos locales para ejecutar nueva inferencia. El notebook original consulta `../runs/`; esta copia de evidencia puede consultarse directamente sin ejecutarlo.

## Pendientes para cerrar la entrega

1. Revisar el EDA del notebook y conectar sus similitudes visuales con las confusiones observadas.
2. Incorporar curvas, matrices y ejemplos concretos de errores al argumento final; están conservados los datos necesarios.
3. Justificar cada comparación y responder la hipótesis con resultados favorables y negativos.
4. Elegir la configuración final con validación y realizar la evaluación final sobre test. No se encontraron métricas test guardadas al hacer esta exportación; no se ejecutó test durante la limpieza.
'''
    selection_path = output / 'final_selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        final = ROOT / 'runs' / selection['run'] / 'test_metrics.json'
        if final.exists():
            metrics = json.loads(final.read_text())
            report = report.replace(
                '4. Elegir la configuración final con validación y realizar la evaluación final sobre test. No se encontraron métricas test guardadas al hacer esta exportación; no se ejecutó test durante la limpieza.',
                '4. Evaluación final completada; ver la sección siguiente. No usar test para seguir ajustando la configuración.',
            )
            report += '\n## Evaluación final sobre test\n\n'
            report += f"Modelo fijado antes de test: `{selection['run']}`. Selección por mayor F1 macro de validación ({selection['validation_macro_f1']:.4f}) entre las corridas completas; checkpoint por menor pérdida de validación. Registro y hash de los pesos: [final_selection.json](final_selection.json).\n\n"
            report += f"Se evaluaron {int(metrics['macro avg']['support'])} imágenes oficiales de test, sin aumentos y en FP32. Accuracy: **{metrics['accuracy']:.4f}**; F1 macro: **{metrics['macro avg']['f1-score']:.4f}**; recall de tumor: **{metrics['tumor_epithelium']['recall']:.4f}**.\n\n"
            report += f"La diferencia F1 macro test - validación es {metrics['macro avg']['f1-score'] - selection['validation_macro_f1']:+.4f}. Describe la generalización de este checkpoint; no demuestra superioridad estadística ni sustituye el análisis de errores.\n\n"
            report += f"Evidencia: [métricas por clase](runs/{selection['run']}/test_metrics.json), [matriz de confusión](runs/{selection['run']}/confusion_matrix.csv) y predicciones comprimidas en esa misma carpeta. Los resultados también se registran en la corrida MLflow original.\n"
    (output / 'README.md').write_text(report)
    print(f'Exportadas {len(rows)} corridas y {len(manifest)} archivos de evidencia; {len(complete)} completas.')


if __name__ == '__main__':
    main()
