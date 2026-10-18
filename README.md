# Proyecto CNN - PathMNIST

Sistemas Inteligentes II + Introducción al Deep Learning  
Universidad El Bosque — Periodo 2026-2  
Docente: Andrés Mauricio Suárez Mantilla  
Estudiante: Juan Bejarano

## Descripción del proyecto

Proyecto de Sistemas Inteligentes II. Pregunta central: **¿qué clases de tejido confunde una CNN sencilla y cómo cambian esas confusiones cuando se conserva más detalle espacial?** La hipótesis formulada tras el baseline de 28 × 28, antes de las nuevas corridas, se encuentra en la sección 2 del notebook: al aumentar la resolución con BatchNorm fijo, esperamos menos casos de `tumor_epithelium → debris` y mayor recall de `tumor_epithelium` en validación.

## Cuaderno principal y resultados

- [Notebook de análisis y resultados](notebooks/01_resultados.ipynb): hipótesis, comparaciones, curvas, errores por clase y evaluación final, con tablas y figuras guardadas
- [Evidencia experimental](results/README.md): resumen de 29 carpetas de experimentos, 27 con evaluación completa
- [Selección final](results/final_selection.json): configuración fijada por validación antes de consultar test

El modelo final, ResNet-18 con aumentos geométricos a 128×128, obtuvo **91,53 % de exactitud** y **0,8843 de F1 macro** sobre las 7.180 imágenes oficiales de test. El F1 macro de validación fue 0,9842; esta diferencia se analiza en el notebook.

## Estructura

- `src/resnet.py`: transferencia con ResNet-18 preentrenada y ajuste de su último bloque
- `src/checkpointing.py`: ahorro de memoria con conservación de estadísticas de BatchNorm
- `src/compatibility.py`: compatibilidad de checkpoints y huellas de arquitectura
- `src/model.py`: CNN base de tres bloques Conv–(BatchNorm opcional)–ReLU–MaxPool, global average pooling y salida de nueve clases.
- `src/data.py`: PathMNIST y particiones oficiales.
- `src/engine.py`: entrenamiento y validación.
- `src/evaluation.py`: métricas por clase, matrices de confusión y predicciones de validación y test.
- `src/features.py`: activaciones intermedias de los tres bloques.
- `src/tracking.py`: registro opcional en MLflow.
- `src/cli.py`: comandos y registro de corridas.
- `notebooks/01_resultados.ipynb`: análisis y comparación de resultados.
- `scripts/`: automatización de experimentos, evaluación final y exportación de evidencia
- `tests/`: comprobaciones de gradientes, acumulación y checkpointing
- `results/`: evidencia ligera versionada
- `data/`: descargas automáticas; `runs/`: modelos y resultados. Ambas carpetas se excluyen de Git.

## Resultados y publicación en GitHub

La evidencia obtenida se conserva en [results/README.md](results/README.md), con
métricas de 29 carpetas de experimentos (27 completas), curvas, matrices de
confusión y predicciones comprimidas. Se puede revisar sin descargar el dataset,
instalar PyTorch ni iniciar MLflow. Para actualizar esta copia desde las corridas
locales, ejecutar `python3 scripts/export_results.py`.

`data/`, `.venv/`, `runs/`, `mlflow.db`, `mlartifacts/` y `mlruns/` permanecen en
el equipo y se excluyen mediante `.gitignore`. No hay que borrarlos para publicar
el código. La base de MLflow y sus artefactos deben conservarse juntos para
seguir consultando los experimentos completos. Los pesos no viajan en Git;
las instrucciones de entrenamiento permiten generarlos nuevamente.

## Evaluación final

La configuración final se fijó por mayor F1 macro de validación antes de consultar
test: ResNet-18 a 128 píxeles con aumentos geométricos. En las 7.180 imágenes
oficiales de test obtuvo accuracy **0,9153** y F1 macro **0,8843**.
La evaluación completa y sus limitaciones están en [results/README.md](results/README.md).

```bash
uv run python scripts/evaluate_final.py
uv run python scripts/export_results.py
```

El primer comando requiere MLflow local activo. Conserva la elección en
`results/final_selection.json`, verifica el hash del checkpoint y reutiliza los
archivos de test si ya existen; no repite la inferencia. Si falló el registro en
MLflow, puede ejecutarse de nuevo para sincronizar los resultados guardados.
No se debe usar test para seguir seleccionando configuraciones.

## Preparación con uv

Se usa Python 3.12. Para obtener el proyecto:

```bash
git clone https://github.com/JuanesBejarano/Proyecto-CNN-PathMNIST.git
cd Proyecto-CNN-PathMNIST
```

Después instala las dependencias y selecciona `.venv/bin/python` como kernel del notebook:

```bash
uv sync --group notebook
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin CUDA')"
```

Si la segunda línea indica `False`, revisa el controlador NVIDIA y la instalación de PyTorch antes de entrenar. El comando de entrenamiento exige CUDA por defecto para evitar que una corrida larga pase inadvertidamente a CPU.

## Primera corrida

```bash
uv run pathmnist-train train --name baseline_28 --size 28 --epochs 10
```

Cada corrida crea una carpeta única en `runs/` con `config.json`, una copia de `model.py` llamada `model_source.py`, `history.csv`, `best_model.pt`, `best_validation.json` y `duration.json`. El mejor modelo se selecciona con la menor pérdida de validación. Para ese modelo se guardan además `validation_metrics.json`, `validation_confusion_matrix.csv` y `validation_predictions.csv`, con métricas por clase y el índice de cada imagen para revisar errores. La partición **test no se consulta durante los experimentos**.

Después de explorar los errores y escribir una hipótesis, se puede repetir la **misma arquitectura** a 64 píxeles:

```bash
uv run pathmnist-train train --name resolucion_64 --size 64 --epochs 10
```

BatchNorm está activado por defecto. Para comparar su efecto a una resolución fija, entrena otra corrida con `--no-batch-norm` y conserva iguales las épocas, el lote y la semilla:

```bash
uv run pathmnist-train train --name baseline_28_sin_bn --size 28 --epochs 10 --no-batch-norm
```

Las opciones `--batch-norm` y `--adaptive-pool` están activadas por defecto y pueden desactivarse con `--no-batch-norm` y `--no-adaptive-pool`. Ambos valores quedan en `config.json` y se registran como parámetros en MLflow. Sin el promedio global, `flatten` conserva todas las posiciones de los mapas y la capa lineal se construye para la resolución indicada por `--size`.

Para generar el análisis de validación de una corrida anterior sin reentrenar:

```bash
uv run pathmnist-train analyze-validation runs/NOMBRE_DE_LA_CORRIDA
```

Cuando se haya elegido la configuración final, se evalúa una sola vez con:

```bash
uv run pathmnist-train test runs/NOMBRE_DE_LA_CORRIDA
```

Esto añade `test_metrics.json`, `confusion_matrix.csv` y `test_predictions.csv` a esa corrida. Los nombres de clase siguen el orden oficial de PathMNIST: adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon, stroma y tumor epithelium.

Si más adelante se modifica `model.py`, la evaluación de una corrida antigua avisará cuando el código ya no sea compatible. Las corridas anteriores al interruptor de BatchNorm se interpretan como `batch_norm=True`; todas las anteriores a esta comparación se interpretan como `adaptive_pool=True`. La copia `model_source.py` permite revisar la arquitectura exacta que produjo sus pesos.

## Acumulación de gradientes y checkpointing

El entrenamiento acepta `--accumulation-steps` (por defecto 1) y `--gradient-checkpointing` (desactivado por defecto). Por ejemplo, un lote físico de 32 acumulado durante 4 pasos da un lote efectivo nominal de 128:

```bash
uv run pathmnist-train train --name simplecnn_128_acc4_checkpoint --size 128 --epochs 10 --batch-size 32 --accumulation-steps 4 --gradient-checkpointing --mlflow-uri http://127.0.0.1:5000
```

Los gradientes se promedian por el número real de imágenes de cada grupo, incluido el último grupo incompleto. BatchNorm sigue viendo el lote físico: 32 imágenes en este ejemplo. El checkpointing recalcula activaciones durante backward para ahorrar memoria, manteniendo una sola actualización de estadísticas de BatchNorm por lote. No modifica los pesos guardados ni se activa en evaluación.

Las dos opciones, el lote físico y el lote efectivo quedan registrados en `config.json` y MLflow. `duration.json` y MLflow registran el tiempo y, en CUDA, el pico de memoria asignada a tensores. Las comparaciones y sus resultados están en [results/README.md](results/README.md).

## MLflow local

MLflow está instalado como dependencia del proyecto. Desde esta carpeta, abre una terminal y deja el servidor funcionando:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Abre `http://127.0.0.1:5000` en el navegador. MLflow gestiona su almacenamiento local automáticamente; no hay que escribir SQL. El entrenamiento puede registrar parámetros, métricas por época y los archivos de la corrida al añadir la dirección del servidor:

```bash
uv run pathmnist-train train --name simplecnn_64 --size 64 --epochs 10 --mlflow-uri http://127.0.0.1:5000
```

La celda de entrenamiento del notebook ya incluye esa dirección. Si el servidor está apagado, la conexión con MLflow avisará antes de empezar a descargar o entrenar. Para trabajar solo con archivos locales, pon `MLFLOW_URI = None` en la celda. Si ya existe una corrida completa en `runs/`, el notebook la importa a MLflow sin volver a entrenarla. También puedes importar una manualmente:

```bash
uv run pathmnist-train sync-mlflow runs/NOMBRE_DE_LA_CORRIDA --mlflow-uri http://127.0.0.1:5000
```

La evaluación test se registra en la misma corrida MLflow únicamente al ejecutar el comando `test` al final.

## Notebook

Abre `notebooks/01_resultados.ipynb` con el kernel de `.venv`. El notebook incluye tablas y figuras guardadas para consultarlas en GitHub. Su ejecución por defecto lee la evidencia de `results/runs/`, con las corridas locales nuevas de `runs/` como alternativa, sin entrenar, descargar datos ni repetir inferencia.

La sección 2 conserva la hipótesis formulada tras el baseline. La sección 3 contiene el entrenamiento opcional de las doce variantes SimpleCNN; `RUN_TRAINING = False` impide iniciarlo accidentalmente. Las secciones 4 a 6 muestran el ranking de validación, las comparaciones controladas y las curvas, separando las extensiones de los controles. La sección 7 analiza errores de validación con predicciones guardadas y permite abrir imágenes locales con `SHOW_ERROR_IMAGES = True`. La sección 8 muestra las activaciones ya conservadas del baseline; no son mapas de la ResNet final.

La sección 9 compara AMP/FP32, aumentos, transferencia y GroupNorm. La sección 10 muestra la evaluación final completada de ResNet-18 en las 7.180 imágenes de test y su diferencia frente a validación. La sección 11 reúne conclusiones y límites. Para profundizar el EDA con los datos locales, activa `LOAD_LOCAL_EDA`; la opción no descarga archivos. Las resoluciones de 64, 128 y 224 de MedMNIST+ proceden de las imágenes originales, no de agrandar la versión de 28×28.


### Transferencia con ResNet-18

ResNet conserva su BatchNorm y promedio global originales. Recibe imágenes RGB
entre 0 y 1 y aplica dentro del modelo la normalización de ImageNet, conservando
la resolución nativa. La primera ejecución descarga los pesos preentrenados.
Al cargar un checkpoint guardado no necesita descargarlos otra vez.

```bash
uv run pathmnist-train train --name resnet18_128 --model resnet18 \
  --size 128 --epochs 10 --head-epochs 2 --finetune-lr 0.0001 \
  --batch-size 32 --accumulation-steps 4 --augmentation none \
  --mlflow-uri http://127.0.0.1:5000
```

Las primeras dos épocas entrenan solo la capa final; las ocho restantes ajustan
además `layer4`. Los bloques anteriores permanecen congelados, incluidas sus
estadísticas de BatchNorm. El optimizador conserva el estado de la capa final
al añadir el último bloque. CUDA activa la precisión mixta del motor existente.
Para comparar orientación, usar `--augmentation geometric`; para añadir cambios
suaves de color, `--augmentation geometric_color`. Validación y test no reciben
aumentos. `--gradient-checkpointing` permite recalcular layer4 durante su ajuste.
Los parámetros, métricas y pesos se guardan con el mismo flujo de MLflow.

### Comparación con GroupNorm

`--normalization group` sustituye BatchNorm por GroupNorm con ocho grupos en
cada bloque de SimpleCNN. Cada imagen tiene sus propias estadísticas, tanto
en entrenamiento como en evaluación. Los canales 32, 64 y 128 son divisibles
entre ocho. La opción explícita `--normalization` tiene prioridad sobre
`--batch-norm`; omitirla conserva el comportamiento anterior.

Configuración preparada frente al control de 128×128 con BatchNorm, GAP y 32×4:

```bash
uv run pathmnist-train train --name simplecnn_128_groupnorm --model simplecnn \
  --normalization group --adaptive-pool --size 128 --epochs 10 \
  --batch-size 32 --accumulation-steps 4 --augmentation none \
  --seed 42 --no-gradient-checkpointing --mlflow-uri http://127.0.0.1:5000
```

La precisión mixta se activa en CUDA. MLflow registra el tipo de normalización
 y los ocho grupos. Esta corrida es independiente de la cola de aumentos y ResNet.

## Pruebas

```bash
uv run python -m unittest discover -s tests -v
```

Las pruebas comprueban la equivalencia de acumulación frente a lotes grandes, el grupo final incompleto y la conservación de gradientes y buffers de BatchNorm con checkpointing, incluida precisión mixta. Pueden ejecutarse en CPU.
