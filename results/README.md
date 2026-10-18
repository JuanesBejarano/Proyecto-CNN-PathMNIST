# Resultados conservados de PathMNIST

## Alcance y entrega

El proyecto documenta la trazabilidad de los experimentos y sus resultados. Este resumen conserva la evidencia en GitHub y sirve de base para la argumentación final. No sustituye el EDA ni el análisis visual de errores.

La pregunta es cómo cambian las confusiones de tejidos al conservar más detalle espacial. La hipótesis inicial espera menos errores tumor_epithelium → debris y mayor recall de tumor al aumentar resolución, manteniendo fija la normalización y el clasificador.

Los checkpoints se seleccionaron por menor pérdida de validación. La tabla usa la evaluación de validación guardada; los experimentos recientes indican evaluación FP32, aunque las curvas pueden proceder de AMP. Esta exportación no entrena ni evalúa modelos.

## Evidencia disponible

Se encontraron 29 carpetas de experimentos, 27 completas con métricas y checkpoint. Las carpetas sin evaluación no se incluyen en el ranking.

| Corrida | F1 macro | Accuracy | Recall tumor | Tumor → debris | Tiempo (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [resnet18_geometric](runs/20261016_064401_184180_resnet18_geometric/validation_metrics.json) | 0.9842 | 0.9839 | 0.9756 | 0.63% | 675.8 |
| [resnet18_pretrained](runs/20261016_063544_988449_resnet18_pretrained/validation_metrics.json) | 0.9758 | 0.9755 | 0.9728 | 0.77% | 454.2 |
| [amp_comparison_seed42_fp32](runs/20261016_023922_916592_amp_comparison_seed42_fp32/validation_metrics.json) | 0.9705 | 0.9714 | 0.9784 | 0.28% | 728.6 |
| [simplecnn_128_bn1_gap1_b32_acc4_cp0](runs/20261015_235352_787695_simplecnn_128_bn1_gap1_b32_acc4_cp0/validation_metrics.json) | 0.9664 | 0.9674 | 0.9777 | 0.21% | 484.7 |
| [simplecnn_128_bn1_gap1_b32_acc4_cp1](runs/20261016_000237_537728_simplecnn_128_bn1_gap1_b32_acc4_cp1/validation_metrics.json) | 0.9664 | 0.9674 | 0.9777 | 0.21% | 763.9 |
| [amp_comparison_seed42_amp](runs/20261016_023005_471048_amp_comparison_seed42_amp/validation_metrics.json) | 0.9664 | 0.9674 | 0.9777 | 0.21% | 485.1 |
| [simplecnn_geometric](runs/20261016_065605_330697_simplecnn_geometric/validation_metrics.json) | 0.9654 | 0.9663 | 0.9302 | 0.28% | 553.7 |
| [simplecnn_128_groupnorm](runs/20261016_071554_498079_simplecnn_128_groupnorm/validation_metrics.json) | 0.9570 | 0.9586 | 0.9497 | 0.07% | 783.2 |
| [simplecnn_128_bn0_gap0_b128_acc1_cp0](runs/20261015_231738_456870_simplecnn_128_bn0_gap0_b128_acc1_cp0/validation_metrics.json) | 0.9490 | 0.9503 | 0.9602 | 0.00% | 422.4 |
| [amp_comparison_seed44_amp](runs/20261016_031530_592083_amp_comparison_seed44_amp/validation_metrics.json) | 0.9469 | 0.9476 | 0.9832 | 0.07% | 1967.0 |
| [simplecnn_64_bn0_gap0_b128_acc1_cp0](runs/20261015_224457_813388_simplecnn_64_bn0_gap0_b128_acc1_cp0/validation_metrics.json) | 0.9447 | 0.9448 | 0.9204 | 0.84% | 131.0 |
| [amp_comparison_seed44_fp32](runs/20261016_062110_345620_amp_comparison_seed44_fp32/validation_metrics.json) | 0.9219 | 0.9211 | 0.9385 | 0.00% | 755.5 |
| [amp_comparison_seed43_fp32](runs/20261016_025215_509860_amp_comparison_seed43_fp32/validation_metrics.json) | 0.9212 | 0.9236 | 0.9728 | 0.14% | 725.1 |
| [simplecnn_geometric_color](runs/20261016_070603_522618_simplecnn_geometric_color/validation_metrics.json) | 0.9181 | 0.9177 | 0.9420 | 0.91% | 542.4 |
| [simplecnn_128_bn0_gap1_b32_acc4_cp0](runs/20261015_233703_155215_simplecnn_128_bn0_gap1_b32_acc4_cp0/validation_metrics.json) | 0.9087 | 0.9109 | 0.9420 | 0.07% | 401.4 |
| [simplecnn_128_bn0_gap1_b32_acc4_cp1](runs/20261015_234424_613445_simplecnn_128_bn0_gap1_b32_acc4_cp1/validation_metrics.json) | 0.9087 | 0.9109 | 0.9420 | 0.07% | 523.7 |
| [amp_comparison_seed43_amp](runs/20261016_030459_957935_amp_comparison_seed43_amp/validation_metrics.json) | 0.9040 | 0.9102 | 0.8610 | 0.00% | 591.2 |
| [simplecnn_64_bn_off](runs/20261015_193412_823239_simplecnn_64_bn_off/validation_metrics.json) | 0.9014 | 0.9033 | 0.9330 | 0.35% | 121.0 |
| [simplecnn_128_bn0_gap1_b128_acc1_cp0](runs/20261015_230833_409274_simplecnn_128_bn0_gap1_b128_acc1_cp0/validation_metrics.json) | 0.8991 | 0.9017 | 0.9406 | 0.42% | 499.4 |
| [simplecnn_128_bn1_gap0_b128_acc1_cp0](runs/20261015_225849_223310_simplecnn_128_bn1_gap0_b128_acc1_cp0/validation_metrics.json) | 0.8703 | 0.8740 | 0.7877 | 0.00% | 533.1 |
| [simplecnn_28_bn0_gap0_b128_acc1_cp0](runs/20261015_224132_416189_simplecnn_28_bn0_gap0_b128_acc1_cp0/validation_metrics.json) | 0.8686 | 0.8692 | 0.9015 | 3.07% | 37.3 |
| [simplecnn_64_bn1_gap0_b128_acc1_cp0](runs/20261015_224217_387934_simplecnn_64_bn1_gap0_b128_acc1_cp0/validation_metrics.json) | 0.8594 | 0.8683 | 0.9001 | 6.15% | 143.5 |
| [simplecnn_64_bn_on](runs/20261015_193144_498352_simplecnn_64_bn_on/validation_metrics.json) | 0.8566 | 0.8624 | 0.9811 | 0.00% | 133.2 |
| [simplecnn_28_bn_off](runs/20261015_185209_947037_simplecnn_28_bn_off/validation_metrics.json) | 0.8256 | 0.8259 | 0.7982 | 7.40% | 36.2 |
| [simplecnn_28_bn1_gap0_b128_acc1_cp0](runs/20261015_224048_365789_simplecnn_28_bn1_gap0_b128_acc1_cp0/validation_metrics.json) | 0.7778 | 0.7989 | 0.6767 | 9.85% | 36.6 |
| [baseline_28](runs/20261010_192602_801787_baseline_28/validation_metrics.json) | 0.7685 | 0.7637 | 0.5649 | 24.30% | 36.4 |
| [simplecnn_128_bn1_gap1_b128_acc1_cp0](runs/20261015_224726_759401_simplecnn_128_bn1_gap1_b128_acc1_cp0/validation_metrics.json) | 0.7669 | 0.7746 | 0.7053 | 10.96% | 629.5 |

## Lectura de la hipótesis principal

Las siguientes comparaciones mantienen GAP, lote físico 128, sin acumulación ni checkpointing, SimpleCNN sin aumentos y semilla 42. Se usan únicamente corridas completas y se separa el estado de BatchNorm.

| BatchNorm | Resolución | F1 macro | Recall tumor | Tumor → debris |
| --- | ---: | ---: | ---: | ---: |
| Sí | 28 | 0.7685 | 0.5649 | 24.30% |
| Sí | 64 | 0.8566 | 0.9811 | 0.00% |
| Sí | 128 | 0.7669 | 0.7053 | 10.96% |
| No | 28 | 0.8256 | 0.7982 | 7.40% |
| No | 64 | 0.9014 | 0.9330 | 0.35% |
| No | 128 | 0.8991 | 0.9406 | 0.42% |

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
4. Evaluación final completada; ver la sección siguiente. No usar test para seguir ajustando la configuración.

## Evaluación final sobre test

Modelo fijado antes de test: `20261016_064401_184180_resnet18_geometric`. Selección por mayor F1 macro de validación (0.9842) entre las corridas completas; checkpoint por menor pérdida de validación. Registro y hash de los pesos: [final_selection.json](final_selection.json).

Se evaluaron 7180 imágenes oficiales de test, sin aumentos y en FP32. Accuracy: **0.9153**; F1 macro: **0.8843**; recall de tumor: **0.9481**.

La diferencia F1 macro test - validación es -0.0998. Describe la generalización de este checkpoint; no demuestra superioridad estadística ni sustituye el análisis de errores.

Evidencia: [métricas por clase](runs/20261016_064401_184180_resnet18_geometric/test_metrics.json), [matriz de confusión](runs/20261016_064401_184180_resnet18_geometric/confusion_matrix.csv) y predicciones comprimidas en esa misma carpeta. Los resultados también se registran en la corrida MLflow original.
