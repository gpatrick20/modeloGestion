# Experimento: notas académicas como antecedentes

Este experimento compara un modelo que incorpora los puntajes de lectura y ciencias como antecedentes académicos, manteniendo el objetivo de riesgo en matemática.

- Variable objetivo: `TestScore_Math < 70` es `en_riesgo`; desde 70 es `sin_riesgo`.
- Entradas experimentales: `AttendanceRate`, `StudyHours`, `TestScore_Reading`, `TestScore_Science`.
- Modelo: regresión logística balanceada con `StandardScaler`.
- Entrenamiento: solo `data/samples_math/train_math.csv`.
- Evaluación automática: solo `data/samples_math/validation_math.csv`.
- El test no se abre ni se usa para seleccionar modelos.

## Ejecución

Desde la raíz, con el entorno activo:

```powershell
python -m unittest discover -s tests -p 'test_notas_academicas.py'
python -m src.experimento_notas_academicas
```

## Comparación con el modelo base

Tras ejecutar ambos modelos, compara los archivos:

```text
reports/riesgo70/resumen_validation.csv
reports/experimento_notas_academicas/resumen_validacion.csv
```

Prioriza `recall_en_riesgo` y `precision_en_riesgo`; el accuracy no es suficiente para decidir, porque las clases están desbalanceadas. No compares resultados de test con validación.

## Límites

El uso de lectura y ciencias requiere que esas notas estén disponibles. El modelo no es una alerta temporal validada: el dataset no documenta la secuencia entre evaluaciones. Los scores no están calibrados y no deben interpretarse como probabilidades fiables de necesidad de apoyo. La decisión de acompañamiento corresponde al docente.
