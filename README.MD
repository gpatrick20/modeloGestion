# Riesgo académico en matemática: corte 70

Prototipo académico con datos sintéticos. Clasificación binaria: `TestScore_Math < 70` define `en_riesgo`; desde 70 define `sin_riesgo` (sin alerta según criterio). Se elimina el amarillo. Una nota exactamente igual a 70 no es roja.

El clasificador es una regresión logística balanceada con StandardScaler. Entradas: AttendanceRate y StudyHours. La nota solo construye etiquetas históricas y, si se aporta al dashboard, se muestra por separado: no entra al modelo ni reemplaza su predicción.

## Instalación y ejecución

Consulta [GUIA_RIESGO70.md](GUIA_RIESGO70.md) para el paso a paso de Git Bash, entrenamiento, pruebas, evaluación opcional y ejecución.

## Versionado

Nuevo modelo: `modelos/semaforo_riesgo70.joblib`. Nuevos reportes: `reports/riesgo70/`. El modelo y reportes anteriores de 60/70 permanecen intactos como históricos; sus métricas NO describen este problema binario. No se reetiqueta el antiguo joblib para simular un reentrenamiento.

El flujo vigente es `python -m src.finalize_risk_math`. Entrena solo con la muestra de train y reporta validación, sin abrir test. La evaluación de test es una orden separada y explícita. Los antiguos scripts exploratorios de análisis/comparación son históricos y no son el punto de entrada de esta versión; no se garantiza su compatibilidad con la nueva definición binaria.

## Uso

CSV UTF-8 separado por comas, decimales con punto. Obligatorios: AttendanceRate (0.70–1.00), StudyHours (0–4). Opcionales: registro_id único, Grade, TestScore_Math (0–100). Límite 10 MB y 50000 filas. No cargar datos personales reales. La aplicación separa datos inválidos, permite filtros y exportación y conserva seguimiento solo durante la sesión.

Orden: riesgo primero y luego score_en_riesgo descendente. Scores no calibrados; no equivalen a probabilidad fiable de necesitar apoyo. No relegar una nota baja observada porque la estimación sea verde.

## Límites

Sin fechas, validación institucional ni impacto educativo medido. No demuestra causalidad ni anticipación. Verde no significa ausencia de necesidades. Decisión final docente; no usar para sanciones ni negar apoyos. No incluye autenticación ni persistencia segura. Cargar únicamente joblib de confianza.
