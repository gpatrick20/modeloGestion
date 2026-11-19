# Ejecutar la versión de riesgo menor de 70

## 1. Descargar el cambio

En Git Bash, dentro de tu copia existente:

```bash
git status
git pull --ff-only origin main
```

Si hay cambios locales, guárdalos mediante un commit propio o resuélvelos antes de actualizar. No uses reset --hard ni borres tus datos para resolver conflictos.

Si no tienes copia:

```bash
git clone https://github.com/gpatrick20/modeloGestion.git
cd modeloGestion
```

## 2. Entorno

Si ya tienes un entorno funcional, actívalo y conserva sus versiones. Para uno nuevo:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip check
```

En Linux/macOS, la activación es `source .venv/bin/activate`. Los comandos siguientes se ejecutan desde la raíz, donde está app_semaforo.py.

## 3. Datos para reentrenar

Deben existir `data/samples_math/train_math.csv` y `data/samples_math/validation_math.csv`, con AttendanceRate, StudyHours y TestScore_Math. Usa las muestras originales, sin mezclar train, validation o test. Conserva sus metadatos. Si no están descargadas, recupéralas de tus archivos locales; el repositorio puede no incluir datos voluminosos. No se crean muestras ficticias como reemplazo del entrenamiento real.

## 4. Pruebas de frontera

```bash
python -m unittest discover -s tests -p 'test_riesgo70.py'
```

Son pruebas unitarias, no una evaluación de desempeño. Este commit no incluye entrenamiento ni test ejecutados remotamente.

## 5. Entrenar y revisar validación

```bash
python -m src.finalize_risk_math
```

Genera `modelos/semaforo_riesgo70.joblib`, métricas de validación, su matriz y `reports/riesgo70/entrenamiento_completado.json`. La normalización y el modelo se ajustan solo con train. NO abre test. El criterio de riesgo es menor de 70; exactamente 70 es sin alerta según criterio.

No sobrescribe resultados existentes. Si aparecen archivos tras un fallo, revisa el error y conserva/archiva esa ejecución antes de reintentar. No borres el modelo anterior ni reports/semaforo_final: son evidencia histórica. No compares el accuracy nuevo y antiguo como si las etiquetas fueran iguales.

## 6. Abrir dashboard

```bash
python -m streamlit run app_semaforo.py
```

Abre la Local URL de la terminal. Prueba primero los cinco perfiles ficticios, luego tu CSV. Los perfiles no prueban desempeño. El seguimiento solo dura la sesión: exporta el lote completo si quieres una copia; recargar el CSV no restaura estados. Ctrl+C detiene el servidor.

## 7. Test opcional, no necesario para abrir

Solo cuando la definición y el modelo estén fijados, si quieres documentar una evaluación con `data/test.csv`:

```bash
python -m src.finalize_risk_math --evaluate-test
```

No reentrena. Guarda métricas de test y protocolo con hashes. El test ya se usó en la versión anterior: esta reutilización NO es una prueba inédita; para una confirmación independiente se necesitan datos nuevos no consultados. No usar resultados de test para seleccionar modelos, variables o cortes. No ejecutarlo repetidamente para mejorar métricas.

## 8. Versionar artefactos locales

El commit de código no contiene el nuevo modelo entrenado. Tras revisar resultados, consulta `git status`, las reglas de .gitignore y los límites de tamaño antes de añadir artefactos. No publiques datasets ni información personal. No fuerces archivos ignorados sin revisar por qué se excluyeron. El dashboard requiere el joblib nuevo; el anterior no sirve para el corte actualizado.
