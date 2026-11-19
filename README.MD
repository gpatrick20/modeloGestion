# Guía de instalación y uso — Semáforo de matemática

Esta guía explica cómo preparar el entorno local, iniciar el dashboard y clasificar un archivo de registros.

> Prototipo académico con datos sintéticos. Las categorías son estimaciones para revisión docente, no diagnósticos ni pronósticos futuros validados. El profesor debe contrastar las alertas con la información disponible.

## 1. Requisitos previos

- Git, para clonar el repositorio.
- Python instalado y disponible desde la terminal.
- Acceso al repositorio y espacio para descargar sus archivos.
- Modelo local `modelos/semaforo_logistica_balanceada.joblib`.

Para máxima compatibilidad, utilizar la misma versión de Python y de librerías con que se entrenó el modelo. Este proyecto no aporta una versión de Python fijada: si reconstruyes el entorno, consulta al equipo autor y revisa los metadatos del modelo en `reports/semaforo_final/protocolo_previo_test.json`.

Si existe `requirements-lock.txt`, contiene las versiones registradas del entorno del autor; si no existe, `requirements.txt` especifica rangos de dependencias, no una reproducción exacta garantizada.

**Seguridad:** cargar solamente el modelo joblib generado por el equipo o proveniente de una fuente de confianza. No descargar y cargar modelos desconocidos.

## 2. Descargar el proyecto

### Opción A: clonar con Git Bash

Sustituye la URL de ejemplo por la URL real del repositorio:

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git ModeloRiesgo
cd ModeloRiesgo
```

No ejecutes este comando dentro de una copia existente del mismo proyecto; en ese caso, abre directamente la carpeta que ya tienes.

### Opción B: descargar ZIP

1. En GitHub, seleccionar **Code → Download ZIP**.
2. Extraer el contenido.
3. Abrir una terminal en la carpeta que contiene `app_semaforo.py` y `requirements.txt`.

El archivo `app_semaforo.py` debe estar en la raíz del proyecto, no dentro de una subcarpeta `app`.

## 3. Comprobar Python

En Git Bash o PowerShell:

```bash
python --version
```

Si Windows no reconoce `python`, prueba:

```bash
py --version
```

Si solo funciona `py`, úsalo para crear el entorno en el siguiente paso. Después de activarlo, utiliza `python` para los comandos del proyecto.

Si ninguno funciona, instala Python y vuelve a abrir la terminal antes de continuar.

## 4. Crear y activar el entorno virtual

Crear el entorno solo la primera vez:

```bash
python -m venv .venv
```

Alternativa en Windows, si utilizaste el lanzador `py`:

```bash
py -m venv .venv
```

No es necesario ejecutar ambas opciones.

### Activar en Git Bash (Windows)

```bash
source .venv/Scripts/activate
```

### Activar en PowerShell (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Activar en Linux o macOS

```bash
source .venv/bin/activate
```

La compatibilidad del modelo y sus dependencias debe verificarse también en esos sistemas; la ejecución del proyecto fue realizada localmente en Windows.

Normalmente la terminal mostrará `(.venv)` al principio de la línea. Puedes comprobar el ejecutable seleccionado:

```bash
python -c "import sys; print(sys.executable)"
```

La ruta debe corresponder al entorno `.venv` del proyecto.

## 5. Instalar dependencias

Con el entorno activo, desde la raíz:

```bash
python -m pip install -r requirements.txt
```

Si el repositorio incluye un `requirements-lock.txt` confirmado por el equipo como perteneciente al entorno funcional y compatible con tu Python, puedes utilizarlo en lugar del archivo anterior:

```bash
python -m pip install -r requirements-lock.txt
```

No es necesario instalar ambos consecutivamente. Para verificar dependencias declaradas:

```bash
python -m pip check
```

Si el entorno ya funciona, no actualices sus paquetes solo para ejecutar el dashboard.

## 6. Archivos necesarios para abrir la aplicación

### Obligatorios

```text
ModeloRiesgo/
├── app_semaforo.py
├── requirements.txt
├── src/
│   └── semaforo_service.py
└── modelos/
    └── semaforo_logistica_balanceada.joblib
```

Conservar `src/__init__.py` si está presente.

### Para mostrar resultados y contexto

```text
reports/
├── semaforo_final/
│   ├── resumen_test.csv
│   ├── metricas_test_por_clase.csv
│   └── matriz_test.csv
└── diagnostico_matematica/
    ├── resumen_matematica.csv
    ├── matematica_por_AttendanceRate.csv
    ├── matematica_por_StudyHours.csv
    ├── matematica_por_factor.csv
    └── cruce_asistencia_estudio/
        └── mapa_asistencia_estudio.png
```

El modelo es obligatorio para clasificar. Los reportes complementan las pestañas de evaluación y diagnóstico; su ausencia no debe resolverse entrenando de nuevo.

**No se necesitan `train.csv`, `validation.csv`, `test.csv` ni la caché del EDA para abrir la demo y clasificar un CSV nuevo.** Los originales son necesarios para reproducir el procesamiento y la evaluación, lo que está fuera del flujo de uso de esta guía.

## 7. Iniciar el dashboard

Con el entorno activo y desde la raíz:

```bash
python -m streamlit run app_semaforo.py
```

La terminal indicará una dirección junto a **Local URL**, normalmente:

```text
http://localhost:8501
```

Abre la dirección que muestre tu terminal. El puerto puede cambiar si ya está ocupado.

- Mantén la terminal abierta mientras utilizas la aplicación.
- Para detener el servidor, presiona `Ctrl+C` en esa terminal.
- Subir el código a GitHub no inicia ni publica automáticamente el dashboard en Internet.

## 8. Primera prueba: cinco perfiles ficticios

1. Abrir la pestaña **Lista de revisión**.
2. Dejar seleccionada la opción **Cinco perfiles ficticios**.
3. Revisar los contadores y la tabla coloreada.
4. Descargar la plantilla si deseas preparar otro archivo.

Estos perfiles incluyen asistencia y estudio. No muestran nota real porque no contienen `TestScore_Math`.

Son ejemplos manuales para comprobar el flujo de la aplicación. No representan personas reales ni constituyen una evaluación independiente de desempeño; no se garantiza que sus combinaciones sean únicas respecto al entrenamiento.

## 9. Preparar un CSV propio

### Columnas obligatorias

| Columna exacta | Descripción | Valores válidos |
|---|---|---|
| `AttendanceRate` | Fracción de días asistidos | 0.70 a 1.00 |
| `StudyHours` | Horas promedio de estudio por día | 0 a 4 |

### Columnas opcionales

| Columna | Uso |
|---|---|
| `registro_id` | Código único dentro del lote; evitar nombres o datos personales |
| `Grade` | Grado; permite filtrar los valores aportados |
| `TestScore_Math` | Nota observada 0–100; se muestra, pero no entra al modelo |

La columna `Grade` se muestra como contexto; no es una entrada del clasificador y la interfaz no certifica la validez académica de sus valores.

### Ejemplo completo ilustrativo

```csv
registro_id,Grade,AttendanceRate,StudyHours,TestScore_Math
DEMO-001,9,0.76,0.4,45
DEMO-002,10,0.83,0.7,63
DEMO-003,11,0.88,1.0,72
DEMO-004,12,0.93,1.4,85
DEMO-005,12,0.97,1.8,94
```

### Reglas de formato

- CSV codificado en UTF-8.
- Separador: coma.
- Decimales: punto.
- Nombres de columnas exactamente como aparecen arriba.
- Asistencia como fracción: `0.85`, no `85`.
- Máximo de esta interfaz: 10 MB y 50.000 registros por lote.
- Si aportas `registro_id`, sus valores deben ser únicos y no estar vacíos.
- Otras columnas se ignoran en el procesamiento del dashboard.
- No cargar nombres, correos, documentos de identidad ni datos personales reales en esta demo.

Un archivo exportado con punto y coma por una hoja de cálculo debe convertirse al formato indicado antes de cargarlo.

## 10. Cargar el archivo

1. Entrar en **Lista de revisión**.
2. Seleccionar **Cargar CSV**.
3. Elegir el archivo desde el equipo.
4. Revisar mensajes de validación y contadores.
5. Consultar la tabla y sus filtros.

Si conservas el ejemplo del proyecto, puedes cargar:

```text
data/prueba_dashboard_200.csv
```

Este archivo contiene una selección de registros para probar la interfaz; no es una evaluación nueva independiente.

No necesitas reiniciar Streamlit para cambiar de CSV.

## 11. Interpretar el semáforo

El modelo utiliza asistencia y estudio para estimar una categoría que fue definida históricamente a partir de matemática:

| Categoría | Significado operativo | Etiqueta histórica de entrenamiento |
|---|---|---|
| Rojo — En riesgo | Prioridad alta de revisión estimada | Nota menor de 60 |
| Amarillo — Medio riesgo | Seguimiento estimado | Nota desde 60 hasta menos de 70 |
| Verde — Sin alerta según criterio | No activa esta alerta estimada | Nota desde 70 |
| No clasificable | Corregir datos antes de clasificar | Sin predicción |

Los cortes son una convención del prototipo, no una norma universal de riesgo. **Verde no significa ausencia de dificultades.**

### Orden y scores

La lista se ordena por rojo, amarillo y verde. Dentro de cada categoría, usa `score_en_riesgo` descendente. Este score es una salida no calibrada del modelo: no interpretarlo como porcentaje fiable de necesidad de apoyo.

La tabla muestra hasta 500 registros; las descargas incluyen todos los que correspondan al filtro o lote indicado.

### Registros no clasificables

Se separan para revisión de datos, por ejemplo:

- Asistencia o estudio ausentes, no numéricos o fuera de rango.
- Identificadores aportados vacíos o duplicados.

Si no se aporta `registro_id`, la aplicación genera códigos de fila para ese lote. No son identificadores estables de alumnos.

Los no clasificables no se consideran verdes. Para revisarlos, abre su bloque específico o descarga el lote completo; no forman parte de las tres categorías seleccionables del semáforo.

## 12. Dónde ver la nota de matemática

La nota aparece solo cuando el CSV incluye `TestScore_Math`. La tabla añade:

| Columna | Contenido |
|---|---|
| `TestScore_Math` | Nota registrada aportada en el archivo |
| `categoria_observada` | Categoría calculada directamente desde esa nota |
| `discrepancia_nota_modelo` | Si difiere de la categoría estimada, cuando ambas son válidas |

Desplaza horizontalmente la tabla si una columna no se ve.

La nota no se utiliza como entrada del clasificador ni modifica su predicción. Una nota opcional inválida se marca como no disponible en su categoría observada; no impide la estimación si las entradas y el identificador son válidos.

**Limitación actual:** la lista se ordena por estimación incluso si existe nota. El docente debe revisar las discrepancias y no relegar una nota baja observada porque el modelo indique verde. Si solo se requiere clasificar una nota conocida por cortes, no hace falta un modelo: esa categoría observada se calcula directamente.

## 13. Filtrar y exportar

Filtros disponibles:

- Categorías estimadas.
- Grados, si se incluyeron.
- Estado de seguimiento.

Botones de descarga:

- **Exportar lista filtrada:** todos los registros de la selección actual, no solo los 500 visibles.
- **Exportar lote completo con validaciones:** todas las filas, incluidos no clasificables y detalles de validación.

El archivo descargado conserva columnas útiles para revisar la estimación y los datos. Guárdalo en una ubicación adecuada para la demostración.

## 14. Registrar seguimiento

1. Desplegar **Registrar seguimiento durante esta sesión**.
2. Introducir el número de `fila_archivo`, no la posición que ocupa la fila después de ordenar.
3. Seleccionar el nuevo estado.
4. Pulsar **Aplicar estado en esta sesión**.
5. Exportar el lote completo si necesitas conservar una copia.

Estados: pendiente de revisión, revisado y en acompañamiento.

**El seguimiento solo vive en la sesión.** No se guarda en una base de datos. Cerrar o perder la sesión puede eliminarlo. Volver a cargar un CSV exportado no restaura automáticamente estos estados: la aplicación ignora esa columna de entrada.

## 15. Consultar evaluación y diagnóstico

### Evaluación del modelo

Muestra reportes finales de test y metadatos del modelo. No vuelve a entrenarlo ni modifica sus parámetros.

Resultados principales documentados:

- Accuracy: 72,20 %.
- Recall rojo: 74,22 %.
- Precisión roja: 40,46 %.
- Cobertura de rojos mediante rojo o amarillo: 98,61 %.
- Registros con alguna alerta: 41,54 %.

El 98,61 % **no** es precisión general. La precisión roja limitada significa que muchas alertas requieren verificación. La aplicación no garantiza detectar todos los casos.

### Diagnóstico y alcance

Permite consultar tablas agregadas, factores y el mapa descriptivo si sus reportes están incluidos. Las asociaciones no demuestran causalidad ni el efecto de aumentar asistencia o estudio.

## 16. Solución de problemas

### “File does not exist: app_semaforo.py”

Verifica que estás en la raíz:

```bash
pwd
ls
```

En PowerShell puedes utilizar `Get-Location` y `Get-ChildItem`.

El archivo debe estar directamente en esa carpeta. Después ejecuta:

```bash
python -m streamlit run app_semaforo.py
```

### “No module named streamlit” u otra dependencia

Activa `.venv`, comprueba `sys.executable` e instala las dependencias en ese entorno:

```bash
python -m pip install -r requirements.txt
```

### PowerShell bloquea Activate.ps1

Como alternativa, abre Git Bash en la raíz y activa con:

```bash
source .venv/Scripts/activate
```

No necesitas cambiar la política de seguridad del equipo para utilizar esta alternativa.

### No se encuentra el modelo

Comprueba la ruta exacta:

```text
modelos/semaforo_logistica_balanceada.joblib
```

No lo sustituyas por un modelo anterior ni descargues uno desconocido. Solicita el artefacto final al equipo. No es necesario reentrenar para corregir una ubicación equivocada.

### Error o advertencia de versiones al cargar el modelo

Utiliza un entorno compatible con el entrenamiento. Consulta el protocolo y las versiones del equipo. No ignores el aviso ni asumas que actualizar scikit-learn solucionará el problema.

### Faltan columnas o todos los registros son inválidos

Revisa nombres exactos, separador coma, decimales con punto, asistencia en escala 0.70–1.00 y estudio en 0–4. Revisa también identificadores duplicados.

### No aparece la nota

Añade `TestScore_Math` al CSV. Los cinco perfiles predeterminados no contienen esa columna.

### Faltan reportes de evaluación

Recupera los archivos de `reports/semaforo_final/` desde la versión final del repositorio. No vuelvas a ejecutar test solo para llenar una pestaña.

### El navegador no abre o el puerto está ocupado

Consulta la URL que muestra la terminal. Si necesitas elegir otro puerto:

```bash
python -m streamlit run app_semaforo.py --server.port 8502
```

Abre la dirección que indique esa ejecución. Comprueba que el servidor no se haya detenido.

## 17. Uso responsable y límites

- No utilizar la demo para sancionar, excluir o negar apoyos.
- No interpretar la salida como diagnóstico de salud emocional ni explicación causal.
- No afirmar predicción futura validada: el dataset no documenta fechas.
- No interpretar la exclusión de atributos sensibles como prueba de ausencia de sesgos.
- No publicar la demo con información real: no incluye autenticación, permisos ni persistencia segura.
- El ahorro de tiempo y la mejora del aprendizaje no se han medido.
- La decisión final de revisión o acompañamiento corresponde al docente, contrastando evidencia disponible.

## 18. Inicio rápido para las siguientes sesiones

### Git Bash en Windows

Desde la raíz del proyecto:

```bash
source .venv/Scripts/activate
python -m streamlit run app_semaforo.py
```

### PowerShell en Windows

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run .\app_semaforo.py
```

Al terminar, `Ctrl+C` detiene Streamlit. Para salir del entorno:

```bash
deactivate
```

No ejecutar los scripts de preparación, entrenamiento o evaluación para abrir el dashboard. Esos scripts documentan el proceso de desarrollo y se describen en el README técnico.
