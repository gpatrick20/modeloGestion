import hashlib
import io
import pandas as pd
import streamlit as st
from src.semaforo_service import load_bundle, classify, csv_bytes, DISPLAY
from src.risk_math_common import MODEL, REPORT, ROOT

st.set_page_config(page_title='Riesgo académico en matemática', layout='wide')
st.title('Revisión de riesgo académico en matemática')
st.warning('Prototipo académico con datos sintéticos. No es diagnóstico ni alerta temporal validada. No cargar datos personales reales.')
st.caption('Dos categorías: rojo para la etiqueta histórica TestScore_Math < 70; verde desde 70. El modelo estima con asistencia y estudio, no con la nota. Scores no calibrados.')


@st.cache_resource
def get_bundle(stamp):
    return load_bundle()


try:
    bundle = get_bundle(MODEL.stat().st_mtime_ns if MODEL.exists() else 0)
except Exception as exc:
    st.error(str(exc))
    st.info('Desde la raíz: python -m src.finalize_risk_math. Consulta GUIA_RIESGO70.md. No cargues archivos joblib de fuentes desconocidas.')
    st.stop()

main, evaluation, context = st.tabs(['Lista de revisión', 'Evaluación', 'Diagnóstico y alcance'])
with main:
    template = pd.DataFrame({'registro_id': [f'DEMO-{i:03d}' for i in range(1, 6)], 'AttendanceRate': [.76, .83, .88, .93, .97], 'StudyHours': [.4, .7, 1, 1.4, 1.8]})
    st.download_button('Descargar plantilla ficticia', csv_bytes(template), 'plantilla_semaforo.csv', 'text/csv')
    st.caption('CSV UTF-8, comas, decimales con punto. AttendanceRate: 0.70–1.00; StudyHours: 0–4. Opcionales: registro_id único, Grade y TestScore_Math 0–100. La demo no evalúa rendimiento.')
    mode = st.radio('Origen', ['Cinco perfiles ficticios', 'Cargar CSV'], horizontal=True)
    frame = None
    blob = b''
    if mode == 'Cinco perfiles ficticios':
        frame = template
        blob = template.to_csv(index=False).encode()
    else:
        uploaded = st.file_uploader('CSV, máximo 10 MB y 50000 registros', type=['csv'])
        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error('Divide el archivo: supera 10 MB.')
            else:
                blob = uploaded.getvalue()
                try:
                    header = pd.read_csv(io.BytesIO(blob), nrows=0, encoding='utf-8-sig')
                    wanted = [c for c in ['registro_id', 'Grade', 'TestScore_Math', 'AttendanceRate', 'StudyHours'] if c in header.columns]
                    frame = pd.read_csv(io.BytesIO(blob), usecols=wanted, nrows=50001, dtype={'registro_id': 'string'}, encoding='utf-8-sig')
                    if len(frame) > 50000:
                        st.error('Divide el archivo: supera 50000 registros.')
                        frame = None
                except Exception as exc:
                    st.error(f'No se pudo leer el CSV: {exc}')
    if frame is not None:
        try:
            result = classify(frame, bundle)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        if 'registro_id' not in frame:
            st.info('Los códigos generados identifican filas de este lote, no estudiantes de forma estable.')
        key = hashlib.sha256(blob).hexdigest()
        states = st.session_state.setdefault('seguimiento_por_lote', {}).setdefault(key, {})
        result['seguimiento'] = result.fila_archivo.map(states).fillna('Pendiente de revisión')
        for box, code, name in zip(st.columns(3), ['en_riesgo', 'sin_riesgo', 'no_clasificable'], ['En riesgo', 'Sin alerta según criterio', 'No clasificables']):
            box.metric(name, int(result.categoria_estimada.eq(code).sum()))
        invalid = result[result.categoria_estimada.eq('no_clasificable')]
        if len(invalid):
            with st.expander('Corregir datos: no clasificables', expanded=True):
                st.dataframe(invalid[['fila_archivo', 'registro_id', 'detalle_validacion']], hide_index=True)
        if 'categoria_observada' in result:
            st.info('Nota y categoría observadas se muestran por separado. No descartes una nota baja porque el modelo estime verde. El orden se basa en la estimación, no en la nota.')
        chosen = st.multiselect('Categorías estimadas', list(DISPLAY.values()), default=list(DISPLAY.values()))
        view = result[result.semaforo.isin(chosen)].copy()
        if 'Grade' in result:
            selected = st.multiselect('Grados (vacío = todos)', sorted(result.Grade.dropna().astype(str).unique()))
            if selected:
                view = view[view.Grade.astype(str).isin(selected)]
        statuses = st.multiselect('Seguimiento (vacío = todos)', ['Pendiente de revisión', 'Revisado', 'En acompañamiento'])
        if statuses:
            view = view[view.seguimiento.isin(statuses)]
        st.caption('Orden: rojo y verde, luego score_en_riesgo descendente. Se muestran hasta 500 filas; la exportación incluye todas las filtradas.')
        colors = {'En riesgo': 'background-color: #7f1d1d; color: white', 'Sin alerta según criterio': 'background-color: #166534; color: white'}
        columns = [c for c in ['fila_archivo', 'registro_id', 'semaforo', 'asistencia_pct', 'StudyHours', 'Grade', 'TestScore_Math', 'categoria_observada', 'discrepancia_nota_modelo', 'score_en_riesgo', 'seguimiento'] if c in view]
        styled = view[columns].head(500).style.apply(lambda col: [colors.get(v, '') for v in col], subset=['semaforo'])
        st.dataframe(styled, hide_index=True, use_container_width=True)
        st.download_button('Exportar lista filtrada', csv_bytes(view), 'lista_priorizada.csv', 'text/csv')
        st.download_button('Exportar lote completo con validaciones', csv_bytes(result), 'lote_completo.csv', 'text/csv')
        with st.expander('Registrar seguimiento durante esta sesión'):
            st.warning('No hay persistencia en base de datos. Exporta para conservar una copia; recargar el CSV no restaura estados.')
            row = st.number_input('fila_archivo', min_value=1, max_value=len(frame), step=1)
            state = st.selectbox('Estado', ['Pendiente de revisión', 'Revisado', 'En acompañamiento'])
            if st.button('Aplicar estado'):
                states[int(row)] = state
                st.rerun()
with evaluation:
    st.info('Solo reportes de riesgo70. Los resultados históricos 60/70 no son métricas de este modelo. Aquí no se entrena.')
    for partition in ['validation', 'test']:
        st.subheader(partition)
        for name in [f'resumen_{partition}.csv', f'metricas_{partition}_por_clase.csv', f'matriz_{partition}.csv']:
            path = REPORT / name
            if path.exists():
                st.write(name)
                st.dataframe(pd.read_csv(path), hide_index=True)
            else:
                st.caption(f'No disponible: {name}')
    st.warning('El test histórico, si se evalúa otra vez, es reutilizado. No usarlo para ajustar ni seleccionar modelos.')
    st.json({k: bundle.get(k) for k in ['model_name', 'features', 'cuts', 'training_rows', 'scope']})
with context:
    st.write('El EDA histórico permanece en reports/diagnostico_matematica. Asociaciones no demuestran causalidad ni efecto de intervenciones.')
    folder = ROOT / 'reports' / 'diagnostico_matematica'
    for name in ['resumen_matematica.csv', 'matematica_por_AttendanceRate.csv', 'matematica_por_StudyHours.csv', 'matematica_por_factor.csv']:
        path = folder / name
        if path.exists():
            with st.expander(name):
                st.dataframe(pd.read_csv(path), hide_index=True)
    st.write('Datos sintéticos sin fechas. Sin validación institucional, evaluación de bienestar ni impacto educativo demostrado. Sin autenticación ni persistencia segura. Verde no significa ausencia de necesidades. El docente decide el acompañamiento; no usar para sancionar o negar apoyos.')
    st.markdown('[Dataset Student Performance](https://www.kaggle.com/datasets/neuralsorcerer/student-performance)')
