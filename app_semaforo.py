from pathlib import Path
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from src.semaforo_service import load_bundle,classify,csv_bytes,ROOT,LABELS,DISPLAY

st.set_page_config(page_title='Semáforo de acompañamiento',layout='wide')
st.title('Semáforo de acompañamiento en matemática')
st.warning('Prototipo académico con datos sintéticos. Categorías estimadas para revisión docente; no diagnóstico integral ni alerta temporal validada.')
st.caption('Entradas: asistencia y horas diarias de estudio. Etiqueta histórica: matemática <60, 60–<70 y ≥70. Los scores no están calibrados como probabilidades fiables.')

@st.cache_resource
def get_bundle(stamp):
    return load_bundle()
modelpath=ROOT/'modelos'/'semaforo_logistica_balanceada.joblib'
try:
    bundle=get_bundle(modelpath.stat().st_mtime_ns if modelpath.exists() else 0)
except Exception as exc:
    st.error(str(exc));st.info('Usa únicamente el modelo joblib generado por tus scripts, nunca uno subido por terceros.');st.stop()

main,evaltab,context=st.tabs(['Lista de revisión','Evaluación del modelo','Diagnóstico y alcance'])
with main:
    st.markdown('### Cargar registros')
    template=pd.DataFrame({'registro_id':['DEMO-001','DEMO-002','DEMO-003','DEMO-004','DEMO-005'],'AttendanceRate':[.76,.83,.88,.93,.97],'StudyHours':[.4,.7,1,1.4,1.8]})
    st.download_button('Descargar plantilla de cinco perfiles ficticios',csv_bytes(template),'plantilla_semaforo.csv','text/csv')
    mode=st.radio('Origen',['Cinco perfiles ficticios','Cargar CSV'],horizontal=True)
    st.caption('CSV separado por comas y decimales con punto. AttendanceRate: 0.70–1.00, no 70–100. StudyHours: 0–4 horas/día. Opcionales: registro_id, Grade y TestScore_Math. No subir nombres ni datos personales reales a esta demo.')
    frame=None;blob=None
    if mode=='Cinco perfiles ficticios':
        frame=template;blob=template.to_csv(index=False).encode()
    else:
        uploaded=st.file_uploader('Archivo CSV (máximo 10 MB y 50000 registros)',type=['csv'])
        if uploaded is not None:
            if uploaded.size>10*1024*1024:st.error('El archivo supera 10 MB. Divide el lote.')
            else:
                blob=uploaded.getvalue()
                try:
                    header=pd.read_csv(io.BytesIO(blob),nrows=0)
                    wanted=[c for c in ['registro_id','Grade','TestScore_Math','AttendanceRate','StudyHours'] if c in header.columns]
                    frame=pd.read_csv(io.BytesIO(blob),usecols=wanted,nrows=50001,dtype={'registro_id':'string'},encoding='utf-8-sig')
                    if len(frame)>50000:st.error('El lote supera 50000 registros. Divide el archivo.');frame=None
                except Exception as exc:st.error(f'No se pudo leer el CSV: {exc}')
    if frame is not None:
        try:result=classify(frame,bundle)
        except Exception as exc:st.error(str(exc));st.stop()
        if 'registro_id' not in frame:st.info('Se generaron códigos de fila solo para este lote. No son identificadores estables de estudiantes.')
        key=hashlib.sha256(blob).hexdigest()
        states=st.session_state.setdefault('seguimiento_por_lote',{}).setdefault(key,{})
        result['seguimiento']=result.fila_archivo.map(states).fillna('Pendiente de revisión')
        boxes=st.columns(4)
        for box,label,keyclass in zip(boxes,['En riesgo','Medio riesgo','Sin alerta según criterio','No clasificables'],LABELS+['no_clasificable']):
            box.metric(label,int((result.categoria_estimada==keyclass).sum()))
        invalid=result[result.categoria_estimada=='no_clasificable']
        if len(invalid):
            st.error(f'{len(invalid)} registros requieren corregir datos; no reciben verde ni una predicción.')
            with st.expander('Ver registros no clasificables',expanded=True):st.dataframe(invalid[['fila_archivo','registro_id','detalle_validacion']],hide_index=True)
        if 'categoria_observada' in result:
            st.info('La nota registrada se muestra por separado y NO entra al modelo. Si existe una dificultad observada, no la descartes porque la estimación sea verde. Esta tabla está ordenada por categoría estimada, no por nota.')
        chosen=st.multiselect('Categorías estimadas',list(DISPLAY.values()),default=list(DISPLAY.values()))
        view=result[result.semaforo.isin(chosen)].copy()
        if 'Grade' in result:
            grades=sorted(result.Grade.dropna().astype(str).unique())
            selected=st.multiselect('Grados (vacío = todos)',grades)
            if selected:view=view[view.Grade.astype(str).isin(selected)]
        statuses=st.multiselect('Estado de seguimiento (vacío = todos)',['Pendiente de revisión','Revisado','En acompañamiento'])
        if statuses:view=view[view.seguimiento.isin(statuses)]
        st.caption('Orden: rojo, amarillo y verde; dentro de cada categoría, score de en_riesgo descendente. No es una prioridad clínica ni una probabilidad calibrada.')
        columns=['fila_archivo','registro_id','semaforo','asistencia_pct','StudyHours']
        columns += [c for c in ['Grade','TestScore_Math','categoria_observada','discrepancia_nota_modelo'] if c in view]
        columns+=['score_en_riesgo','seguimiento']
        colors={'En riesgo':'background-color: #7f1d1d; color: white','Medio riesgo':'background-color: #facc15; color: black','Sin alerta según criterio':'background-color: #166534; color: white'}
        styled=view[columns].head(500).style.applymap(lambda v:colors.get(v,''),subset=['semaforo']).format({'asistencia_pct':'{:.1f}','StudyHours':'{:.2f}','score_en_riesgo':'{:.3f}'},na_rep='—')
        st.dataframe(styled,hide_index=True,use_container_width=True)
        st.caption(f'{len(view):,} registros filtrados. La tabla muestra hasta 500; la exportación incluye todos los filtrados.')
        st.download_button('Exportar lista filtrada',csv_bytes(view),'lista_priorizada.csv','text/csv')
        st.download_button('Exportar lote completo con validaciones',csv_bytes(result),'lote_completo.csv','text/csv')
        with st.expander('Registrar seguimiento durante esta sesión'):
            st.warning('El seguimiento solo vive en la sesión del navegador. Exporta el lote completo para conservar una copia. No se guarda en una base de datos ni se restaura automáticamente al recargar el CSV.')
            row=st.number_input('Número de fila del archivo',min_value=1,max_value=len(frame),step=1)
            state=st.selectbox('Nuevo estado',['Pendiente de revisión','Revisado','En acompañamiento'])
            if st.button('Aplicar estado en esta sesión'):
                states[int(row)]=state;st.rerun()

with evaltab:
    st.subheader('Resultados fijos de test; aquí no se entrena ni se ajusta el modelo')
    folder=ROOT/'reports'/'semaforo_final'
    for filename in ['resumen_test.csv','metricas_test_por_clase.csv','matriz_test.csv']:
        path=folder/filename
        if path.exists():
            st.markdown(f'#### {filename}')
            try:st.dataframe(pd.read_csv(path),hide_index=True,use_container_width=True)
            except Exception as exc:st.error(str(exc))
        else:st.info(f'Reporte no encontrado: {filename}')
    st.write('Recall de rojo: fracción de casos rojos reales detectados como rojos. Precisión de rojo: fracción de alertas rojas que son rojas según la etiqueta histórica. Rojo o amarillo NO equivale a clasificación correcta de rojo.')
    st.write('La selección balanceada buscó reducir casos de riesgo alto enviados a verde, aceptando más alertas. No se garantiza una reducción de problemas educativos ni detección futura.')
    st.json({k:bundle.get(k) for k in ['model_name','features','cuts','training_rows','scope']})

with context:
    st.subheader('Diagnóstico agregado')
    folder=ROOT/'reports'/'diagnostico_matematica'
    for file in ['resumen_matematica.csv','matematica_por_AttendanceRate.csv','matematica_por_StudyHours.csv']:
        p=folder/file
        if p.exists():
            with st.expander(file):st.dataframe(pd.read_csv(p),hide_index=True,use_container_width=True)
    p=folder/'matematica_por_factor.csv'
    if p.exists():
        factors=pd.read_csv(p);factor=st.selectbox('Factor descriptivo',factors.variable.unique())
        st.dataframe(factors[factors.variable==factor],hide_index=True,use_container_width=True)
    p=folder/'cruce_asistencia_estudio'/'mapa_asistencia_estudio.png'
    if p.exists():st.image(str(p))
    st.markdown('### Límites y uso')
    st.write('Fuente: Student Performance, Kaggle. Datos sintéticos. Sin fechas para validar anticipación. Sin evaluación directa de bienestar emocional. Identificadores de demo no representan personas reales.')
    st.write('El profesor revisa la evidencia y decide el acompañamiento. No usar para sanciones, excluir estudiantes o negar apoyos. Sin alerta no significa ausencia de necesidades.')
    st.write('Esta demo no incluye autenticación, permisos ni persistencia segura para datos reales. No publicarla como sistema institucional sin esas medidas y una validación adecuada.')
    st.markdown('[Ficha del dataset](https://www.kaggle.com/datasets/neuralsorcerer/student-performance)')
