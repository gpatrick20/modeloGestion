from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'modelos'/'semaforo_logistica_balanceada.joblib'
LABELS=['en_riesgo','medio_riesgo','sin_riesgo']
DISPLAY={'en_riesgo':'En riesgo','medio_riesgo':'Medio riesgo','sin_riesgo':'Sin alerta según criterio'}
FEATURES=['AttendanceRate','StudyHours']

def load_bundle():
    if not MODEL.exists():raise FileNotFoundError(f'Falta el modelo local: {MODEL}')
    bundle=joblib.load(MODEL)
    if bundle.get('features')!=FEATURES or bundle.get('cuts')!=[60,70]:
        raise ValueError('El modelo no tiene las entradas o cortes esperados.')
    if set(bundle['model'].classes_)!=set(LABELS):raise ValueError('Clases del modelo inesperadas.')
    return bundle

def classify(frame,bundle):
    missing=set(FEATURES)-set(frame.columns)
    if missing:raise ValueError(f'Faltan columnas: {sorted(missing)}')
    if frame.empty:raise ValueError('El CSV no tiene registros.')
    # Whitelist columns; never trust uploaded categories, scores or priorities.
    cols=[c for c in ['registro_id','Grade','TestScore_Math']+FEATURES if c in frame]
    df=frame[cols].copy().reset_index(drop=True)
    df.insert(0,'fila_archivo',np.arange(1,len(df)+1))
    if 'registro_id' not in df:
        df['registro_id']=[f'REG-{i:06d}' for i in range(1,len(df)+1)]
    reasons=pd.Series('',index=df.index,dtype='object')
    ids=df.registro_id.astype('string').str.strip()
    bad_id=ids.isna()|ids.eq('')|ids.duplicated(keep=False)
    reasons.loc[bad_id]+='Identificador vacío o duplicado; '
    df['registro_id']=ids
    for col,lo,hi in [('AttendanceRate',.7,1),('StudyHours',0,4)]:
        df[col]=pd.to_numeric(df[col],errors='coerce')
        bad=~(np.isfinite(df[col]) & df[col].between(lo,hi))
        reasons.loc[bad]+=f'{col} faltante o fuera de [{lo}, {hi}]; '
    ok=reasons.eq('')
    df['categoria_estimada']='no_clasificable'
    df['detalle_validacion']=reasons.str.rstrip('; ')
    for c in LABELS:df[f'score_{c}']=np.nan
    if ok.any():
        model=bundle['model'];x=df.loc[ok,FEATURES]
        df.loc[ok,'categoria_estimada']=model.predict(x)
        proba=model.predict_proba(x)
        for i,c in enumerate(model.classes_):df.loc[ok,f'score_{c}']=proba[:,i]
    df['semaforo']=df.categoria_estimada.map({**DISPLAY,'no_clasificable':'No clasificable'})
    df['orden_prioridad']=df.categoria_estimada.map({'en_riesgo':0,'medio_riesgo':1,'sin_riesgo':2,'no_clasificable':3})
    df['asistencia_pct']=df.AttendanceRate*100
    if 'TestScore_Math' in df:
        score=pd.to_numeric(df.TestScore_Math,errors='coerce')
        usable=np.isfinite(score)&score.between(0,100)
        df['categoria_observada']='Nota ausente o inválida'
        df.loc[usable,'categoria_observada']=np.where(score[usable]<60,DISPLAY['en_riesgo'],np.where(score[usable]<70,DISPLAY['medio_riesgo'],DISPLAY['sin_riesgo']))
        df['discrepancia_nota_modelo']=usable & ok & df.categoria_observada.ne(df.semaforo)
    df['base_estimacion']='Modelo: asistencia y estudio; no identifica causas individuales'
    return df.sort_values(['orden_prioridad','score_en_riesgo','fila_archivo'],ascending=[True,False,True],na_position='last').reset_index(drop=True)

def csv_bytes(df):
    safe=df.copy()
    for c in safe.select_dtypes(include=['object','string']).columns:
        safe[c]=safe[c].map(lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')
