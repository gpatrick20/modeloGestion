import joblib
import numpy as np
import pandas as pd
from .risk_math_common import ROOT, MODEL, FEATURES, LABELS, CUTS, label

DISPLAY = {'en_riesgo': 'En riesgo', 'sin_riesgo': 'Sin alerta según criterio'}


def load_bundle():
    if not MODEL.exists():
        raise FileNotFoundError('Falta modelos/semaforo_riesgo70.joblib. Ejecuta: python -m src.finalize_risk_math')
    bundle = joblib.load(MODEL)
    if bundle.get('features') != FEATURES or bundle.get('cuts') != CUTS:
        raise ValueError('Entradas o cortes incompatibles. Se requiere el modelo de riesgo menor de 70.')
    if set(bundle['model'].classes_) != set(LABELS):
        raise ValueError('Clases incompatibles.')
    return bundle


def classify(frame, bundle):
    if frame.empty:
        raise ValueError('El CSV no tiene registros.')
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    cols = [c for c in ['registro_id', 'Grade', 'TestScore_Math'] + FEATURES if c in frame]
    df = frame[cols].copy().reset_index(drop=True)
    df.insert(0, 'fila_archivo', np.arange(1, len(df) + 1))
    if 'registro_id' not in df:
        df['registro_id'] = [f'REG-{i:06d}' for i in range(1, len(df) + 1)]
    reasons = pd.Series('', index=df.index, dtype='object')
    ids = df.registro_id.astype('string').str.strip()
    bad = ids.isna() | ids.eq('') | ids.duplicated(keep=False)
    reasons.loc[bad] += 'Identificador vacío o duplicado; '
    df['registro_id'] = ids
    for col, lo, hi in [('AttendanceRate', .7, 1), ('StudyHours', 0, 4)]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        bad = ~(np.isfinite(df[col]) & df[col].between(lo, hi))
        reasons.loc[bad] += f'{col} faltante o fuera de rango; '
    ok = reasons.eq('')
    df['detalle_validacion'] = reasons.str.rstrip('; ')
    df['categoria_estimada'] = 'no_clasificable'
    for c in LABELS:
        df[f'score_{c}'] = np.nan
    if ok.any():
        model = bundle['model']
        df.loc[ok, 'categoria_estimada'] = model.predict(df.loc[ok, FEATURES])
        probs = model.predict_proba(df.loc[ok, FEATURES])
        for i, c in enumerate(model.classes_):
            df.loc[ok, f'score_{c}'] = probs[:, i]
    df['semaforo'] = df.categoria_estimada.map({**DISPLAY, 'no_clasificable': 'No clasificable'})
    df['orden_prioridad'] = df.categoria_estimada.map({'en_riesgo': 0, 'sin_riesgo': 1, 'no_clasificable': 2})
    df['asistencia_pct'] = df.AttendanceRate * 100
    if 'TestScore_Math' in df:
        score = pd.to_numeric(df.TestScore_Math, errors='coerce')
        usable = np.isfinite(score) & score.between(0, 100)
        df['categoria_observada'] = 'Nota ausente o inválida'
        df.loc[usable, 'categoria_observada'] = pd.Series(label(score[usable]), index=score[usable].index).map(DISPLAY)
        df['discrepancia_nota_modelo'] = usable & ok & df.categoria_observada.ne(df.semaforo)
    df['base_estimacion'] = 'Asistencia y estudio; no identifica causas individuales'
    return df.sort_values(['orden_prioridad', 'score_en_riesgo', 'fila_archivo'], ascending=[True, False, True], na_position='last').reset_index(drop=True)


def csv_bytes(df):
    safe = df.copy()
    for c in safe.select_dtypes(include=['object', 'string']).columns:
        safe[c] = safe[c].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')
