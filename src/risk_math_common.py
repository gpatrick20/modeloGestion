from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'samples_math'
REPORT=ROOT/'reports'/'clasificacion_matematica'
MODEL=ROOT/'modelos'/'clasificador_riesgo_matematica.joblib'
FEATURES=['AttendanceRate','StudyHours']
TARGET='TestScore_Math'
LABELS=['en_riesgo','medio_riesgo','sin_riesgo']

def validate(frame, target=False):
    if frame.empty:
        raise ValueError('El archivo no contiene registros.')
    ranges={'AttendanceRate':(.70,1),'StudyHours':(0,4)}
    if target:
        ranges[TARGET]=(0,100)
    for col,(lo,hi) in ranges.items():
        if col not in frame:
            raise ValueError(f'Falta columna: {col}')
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
        ok=np.isfinite(frame[col]) & frame[col].between(lo,hi)
        if not ok.all():
            raise ValueError(f'{col}: {(~ok).sum()} valores invalidos. No se clasificaran como sin riesgo.')
    return frame

def label(scores):
    return np.where(scores<60,'en_riesgo',np.where(scores<70,'medio_riesgo','sin_riesgo'))
