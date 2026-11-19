"""Experimento: riesgo de matematica usando antecedentes academicos."""
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'samples_math'
REPORT = ROOT / 'reports' / 'experimento_notas_academicas'
MODEL = ROOT / 'modelos' / 'experimento_notas_academicas.joblib'
TARGET = 'TestScore_Math'
FEATURES = ['AttendanceRate', 'StudyHours', 'TestScore_Reading', 'TestScore_Science']
LABELS = ['en_riesgo', 'sin_riesgo']
CUT = 70


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(frame, target=False):
    if frame.empty:
        raise ValueError('El archivo no contiene registros.')
    ranges = {'AttendanceRate': (.70, 1), 'StudyHours': (0, 4), 'TestScore_Reading': (0, 100), 'TestScore_Science': (0, 100)}
    if target:
        ranges[TARGET] = (0, 100)
    for column, (low, high) in ranges.items():
        if column not in frame:
            raise ValueError(f'Falta columna: {column}')
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
        valid = np.isfinite(frame[column]) & frame[column].between(low, high)
        if not valid.all():
            raise ValueError(f'{column}: {(~valid).sum()} valores invalidos.')
    return frame


def label(scores):
    return np.where(scores < CUT, 'en_riesgo', 'sin_riesgo')


def metrics(matrix):
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(2), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(2), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=(precision + recall) > 0)
    total = int(matrix.sum())
    if not total:
        raise ValueError('Matriz vacia.')
    return {
        'accuracy': float(true_positive.sum() / total),
        'f1_macro': float(f1.mean()),
        'balanced_accuracy': float(recall.mean()),
        'recall_en_riesgo': float(recall[0]),
        'precision_en_riesgo': float(precision[0]),
        'en_riesgo_a_verde': int(matrix[0, 1]),
        'alertas_rojas': int(predicted[0]),
        'fraccion_total_con_alerta': float(predicted[0] / total),
    }


def save_outputs(matrix, name):
    summary = metrics(matrix)
    pd.DataFrame({'clase': LABELS, 'precision': np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(2), where=matrix.sum(axis=0) > 0), 'recall': np.divide(np.diag(matrix), matrix.sum(axis=1), out=np.zeros(2), where=matrix.sum(axis=1) > 0), 'n_real': matrix.sum(axis=1), 'n_estimado': matrix.sum(axis=0)}).to_csv(REPORT / f'metricas_{name}_por_clase.csv', index=False)
    pd.DataFrame([summary]).to_csv(REPORT / f'resumen_{name}.csv', index=False)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis('real').to_csv(REPORT / f'matriz_{name}.csv')
    return summary


def main():
    if REPORT.exists() or MODEL.exists():
        raise FileExistsError('Ya existe el experimento. No se sobrescribe; conserva la evidencia o usa otra version.')
    paths = {name: DATA / f'{name}_math.csv' for name in ['train', 'validation']}
    frames = {name: validate(pd.read_csv(path), True) for name, path in paths.items()}
    for name, frame in frames.items():
        if set(label(frame[TARGET])) != set(LABELS):
            raise ValueError(f'Faltan clases en {name}.')
    train, valid = frames['train'], frames['validation']
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    model.fit(train[FEATURES], label(train[TARGET]))
    validation_matrix = confusion_matrix(label(valid[TARGET]), model.predict(valid[FEATURES]), labels=LABELS)
    metadata = {
        'experiment': 'notas_academicas',
        'features': FEATURES,
        'target': TARGET,
        'labels': LABELS,
        'cut': CUT,
        'training_rows': len(train),
        'validation_rows': len(valid),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sklearn_version': sklearn.__version__,
        'sample_sha256': {name: sha(path) for name, path in paths.items()},
        'scope': 'Experimento de clasificacion con antecedentes academicos; no alerta temporal validada ni impacto educativo demostrado.',
        'scores': 'No calibrados; no interpretar como probabilidad fiable de necesidad de apoyo.',
    }
    REPORT.mkdir(parents=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    summary = save_outputs(validation_matrix, 'validacion')
    joblib.dump({**metadata, 'model': model}, MODEL)
    (REPORT / 'entrenamiento_completado.json').write_text(json.dumps({**metadata, 'model_sha256': sha(MODEL)}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary)
    print('Modelo experimental guardado:', MODEL)
    print('Test no abierto.')


if __name__ == '__main__':
    main()
