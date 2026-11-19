"""Entrena riesgo70; el test requiere una orden separada y explícita."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from .risk_math_common import ROOT, DATA, REPORT, MODEL, FEATURES, TARGET, LABELS, CUTS, validate, label


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(cm):
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros(len(LABELS)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(LABELS)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(LABELS)), where=(precision + recall) > 0)
    n = int(cm.sum())
    if not n:
        raise ValueError('Evaluación vacía.')
    table = pd.DataFrame({'clase': LABELS, 'precision': precision, 'recall': recall, 'f1': f1, 'n_real': support, 'n_estimado': predicted})
    summary = {'n': n, 'accuracy': float(tp.sum() / n), 'f1_macro': float(f1.mean()), 'balanced_accuracy': float(recall.mean()), 'recall_en_riesgo': float(recall[0]), 'precision_en_riesgo': float(precision[0]), 'en_riesgo_a_verde': int(cm[0, 1]), 'fraccion_total_con_alerta': float(predicted[0] / n)}
    return table, summary


def save_metrics(cm, prefix):
    table, summary = evaluate(cm)
    table.to_csv(REPORT / f'metricas_{prefix}_por_clase.csv', index=False)
    pd.DataFrame([summary]).to_csv(REPORT / f'resumen_{prefix}.csv', index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis('real').to_csv(REPORT / f'matriz_{prefix}.csv')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-test', action='store_true', help='Evalúa explícitamente el modelo ya entrenado; no lo reajusta.')
    args = parser.parse_args()
    if args.evaluate_test:
        if not MODEL.exists() or not (REPORT / 'entrenamiento_completado.json').exists():
            raise FileNotFoundError('Primero ejecuta el entrenamiento sin argumentos.')
        if list(REPORT.glob('*test*')):
            raise FileExistsError('Ya existen resultados o protocolo de test. No se repite automáticamente; revisa y archiva antes de reintentar.')
        bundle = joblib.load(MODEL)
        if bundle.get('cuts') != CUTS or bundle.get('features') != FEATURES or set(bundle['model'].classes_) != set(LABELS):
            raise ValueError('Modelo incompatible.')
        test = ROOT / 'data' / 'test.csv'
        if not test.exists():
            raise FileNotFoundError(test)
        note = 'Test reutilizado tras cambiar la definición de clases; no es una prueba inédita. No usar para seleccionar ni ajustar modelos.'
        (REPORT / 'protocolo_test.json').write_text(json.dumps({'note': note, 'model_sha256': sha(MODEL), 'test_sha256': sha(test)}, indent=2), encoding='utf-8')
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        for chunk in pd.read_csv(test, usecols=FEATURES + [TARGET], chunksize=100000):
            chunk = validate(chunk, True)
            cm += confusion_matrix(label(chunk[TARGET]), bundle['model'].predict(chunk[FEATURES]), labels=LABELS)
        if (cm.sum(axis=1) == 0).any():
            raise ValueError('Faltan clases en test.')
        print(save_metrics(cm, 'test'))
        print(note)
        return
    if MODEL.exists() or REPORT.exists():
        raise FileExistsError('Ya existe el modelo o reports/riesgo70. No se sobrescribe. Conserva los resultados y usa otra versión para nuevos experimentos.')
    paths = {name: DATA / f'{name}_math.csv' for name in ['train', 'validation']}
    frames = {name: validate(pd.read_csv(path), True) for name, path in paths.items()}
    for name, df in frames.items():
        if set(label(df[TARGET])) != set(LABELS):
            raise ValueError(f'Faltan clases en {name}.')
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    train, valid = frames['train'], frames['validation']
    model.fit(train[FEATURES], label(train[TARGET]))
    cm = confusion_matrix(label(valid[TARGET]), model.predict(valid[FEATURES]), labels=LABELS)
    metadata = {'model_name': 'logistica_balanceada_riesgo70', 'features': FEATURES, 'labels': LABELS, 'cuts': CUTS, 'training_rows': len(train), 'validation_rows': len(valid), 'fit_scope': 'Solo entrenamiento; validación no usada para ajustar', 'created_utc': datetime.now(timezone.utc).isoformat(), 'sklearn_version': sklearn.__version__, 'sample_sha256': {name: sha(path) for name, path in paths.items()}, 'predict_rule': 'argmax; scores no calibrados', 'scope': 'Prototipo sintético, sin validación temporal ni impacto educativo demostrado'}
    REPORT.mkdir(parents=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    summary = save_metrics(cm, 'validation')
    joblib.dump({**metadata, 'model': model}, MODEL)
    (REPORT / 'entrenamiento_completado.json').write_text(json.dumps({**metadata, 'model_sha256': sha(MODEL)}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary)
    print('Modelo guardado:', MODEL)
    print('Test NO abierto. El dashboard ya puede ejecutarse.')


if __name__ == '__main__':
    main()
