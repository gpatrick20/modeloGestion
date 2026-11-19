from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .risk_math_common import ROOT, DATA, FEATURES, TARGET, LABELS, validate, label

OUT = ROOT / 'reports' / 'semaforo_final'
MODEL = ROOT / 'modelos' / 'semaforo_logistica_balanceada.joblib'
TEST = ROOT / 'data' / 'test.csv'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def evaluate(cm):
    support=cm.sum(axis=1); predicted=cm.sum(axis=0);tp=np.diag(cm)
    precision=np.divide(tp,predicted,out=np.zeros(3,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(3,dtype=float),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3,dtype=float),where=(precision+recall)>0)
    table=pd.DataFrame({'clase':LABELS,'precision':precision,'recall':recall,'f1':f1,'n_real':support,'n_estimado':predicted})
    n=int(cm.sum());risk=int(support[0]);alerts=int(predicted[:2].sum())
    summary={'n_test':n,'accuracy':float(tp.sum()/n),'f1_macro':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'recall_en_riesgo':float(recall[0]),'precision_en_riesgo':float(precision[0]),'riesgo_alto_a_verde':int(cm[0,2]),'riesgo_alto_con_alguna_alerta':int(cm[0,:2].sum()),'fraccion_riesgo_alto_con_alerta':float(cm[0,:2].sum()/risk) if risk else None,'alertas_rojas':int(predicted[0]),'alertas_amarillas':int(predicted[1]),'fraccion_total_con_alerta':float(alerts/n),'medio_riesgo_a_verde':int(cm[1,2])}
    return table,summary

def main():
    if OUT.exists() or MODEL.exists():
        raise FileExistsError('Ya existe una salida semaforo_final o el modelo final. No se sobrescribe ni se repite test automaticamente. Conserva resultados; si hubo un fallo, revisa el error antes de reintentar.')
    if not TEST.exists():
        raise FileNotFoundError(TEST)
    paths={k:DATA/f'{k}_math.csv' for k in ['train','validation']}
    train=validate(pd.read_csv(paths['train']),True)
    valid=validate(pd.read_csv(paths['validation']),True)
    for name,df in [('train',train),('validation',valid)]:
        if set(label(df[TARGET])) != set(LABELS):
            raise ValueError(f'Faltan clases en {name}')
    model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))
    model.fit(train[FEATURES],label(train[TARGET]))
    cmv=confusion_matrix(label(valid[TARGET]),model.predict(valid[FEATURES]),labels=LABELS)
    previous=ROOT/'reports'/'clasificacion_matematica'/'matriz_logistica_balanceada.csv'
    if not previous.exists():
        raise FileNotFoundError(f'Falta la matriz anterior para verificar reproduccion: {previous}')
    expected=pd.read_csv(previous,index_col=0).loc[LABELS,LABELS].to_numpy()
    if not np.array_equal(cmv,expected):
        raise ValueError('La matriz de validacion no reproduce la anterior. Test no fue abierto. Revisa muestras, versiones y configuracion antes de continuar.')
    metadata={'model_name':'logistica_balanceada','features':FEATURES,'labels':LABELS,'cuts':[60,70],'training_rows':len(train),'validation_rows':len(valid),'fit_scope':'Solo muestra original de train; sin reajustar con validation','decision':'Seleccion operativa tras revisar validacion: reducir riesgo alto enviado a verde; acepta mayor carga de revision','predict_rule':'argmax de predict_proba sin ajustar umbrales','scope':'Prototipo sintetico, no alerta temporal validada ni diagnostico integral','scores':'No calibrados; no interpretar como probabilidad fiable de necesidad de apoyo','created_utc':datetime.now(timezone.utc).isoformat(),'sklearn_version':sklearn.__version__,'sample_sha256':{k:sha(p) for k,p in paths.items()},'test_source':str(TEST.resolve()),'test_bytes':TEST.stat().st_size,'test_mtime_ns':TEST.stat().st_mtime_ns,'test_plan':'Archivo completo por bloques de 100000. Sin seleccion de modelos ni ajuste posterior basado en test.'}
    for k in paths:
        p=paths[k].with_suffix('.json')
        metadata[f'sampling_{k}']=json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    OUT.mkdir(parents=True);MODEL.parent.mkdir(parents=True,exist_ok=True)
    (OUT/'protocolo_previo_test.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    joblib.dump({**metadata,'model':model},MODEL)
    pd.DataFrame(cmv,index=LABELS,columns=LABELS).rename_axis('real').to_csv(OUT/'matriz_validacion_reproducida.csv')
    cm=np.zeros((3,3),dtype=np.int64);total=0
    print('Configuracion fijada y modelo guardado. Evaluando test completo...',flush=True)
    with pd.read_csv(TEST,usecols=FEATURES+[TARGET],chunksize=100000) as reader:
        for block,chunk in enumerate(reader,1):
            chunk=validate(chunk,True)
            cm+=confusion_matrix(label(chunk[TARGET]),model.predict(chunk[FEATURES]),labels=LABELS)
            total+=len(chunk);print(f'Bloque {block}: {total:,} registros',flush=True)
    if total==0:
        raise ValueError('Test vacio; evaluacion incompleta.')
    if set(np.where(cm.sum(axis=1)>0)[0]) != {0,1,2}:
        raise ValueError('Faltan clases en test. Revisar antes de reportar metricas.')
    table,summary=evaluate(cm)
    table.to_csv(OUT/'metricas_test_por_clase.csv',index=False)
    pd.DataFrame([summary]).to_csv(OUT/'resumen_test.csv',index=False)
    pd.DataFrame(cm,index=LABELS,columns=LABELS).rename_axis('real').to_csv(OUT/'matriz_test.csv')
    fig,ax=plt.subplots(figsize=(7,5));ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False,values_format='d')
    ax.set_title('Test completo: logistica balanceada');fig.tight_layout();fig.savefig(OUT/'matriz_test.png',dpi=150);plt.close(fig)
    demo=pd.DataFrame({'registro_id':[f'DEMO-{i:03d}' for i in range(1,6)],'AttendanceRate':[.76,.83,.88,.93,.97],'StudyHours':[.4,.7,1,1.4,1.8]})
    demo['categoria_estimada']=model.predict(demo[FEATURES]);probs=model.predict_proba(demo[FEATURES])
    for i,c in enumerate(model.classes_):demo[f'score_{c}']=probs[:,i]
    demo['tipo_registro']='perfil_ficticio_manual_no_evaluacion'
    demo.to_csv(OUT/'cinco_predicciones_demo.csv',index=False)
    completion={'finished_utc':datetime.now(timezone.utc).isoformat(),'test_rows':total,'model_sha256':sha(MODEL),'note':'Test evaluado sin reajuste. No usarlo para seguir seleccionando modelos. Demo no prueba desempeno ni garantiza combinaciones ausentes del entrenamiento.'}
    (OUT/'evaluacion_completada.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
    print('\nMETRICAS TEST:');print(table.to_string(index=False))
    print('\nRESUMEN:');print(pd.DataFrame([summary]).to_string(index=False))
    print('\nCINCO PERFILES FICTICIOS:');print(demo.to_string(index=False))
    print('\nReportes:',OUT)
    print('El modelo anterior y el dashboard no fueron modificados.')

if __name__=='__main__':
    main()
