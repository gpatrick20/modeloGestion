import json
import time
import hashlib
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score,balanced_accuracy_score
from .risk_math_common import *

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    REPORT.mkdir(parents=True,exist_ok=True);MODEL.parent.mkdir(parents=True,exist_ok=True)
    paths={s:DATA/f'{s}_math.csv' for s in ['train','validation']}
    train=validate(pd.read_csv(paths['train']),True)
    valid=validate(pd.read_csv(paths['validation']),True)
    yt=label(train[TARGET]);yv=label(valid[TARGET])
    counts=[]
    for split,y in [('train',yt),('validation',yv)]:
        for name in LABELS:
            n=int((y==name).sum())
            counts.append({'particion':split,'clase':name,'n':n,'porcentaje':100*n/len(y)})
            if n==0:
                raise ValueError(f'Clase ausente: {split}/{name}')
    dist=pd.DataFrame(counts);dist.to_csv(REPORT/'distribucion_clases.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    for offset,split in [(-.2,'train'),(.2,'validation')]:
        v=dist[dist.particion==split].set_index('clase').loc[LABELS]
        ax.bar(np.arange(3)+offset,v.porcentaje,width=.4,label=split)
    ax.set_xticks(range(3),LABELS);ax.set_ylabel('Porcentaje');ax.legend();fig.tight_layout()
    fig.savefig(REPORT/'distribucion_clases.png',dpi=150);plt.close(fig)
    models={
        'baseline_mayoritaria':DummyClassifier(strategy='most_frequent'),
        'logistica':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)),
        'logistica_balanceada':make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)),
        'random_forest_balanceado':RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=30,class_weight='balanced',n_jobs=2,random_state=42),
    }
    rows=[];best=None
    for name,model in models.items():
        print('Entrenando',name,flush=True)
        start=time.perf_counter();model.fit(train[FEATURES],yt);elapsed=time.perf_counter()-start
        pred=model.predict(valid[FEATURES])
        report=classification_report(yv,pred,labels=LABELS,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(REPORT/f'metricas_{name}.csv')
        cm=confusion_matrix(yv,pred,labels=LABELS)
        pd.DataFrame(cm,index=LABELS,columns=LABELS).rename_axis('real').to_csv(REPORT/f'matriz_{name}.csv')
        fig,ax=plt.subplots(figsize=(7,5));ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False,values_format='d')
        ax.set_title(name);fig.tight_layout();fig.savefig(REPORT/f'matriz_{name}.png',dpi=150);plt.close(fig)
        row={'modelo':name,'accuracy':accuracy_score(yv,pred),'f1_macro':f1_score(yv,pred,labels=LABELS,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(yv,pred),'recall_en_riesgo':report['en_riesgo']['recall'],'precision_en_riesgo':report['en_riesgo']['precision'],'segundos_fit':elapsed,'n_train':len(train),'n_validation':len(valid)}
        rows.append(row)
        # Selection fixed before evaluation: macro F1, then risk recall.
        score=(row['f1_macro'],row['recall_en_riesgo'])
        if best is None or score>best['score']:
            best={'model':model,'score':score,'model_name':name,'metrics':row}
    results=pd.DataFrame(rows).sort_values(['f1_macro','recall_en_riesgo'],ascending=False)
    results.to_csv(REPORT/'comparacion_clasificadores.csv',index=False)
    metadata={'features':FEATURES,'labels':LABELS,'target_origin':TARGET,'cuts':[60,70],'cut_interpretation':'Prototipo: <60, 60 a <70, >=70; no politica oficial del dataset','selection':'validation_macro_f1_then_recall_en_riesgo','created_utc':datetime.now(timezone.utc).isoformat(),'sklearn_version':sklearn.__version__,'data_sha256':{k:digest(p) for k,p in paths.items()},'scope':'Clasificacion experimental con datos sinteticos. Sin validacion temporal; no diagnostico integral. Probabilidades no calibradas.'}
    bundle={**metadata,'model':best['model'],'model_name':best['model_name'],'metrics':best['metrics']}
    joblib.dump(bundle,MODEL)
    (REPORT/'modelo_seleccionado.json').write_text(json.dumps({k:v for k,v in bundle.items() if k!='model'},indent=2,ensure_ascii=False),encoding='utf-8')
    pred=best['model'].predict(valid[FEATURES]);proba=best['model'].predict_proba(valid[FEATURES])
    details=valid[FEATURES+[TARGET]].copy();details.insert(0,'fila_validacion',np.arange(len(valid)))
    details['clase_observada']=yv;details['clase_estimada']=pred
    for i,c in enumerate(best['model'].classes_):
        details[f'score_{c}']=proba[:,i]
    details.to_csv(REPORT/'predicciones_validacion.csv',index=False)
    print('\nDISTRIBUCION:');print(dist.to_string(index=False))
    print('\nCOMPARACION:');print(results.to_string(index=False))
    print('\nSeleccionado provisional:',best['model_name'])
    print('Modelo:',MODEL)
    print('No se abrio test ni se modifico el dashboard. Revisar errores antes de integrar.')

if __name__=='__main__':
    main()
