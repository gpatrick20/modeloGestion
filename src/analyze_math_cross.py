from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "eda_cache"
OUT = ROOT / "reports" / "diagnostico_matematica" / "cruce_asistencia_estudio"

def main():
    train = ROOT / "data" / "train.csv"
    db = CACHE / "train_eda.duckdb"
    meta = CACHE / "source_metadata.json"
    if not db.exists() or not meta.exists():
        raise FileNotFoundError("Falta la cache del EDA original.")
    fingerprint = {"path": str(train.resolve()), "bytes": train.stat().st_size,
                   "mtime_ns": train.stat().st_mtime_ns}
    if json.loads(meta.read_text()) != fingerprint:
        raise ValueError("Cache desactualizada. Actualiza primero el EDA original.")
    OUT.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db), read_only=True) as con:
        con.execute("SET memory_limit='1GB'")
        con.execute("SET threads=2")
        base = """WITH typed AS (
            SELECT try_cast(TestScore_Math AS DOUBLE) AS math,
                   try_cast(AttendanceRate AS DOUBLE) AS asistencia,
                   try_cast(StudyHours AS DOUBLE) AS estudio FROM raw),
            checked AS (SELECT *,
                coalesce(isfinite(math) AND math BETWEEN 0 AND 100, false) AS ok_math,
                coalesce(isfinite(asistencia) AND asistencia BETWEEN .70 AND 1, false) AS ok_asistencia,
                coalesce(isfinite(estudio) AND estudio BETWEEN 0 AND 4, false) AS ok_estudio
                FROM typed),
            valid AS (SELECT *,
                least(5, floor(round((asistencia-.70)/.05,10)))::INTEGER AS a,
                least(7, floor(round(estudio/.5,10)))::INTEGER AS e
                FROM checked WHERE ok_math AND ok_asistencia AND ok_estudio)
        """
        quality = con.execute(base + """SELECT count(*) AS n_total,
            count(*) FILTER (WHERE NOT ok_math) AS math_no_valido,
            count(*) FILTER (WHERE NOT ok_asistencia) AS asistencia_no_valida,
            count(*) FILTER (WHERE NOT ok_estudio) AS estudio_no_valido,
            count(*) FILTER (WHERE ok_math AND ok_asistencia AND ok_estudio) AS n_incluido
            FROM checked""").df()
        stats = con.execute(base + """SELECT a,e,count(*) AS n, avg(math) AS math_medio,
            stddev_samp(math) AS desviacion, quantile_cont(math,.25) AS p25,
            median(math) AS mediana,quantile_cont(math,.75) AS p75
            FROM valid GROUP BY a,e ORDER BY a,e""").df()
    alabels = [f"{70+5*i}-{75+5*i}%" for i in range(6)]
    elabels = [f"{i*.5:g}-{(i+1)*.5:g}" for i in range(8)]
    grid = pd.MultiIndex.from_product([range(6),range(8)],names=['a','e']).to_frame(index=False)
    full = grid.merge(stats,on=['a','e'],how='left')
    full['n'] = full.n.fillna(0).astype('int64')
    full['asistencia_intervalo'] = full.a.map(dict(enumerate(alabels)))
    full['estudio_horas_diarias_intervalo'] = full.e.map(dict(enumerate(elabels)))
    full['grupo_menor_30'] = full.n < 30
    full['porcentaje_total_incluido'] = full.n / max(int(quality.n_incluido.iloc[0]),1)*100
    for key,name in [('a','asistencia'),('e','estudio')]:
        totals = full.groupby(key).n.transform('sum')
        full[f'porcentaje_dentro_{name}'] = np.where(totals>0,100*full.n/totals,np.nan)
    full.to_csv(OUT/'cruce_asistencia_estudio.csv',index=False)
    quality.to_csv(OUT/'calidad_cruce.csv',index=False)
    means = full.pivot(index='a',columns='e',values='math_medio').reindex(index=range(6),columns=range(8))
    counts = full.pivot(index='a',columns='e',values='n').reindex(index=range(6),columns=range(8))
    means.index=alabels;means.columns=elabels
    counts.index=alabels;counts.columns=elabels
    means.to_csv(OUT/'matriz_puntaje_medio.csv')
    counts.to_csv(OUT/'matriz_cantidad_registros.csv')
    shown = means.to_numpy().copy()
    shown[counts.to_numpy()<30] = np.nan
    cmap = plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
    fig,ax=plt.subplots(figsize=(12,6))
    image=ax.imshow(np.ma.masked_invalid(shown),cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for i in range(6):
        for j in range(8):
            n=int(counts.iloc[i,j]);m=means.iloc[i,j]
            label='sin datos' if n==0 else (f'n={n}*' if n<30 else f'{m:.1f}\nn={n:,}')
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(shown[i,j]) and shown[i,j]<55 else 'black')
    ax.set_xticks(range(8),elabels);ax.set_yticks(range(6),alabels)
    ax.set(xlabel='Horas de estudio por dia',ylabel='Asistencia',title='Matematica observada: media y cantidad por combinacion')
    fig.colorbar(image,ax=ax,label='Puntaje medio de matematica')
    fig.text(.01,.01,'Intervalos [inferior, superior), salvo el ultimo, cerrado. * n<30: media oculta solo en el grafico; criterio visual, no garantia estadistica.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'mapa_asistencia_estudio.png',dpi=160);plt.close(fig)
    notes = {"alcance":"Cruce descriptivo, no causal ni pronostico", "intervalos":"Cerrados a izquierda, abiertos a derecha; ultimo incluye 100% y 4 horas", "grupos_pequenos":"n<30 marcado y oculto en grafico, no borrado del CSV; criterio visual exploratorio", "faltantes":"Se excluyen y cuentan valores no validos; los conteos de invalidez pueden solaparse", "interpretacion":"Comparar estudio dentro de filas de asistencia y asistencia dentro de columnas de estudio. Intervalos no equivalen a controlar exactamente ni ajustar otros factores."}
    (OUT/'alcance.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(quality.to_string(index=False))
    print('\nPUNTAJE MEDIO:');print(means.round(2).to_string())
    print('\nCANTIDADES:');print(counts.to_string())
    print('\nReportes:', OUT)
    print('No se leyeron validacion ni test. No se modificaron modelos ni dashboard.')

if __name__ == '__main__':
    main()
