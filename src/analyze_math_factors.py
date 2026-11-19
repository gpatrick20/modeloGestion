from pathlib import Path
import json
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "eda_cache"
REPORT = ROOT / "reports" / "diagnostico_matematica"
TRAIN = ROOT / "data" / "train.csv"

def qi(s):
    return '"' + s.replace('"', '""') + '"'

def main():
    db = CACHE / "train_eda.duckdb"
    meta = CACHE / "source_metadata.json"
    if not db.exists() or not meta.exists():
        raise FileNotFoundError("Ejecuta primero el EDA original para crear la cache de train.")
    fingerprint = {"path": str(TRAIN.resolve()), "bytes": TRAIN.stat().st_size,
                   "mtime_ns": TRAIN.stat().st_mtime_ns}
    if json.loads(meta.read_text()) != fingerprint:
        raise ValueError("La cache no corresponde al train actual. Actualiza primero el EDA original.")
    REPORT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=True)
    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")
    def query(sql):
        return con.execute(sql).df()
    def save(df, name):
        df.to_csv(REPORT / name, index=False)
    y = 'try_cast("TestScore_Math" AS DOUBLE)'
    base = f"WITH base AS (SELECT *, {y} AS math FROM raw), valid AS (SELECT * FROM base WHERE isfinite(math) AND math BETWEEN 0 AND 100) "
    quality = query(f"SELECT count(*) AS n_total, count(*) FILTER (WHERE {y} IS NULL OR NOT isfinite({y}) OR {y} NOT BETWEEN 0 AND 100) AS math_no_valido FROM raw")
    save(quality, "calidad_objetivo.csv")
    summary = query(base + "SELECT count(*) AS n_math_valido, avg(math) AS media, stddev_samp(math) AS desviacion, min(math) AS minimo, quantile_cont(math, .1) AS p10, quantile_cont(math, .25) AS p25, median(math) AS mediana, quantile_cont(math, .75) AS p75, quantile_cont(math, .9) AS p90, max(math) AS maximo FROM valid")
    save(summary, "resumen_matematica.csv")
    hist = query(base + "SELECT least(9, floor(math/10))::INTEGER AS intervalo_10_puntos, count(*) AS n FROM valid GROUP BY 1 ORDER BY 1")
    hist['porcentaje'] = hist.n / hist.n.sum() * 100
    save(hist, "distribucion_matematica.csv")
    categories = ["Gender", "Race", "ParentalEducation", "SchoolType", "Locale"]
    numeric = {"Age":(14,18), "Grade":(9,12), "SES_Quartile":(1,4), "InternetAccess":(0,1), "Extracurricular":(0,1), "PartTimeJob":(0,1), "ParentSupport":(0,1), "Romantic":(0,1), "FreeTime":(1,5), "GoOut":(1,5)}
    tables = []
    stats = "count(*) AS n, avg(math) AS math_medio, stddev_samp(math) AS desviacion, quantile_cont(math,.25) AS p25, median(math) AS mediana, quantile_cont(math,.75) AS p75"
    for col in categories + list(numeric):
        print("Analizando", col, flush=True)
        c = qi(col)
        if col in categories:
            expr = f"coalesce(nullif(trim({c}), ''), '[FALTANTE]')"
        else:
            lo,hi = numeric[col]
            x = f"try_cast({c} AS DOUBLE)"
            expr = f"CASE WHEN isfinite({x}) AND {x} BETWEEN {lo} AND {hi} AND {x}=floor({x}) THEN cast({x} AS VARCHAR) ELSE '[FALTANTE_O_INVALIDO]' END"
        df = query(base + f"SELECT {expr} AS grupo, {stats} FROM valid GROUP BY 1 ORDER BY 1")
        df.insert(0,'variable',col)
        df['porcentaje_math_valido'] = df.n / int(summary.n_math_valido.iloc[0]) * 100
        df['diferencia_media_global'] = df.math_medio - float(summary.media.iloc[0])
        tables.append(df)
    save(pd.concat(tables,ignore_index=True), "matematica_por_factor.csv")
    for col,lo,hi,step in [("AttendanceRate",.70,1,.05),("StudyHours",0,4,.5)]:
        x = f"try_cast({qi(col)} AS DOUBLE)"
        k = round((hi-lo)/step)
        expr = f"least({k-1}, floor(round(({x}-{lo})/{step},10)))::INTEGER"
        df = query(base + f"SELECT {expr} AS intervalo, {stats} FROM valid WHERE isfinite({x}) AND {x} BETWEEN {lo} AND {hi} GROUP BY 1 ORDER BY 1")
        df['limite_inferior'] = lo + df.intervalo*step
        df['limite_superior'] = df.limite_inferior + step
        df['incluye_limite_superior'] = df.intervalo == k-1
        save(df,f"matematica_por_{col}.csv")
        excluded = query(base + f"SELECT count(*) AS n_factor_faltante_o_invalido FROM valid WHERE {x} IS NULL OR NOT isfinite({x}) OR {x} NOT BETWEEN {lo} AND {hi}")
        save(excluded,f"calidad_{col}.csv")
    notes = {"alcance":"Diagnostico descriptivo de matematica, no pronostico ni causalidad", "fuente":"Cache completa de train.csv del EDA original", "umbral_bajo":"No definido; no se crean etiquetas de riesgo", "StudyHours":"Horas diarias segun diccionario", "distribucion":"Intervalos [0,10), ..., [90,100]", "faltantes":"Objetivo invalido excluido y contado; factores categoricos faltantes conservados; invalidos numericos agrupados; asistencia y estudio invalidos excluidos y contados", "precauciones":["Comparaciones sin ajuste por otros factores", "Diferencias grupales no son efectos de intervenciones", "No usar atributos sensibles para asignar apoyo automaticamente", "No se lee validacion ni test y no se modifica la base cache"]}
    (REPORT / "alcance.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf-8")
    con.close()
    print("\nRESUMEN MATEMATICA:")
    print(summary.to_string(index=False))
    print("\nReportes:", REPORT)
    print("No se entrenaron modelos ni se modificaron dashboard, GPA o datos originales.")

if __name__ == "__main__":
    main()
