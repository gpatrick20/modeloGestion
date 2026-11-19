from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["AttendanceRate", "StudyHours", "TestScore_Math"]
DEST = ROOT / "data" / "samples_math"

def prepare(name, size, seed):
    path = ROOT / "data" / f"{name}.csv"
    rng = np.random.default_rng(seed)
    selected = None
    total = 0
    bounds = {"AttendanceRate": (.70, 1.0), "StudyHours": (0.0, 4.0), "TestScore_Math": (0.0, 100.0)}
    with pd.read_csv(path, usecols=COLS, chunksize=100_000) as reader:
        for i, chunk in enumerate(reader, 1):
            for col, (low, high) in bounds.items():
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                if not (np.isfinite(chunk[col]) & chunk[col].between(low, high)).all():
                    raise ValueError(f"Valores nulos o inválidos en {name}, bloque {i}, columna {col}. No se eliminan silenciosamente.")
            total += len(chunk)
            chunk["_key"] = rng.random(len(chunk))
            candidates = chunk if selected is None else pd.concat([selected, chunk], ignore_index=True)
            selected = candidates.nsmallest(min(size, len(candidates)), "_key").copy()
            print(f"{name}: {total:,} filas recorridas; {len(selected):,} retenidas", flush=True)
    if selected is None or selected.empty:
        raise ValueError(f"Archivo vacío: {path}")
    sample = selected.drop(columns="_key").reset_index(drop=True)
    out = DEST / f"{name}_math.csv"
    sample.to_csv(out, index=False)
    meta = {"source": str(path.resolve()), "source_bytes": path.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns, "source_rows": total,
            "sample_rows": len(sample), "seed": seed, "columns": COLS,
            "method": "uniform random priorities across all chunks"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Guardado: {out}", flush=True)

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    prepare("train", 100_000, 42)
    prepare("validation", 30_000, 43)
    print("Listo. No se abrió test.csv ni se modificaron CSV originales.")
