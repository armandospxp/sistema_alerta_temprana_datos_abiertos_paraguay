"""
03_score_alertas.py
-------------------
Aplica el modelo entrenado sobre la base procesada y genera
el archivo de alertas tempranas con nivel ALTO / MEDIO / BAJO.

Uso:
    python src/03_score_alertas.py
    python src/03_score_alertas.py --input data/processed/features.csv
    python src/03_score_alertas.py --input nueva_base.parquet --output alertas_junio.csv

Salida: data/processed/alertas_sat.csv
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd

# ── Rutas default ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH    = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "src", "modelo", "modelo_sat.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "src", "modelo", "metricas.json")
OUT_PATH     = os.path.join(BASE_DIR, "data", "processed", "alertas_sat.csv")

EXCLUIR_SCORE = [
    "id_solicitud", "periodo_desembolso", "nombre_departamento_particular",
    "atraso_30", "atraso_60", "atraso_90",
]


def clasificar_alerta(prob: float, umbral_alto: float, umbral_medio: float) -> str:
    if prob >= umbral_alto:
        return "ALTO"
    elif prob >= umbral_medio:
        return "MEDIO"
    else:
        return "BAJO"


def score(input_path: str, output_path: str):
    # 1. Cargar modelo y umbrales
    print(f"── Cargando modelo: {MODEL_PATH}")
    with open(MODEL_PATH, "rb")   as f: pipeline = pickle.load(f)
    with open(METRICS_PATH, "r")  as f: metricas = json.load(f)

    umbral_alto  = metricas["umbral_alerta_alta"]
    umbral_medio = metricas["umbral_alerta_media"]
    print(f"   Umbral ALTO:  > {umbral_alto}")
    print(f"   Umbral MEDIO: > {umbral_medio}")

    # 2. Cargar features
    print(f"── Cargando datos: {input_path}")
    df = pd.read_csv(input_path)

    # Guardar metadatos para el output
    meta_cols = [c for c in ["id_solicitud", "periodo_desembolso",
                              "nombre_departamento_particular",
                              "atraso_30", "atraso_60", "atraso_90"]
                 if c in df.columns]
    df_meta = df[meta_cols].copy()

    # 3. Preparar features
    X = df.drop(columns=EXCLUIR_SCORE, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # 4. Scoring
    print(f"── Generando scores para {len(X):,} registros...")
    prob_mora = pipeline.predict_proba(X)[:, 1]

    # 5. Construir output de alertas
    df_alertas = df_meta.copy()
    df_alertas["prob_mora_30"]  = prob_mora.round(4)
    df_alertas["score_riesgo"]  = (prob_mora * 100).round(1)
    df_alertas["nivel_alerta"]  = [
        clasificar_alerta(p, umbral_alto, umbral_medio) for p in prob_mora
    ]

    # Agregar las features más explicativas para facilitar acción
    features_clave = [
        "ratio_cuota_ingreso", "severidad_atraso", "atraso_promedio_norm",
        "mora_sistema_consumo", "tpm", "ipc_interanual",
        "antiguedad_negofin", "ingreso_real",
    ]
    for f in features_clave:
        if f in df.columns:
            df_alertas[f] = df[f].values

    # Ordenar por riesgo descendente
    df_alertas = df_alertas.sort_values("prob_mora_30", ascending=False).reset_index(drop=True)

    # 6. Resumen
    conteo = df_alertas["nivel_alerta"].value_counts()
    print(f"\n── Distribución de alertas:")
    for nivel in ["ALTO", "MEDIO", "BAJO"]:
        n   = conteo.get(nivel, 0)
        pct = n / len(df_alertas) * 100
        print(f"   {nivel:5s}: {n:5,} clientes ({pct:5.1f}%)")

    if "atraso_30" in df_alertas.columns:
        print(f"\n── Validación cruzada alerta vs realidad:")
        cross = pd.crosstab(df_alertas["nivel_alerta"], df_alertas["atraso_30"],
                            rownames=["Nivel alerta"], colnames=["Mora real"])
        print(cross.to_string())

    # 7. Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_alertas.to_csv(output_path, index=False)
    print(f"\n── Alertas guardadas en: {output_path}")

    # Imprimir top 10 más riesgosos
    print(f"\n── Top 10 clientes con mayor riesgo:")
    cols_show = ["id_solicitud", "score_riesgo", "nivel_alerta",
                 "ratio_cuota_ingreso", "severidad_atraso"]
    cols_show = [c for c in cols_show if c in df_alertas.columns]
    print(df_alertas[cols_show].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="SAT Crédito — Generador de Alertas")
    parser.add_argument("--input",  default=FEAT_PATH,  help="Parquet de features")
    parser.add_argument("--output", default=OUT_PATH,   help="CSV de alertas")
    args = parser.parse_args()
    score(args.input, args.output)


if __name__ == "__main__":
    main()
