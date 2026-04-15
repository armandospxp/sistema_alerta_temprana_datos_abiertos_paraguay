"""
03_score_alertas.py
-------------------
Aplica el modelo de Survival Analysis (Cox PH) sobre la base procesada
y genera alertas tempranas con probabilidades de supervivencia por horizonte.

Marco conceptual:
  - S(t|X): probabilidad de que un cliente NO incumpla hasta el período t
  - P(mora ≤ t|X) = 1 - S(t|X): probabilidad acumulada de incumplimiento
  - El nivel de alerta se basa en P(mora ≤ 30 días) = 1 - S(1)
  - Se reportan también S(2) y S(3) para análisis de severidad y cascada

Columnas de salida:
  - prob_mora_30d / 60d / 90d : probabilidad de incumplir dentro de ese horizonte
  - surv_30d / 60d / 90d      : probabilidad de sobrevivir (no incumplir)
  - tiempo_esperado_mora      : tiempo esperado al incumplimiento (en períodos)
  - nivel_alerta              : ALTO / MEDIO / BAJO (basado en prob_mora_30d)

Uso:
    python src/03_score_alertas.py
    python src/03_score_alertas.py --input data/processed/features.csv
    python src/03_score_alertas.py --input nueva_base.csv --output alertas_julio.csv
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

# Columnas que no son covariables del modelo
EXCLUIR_SCORE = [
    "id_solicitud", "periodo_desembolso", "nombre_departamento_particular",
    "atraso_30", "atraso_60", "atraso_90",
    "tiempo_evento", "evento",
]

# Columnas de metadatos a preservar en el output
META_COLS = [
    "id_solicitud", "periodo_desembolso", "nombre_departamento_particular",
    "atraso_30", "atraso_60", "atraso_90", "tiempo_evento", "evento",
]

# Features clave para adjuntar al output (facilitan la acción preventiva)
FEATURES_CLAVE = [
    "ratio_cuota_ingreso", "severidad_atraso", "atraso_promedio_norm",
    "promedio_atraso_negofin", "maximo_atraso_negofin",
    "mora_sistema_consumo", "tpm", "ipc_interanual",
    "antiguedad_negofin", "ingreso_real", "cant_cuotas",
]


def clasificar_alerta(prob_mora_30d: float, umbral_alto: float, umbral_medio: float) -> str:
    """Clasifica el nivel de alerta en base a P(mora ≤ 30 días)."""
    if prob_mora_30d >= umbral_alto:
        return "ALTO"
    elif prob_mora_30d >= umbral_medio:
        return "MEDIO"
    else:
        return "BAJO"


def calcular_tiempo_esperado(surv_30: float, surv_60: float, surv_90: float) -> float:
    """
    Tiempo esperado al incumplimiento (en períodos de 30 días).
    Aproximación discreta basada en las probabilidades de supervivencia:
      E[T] = sum_{t=1}^{3} P(T >= t) = S(0) + S(1) + S(2)
    donde S(0) = 1 por definición.
    """
    return round(1.0 + surv_30 + surv_60, 3)


def score(input_path: str, output_path: str):
    # 1. Cargar modelo y umbrales
    print(f"── Cargando modelo: {MODEL_PATH}")
    with open(MODEL_PATH,   "rb") as f: artifacts = pickle.load(f)
    with open(METRICS_PATH, "r")  as f: metricas  = json.load(f)

    preprocessor  = artifacts["preprocessor"]
    cph           = artifacts["cox_model"]
    feature_names = artifacts["feature_names"]

    umbral_alto  = metricas["umbral_alerta_alta"]
    umbral_medio = metricas["umbral_alerta_media"]
    print(f"   C-index (test):  {metricas.get('c_index_test', 'N/A')}")
    print(f"   Umbral ALTO:     > {umbral_alto}")
    print(f"   Umbral MEDIO:    > {umbral_medio}")

    # 2. Cargar y preparar datos
    print(f"── Cargando datos: {input_path}")
    df = pd.read_csv(input_path)

    # Preservar metadatos
    meta_cols_presentes = [c for c in META_COLS if c in df.columns]
    df_meta = df[meta_cols_presentes].copy()

    # Preparar covariables (solo las que usó el modelo)
    X = df.drop(columns=EXCLUIR_SCORE, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    # Alinear columnas con las del modelo (rellenar con 0 si falta alguna)
    X = X.reindex(columns=feature_names, fill_value=0)

    # 3. Preprocesar
    X_scaled = pd.DataFrame(preprocessor.transform(X), columns=feature_names)

    # 4. Generar probabilidades de supervivencia en los 3 horizontes
    print(f"── Generando scores de supervivencia para {len(X_scaled):,} registros...")
    surv_funcs = cph.predict_survival_function(X_scaled, times=[1, 2, 3])
    # surv_funcs: DataFrame con índice=tiempo, columnas=registros

    surv_30 = surv_funcs.loc[1].values   # S(1): P(no incumplir en 30d)
    surv_60 = surv_funcs.loc[2].values   # S(2): P(no incumplir en 60d)
    surv_90 = surv_funcs.loc[3].values   # S(3): P(no incumplir en 90d)

    prob_mora_30 = 1 - surv_30
    prob_mora_60 = 1 - surv_60
    prob_mora_90 = 1 - surv_90

    # Score de riesgo relativo (partial hazard de Cox)
    score_riesgo = cph.predict_partial_hazard(X_scaled).values

    # 5. Construir output
    df_alertas = df_meta.copy()

    # Probabilidades de supervivencia
    df_alertas["surv_30d"]  = surv_30.round(4)
    df_alertas["surv_60d"]  = surv_60.round(4)
    df_alertas["surv_90d"]  = surv_90.round(4)

    # Probabilidades acumuladas de mora
    df_alertas["prob_mora_30d"] = prob_mora_30.round(4)
    df_alertas["prob_mora_60d"] = prob_mora_60.round(4)
    df_alertas["prob_mora_90d"] = prob_mora_90.round(4)

    # Score relativo y tiempo esperado
    df_alertas["score_riesgo_relativo"] = score_riesgo.round(4)
    df_alertas["tiempo_esperado_mora"]  = [
        calcular_tiempo_esperado(s1, s2, s3)
        for s1, s2, s3 in zip(surv_30, surv_60, surv_90)
    ]

    # Nivel de alerta basado en P(mora ≤ 30d)
    df_alertas["nivel_alerta"] = [
        clasificar_alerta(p, umbral_alto, umbral_medio) for p in prob_mora_30
    ]

    # Features clave explicativas
    for f in FEATURES_CLAVE:
        if f in df.columns:
            df_alertas[f] = df[f].values

    # Ordenar por riesgo descendente (prob_mora_30d)
    df_alertas = df_alertas.sort_values("prob_mora_30d", ascending=False).reset_index(drop=True)

    # 6. Resumen de alertas
    conteo = df_alertas["nivel_alerta"].value_counts()
    print(f"\n── Distribución de alertas (basado en P(mora ≤ 30d)):")
    for nivel in ["ALTO", "MEDIO", "BAJO"]:
        n   = conteo.get(nivel, 0)
        pct = n / len(df_alertas) * 100
        print(f"   {nivel:5s}: {n:6,} clientes ({pct:5.1f}%)")

    print(f"\n── Probabilidades de mora promedio por nivel:")
    for nivel in ["ALTO", "MEDIO", "BAJO"]:
        sub = df_alertas[df_alertas["nivel_alerta"] == nivel]
        if len(sub) == 0:
            continue
        print(f"   {nivel:5s}: "
              f"P(mora 30d)={sub['prob_mora_30d'].mean():.3f}  "
              f"P(mora 60d)={sub['prob_mora_60d'].mean():.3f}  "
              f"P(mora 90d)={sub['prob_mora_90d'].mean():.3f}")

    # Validación cruzada si hay targets reales disponibles
    if "atraso_30" in df_alertas.columns and "evento" in df_alertas.columns:
        print(f"\n── Validación: alerta vs evento real (survival):")
        cross = pd.crosstab(
            df_alertas["nivel_alerta"],
            df_alertas["evento"],
            rownames=["Nivel alerta"],
            colnames=["Evento (incumplió)"],
        )
        print(cross.to_string())

        print(f"\n── Concordancia: alerta vs atraso_30 observado:")
        cross30 = pd.crosstab(
            df_alertas["nivel_alerta"],
            df_alertas["atraso_30"],
            rownames=["Nivel alerta"],
            colnames=["atraso_30 real"],
        )
        print(cross30.to_string())

    # 7. Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_alertas.to_csv(output_path, index=False)
    print(f"\n── Alertas guardadas en: {output_path}")

    # Top 10 clientes más riesgosos
    print(f"\n── Top 10 clientes con mayor riesgo de mora a 30 días:")
    cols_show = [
        "id_solicitud", "nivel_alerta",
        "prob_mora_30d", "prob_mora_60d", "prob_mora_90d",
        "tiempo_esperado_mora", "severidad_atraso",
    ]
    cols_show = [c for c in cols_show if c in df_alertas.columns]
    print(df_alertas[cols_show].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="SAT Crédito — Scoring de Supervivencia")
    parser.add_argument("--input",  default=FEAT_PATH, help="CSV de features")
    parser.add_argument("--output", default=OUT_PATH,  help="CSV de alertas")
    args = parser.parse_args()
    score(args.input, args.output)


if __name__ == "__main__":
    main()
