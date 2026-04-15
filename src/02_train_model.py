"""
02_train_model.py
-----------------
Entrena un modelo de Survival Analysis (Regresión de Cox) para predecir
el tiempo hasta el primer incumplimiento crediticio.

Marco conceptual — Survival Analysis:
  - Pregunta: "¿Cuándo entrará en mora?" (no solo "¿entrará o no?")
  - tiempo_evento (T): períodos de 30 días hasta el primer incumplimiento
      T=1  → incumplió en los primeros 30 días
      T=2  → primer incumplimiento entre 30 y 60 días
      T=3  → primer incumplimiento entre 60 y 90 días
      T=3* → censurado (no incumplió en la ventana de 90 días observada)
  - evento (E): 1 si incumplió, 0 si fue censurado

Modelo: CoxPHFitter (lifelines)
  - Estima la función de riesgo h(t|X) = h₀(t) · exp(Xβ)
  - h₀(t): función de riesgo base (compartida por todos los clientes)
  - exp(Xβ): factor multiplicativo por covariables (hazard ratio)
  - Salida: S(t|X) = probabilidad de no incumplir hasta tiempo t

Métricas:
  - C-index de Harrell: concordancia entre scores y tiempos (análogo AUC)
  - Brier Score a t=1,2,3: calibración de probabilidades de supervivencia

Pipeline:
  1. Carga features.csv (output de 01_build_features.py)
  2. Preprocesamiento: imputación por mediana + estandarización
  3. Split train/test estratificado por evento
  4. Entrenamiento CoxPHFitter con regularización L2
  5. Evaluación: C-index, Brier Score, distribución de S(t)
  6. Guarda artefactos en src/modelo/modelo_sat.pkl + metricas.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH    = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "src", "modelo")
MODEL_PATH   = os.path.join(MODEL_DIR, "modelo_sat.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metricas.json")
FIMP_PATH    = os.path.join(BASE_DIR, "data", "processed", "feature_importance.csv")

# Columnas de supervivencia
DURATION_COL = "tiempo_evento"
EVENT_COL    = "evento"

# Columnas que NO entran como covariables al modelo
EXCLUIR = [
    "id_solicitud", "periodo_desembolso", "nombre_departamento_particular",
    "atraso_30", "atraso_60", "atraso_90",
    DURATION_COL, EVENT_COL,
]

# Hiperparámetros del Cox PH
COX_PARAMS = {
    "penalizer": 0.10,   # regularización L2 (estabilidad con features correlacionadas)
    "l1_ratio":  0.0,    # 0 = Ridge puro, 1 = Lasso puro
}


def cargar_datos(path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga features y variables de supervivencia."""
    df = pd.read_csv(path)

    y_time  = df[DURATION_COL]
    y_event = df[EVENT_COL]
    X = df.drop(columns=EXCLUIR, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    print(f"   Covariables numéricas: {X.shape[1]}")
    print(f"   Registros: {len(X):,}")
    print(f"   Eventos (incumplió):   {y_event.sum():,} ({y_event.mean()*100:.1f}%)")
    print(f"   Censurados:            {(y_event==0).sum():,} ({(y_event==0).mean()*100:.1f}%)")
    print(f"   Distribución T:  {dict(y_time.value_counts().sort_index())}")
    return X, y_time, y_event


def build_preprocessor() -> Pipeline:
    """Pipeline de preprocesamiento: imputación → estandarización."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def calcular_brier_score(
    cph: CoxPHFitter,
    X_scaled: pd.DataFrame,
    y_time: pd.Series,
    y_event: pd.Series,
    times: list[int],
) -> dict:
    """
    Calcula el Brier Score en cada tiempo t:
      BS(t) = E[ (evento_t - S(t|X))² ]
    donde evento_t = 1 si el evento ocurrió antes de t, 0 si no.
    Usa IPCW (Inverse Probability of Censoring Weights) simplificado.
    """
    surv_funcs = cph.predict_survival_function(X_scaled, times=times)
    brier_scores = {}

    for t in times:
        s_t = surv_funcs.loc[t].values          # S(t|X) para cada cliente
        # Indicador: ¿el evento ocurrió antes o en tiempo t?
        indicator = ((y_event == 1) & (y_time <= t)).astype(float).values
        bs = float(np.mean((indicator - (1 - s_t)) ** 2))
        brier_scores[f"brier_t{t}"] = round(bs, 4)

    return brier_scores


def evaluar(
    cph: CoxPHFitter,
    preprocessor: Pipeline,
    X_test: pd.DataFrame,
    y_time_test: pd.Series,
    y_event_test: pd.Series,
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    y_event_train: pd.Series,
) -> dict:
    """Métricas completas del modelo de supervivencia."""
    X_test_sc  = pd.DataFrame(preprocessor.transform(X_test),
                               columns=X_test.columns)
    X_train_sc = pd.DataFrame(preprocessor.transform(X_train),
                               columns=X_train.columns)

    # C-index: concordancia entre hazard score y tiempos observados
    risk_test  = cph.predict_partial_hazard(X_test_sc).values
    risk_train = cph.predict_partial_hazard(X_train_sc).values

    c_test  = concordance_index(y_time_test,  -risk_test,  y_event_test)
    c_train = concordance_index(y_time_train, -risk_train, y_event_train)

    # Distribución de S(t) en test: mediana de supervivencia por tiempo
    surv_funcs = cph.predict_survival_function(X_test_sc, times=[1, 2, 3])
    s1_median = float(np.median(surv_funcs.loc[1].values))
    s2_median = float(np.median(surv_funcs.loc[2].values))
    s3_median = float(np.median(surv_funcs.loc[3].values))

    # Brier Score (calibración)
    brier = calcular_brier_score(cph, X_test_sc, y_time_test, y_event_test, [1, 2, 3])

    # Umbrales de alerta sobre S(1) = prob de sobrevivir los primeros 30 días
    prob_mora_30 = 1 - surv_funcs.loc[1].values  # P(incumplir en T=1)
    umbral_alto  = float(np.percentile(prob_mora_30, 75))
    umbral_medio = float(np.percentile(prob_mora_30, 40))

    metricas = {
        "modelo": "CoxPHFitter (lifelines)",
        "parametros": COX_PARAMS,
        "duration_col": DURATION_COL,
        "event_col": EVENT_COL,
        "c_index_test":  round(c_test,  4),
        "c_index_train": round(c_train, 4),
        "overfit_gap":   round(c_train - c_test, 4),
        "s1_mediana_test": round(s1_median, 4),
        "s2_mediana_test": round(s2_median, 4),
        "s3_mediana_test": round(s3_median, 4),
        **brier,
        "umbral_alerta_alta":  round(umbral_alto,  4),
        "umbral_alerta_media": round(umbral_medio, 4),
        "n_test":  len(y_time_test),
        "n_train": len(y_time_train),
        "n_eventos_test": int(y_event_test.sum()),
    }
    return metricas


def feature_importance(cph: CoxPHFitter, feature_names: list) -> pd.DataFrame:
    """
    Importancia de features basada en el coeficiente de Cox estandarizado.
    Coeficiente positivo → mayor hazard (más riesgo).
    Se ordena por |coef| para reflejar magnitud del efecto.
    """
    summary = cph.summary.copy()
    fi = pd.DataFrame({
        "feature":    summary.index,
        "coef":       summary["coef"].values,
        "exp_coef":   summary["exp(coef)"].values,    # hazard ratio
        "se_coef":    summary["se(coef)"].values,
        "z":          summary["z"].values,
        "p":          summary["p"].values,
        "importancia_abs": np.abs(summary["coef"].values),
    })
    fi = fi.sort_values("importancia_abs", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    fi["importancia_pct"] = (
        fi["importancia_abs"] / fi["importancia_abs"].sum() * 100
    ).round(2)
    return fi


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Cargar datos
    print("── Cargando features...")
    X, y_time, y_event = cargar_datos(FEAT_PATH)

    # 2. Split estratificado por evento
    X_train, X_test, yt_train, yt_test, ye_train, ye_test = train_test_split(
        X, y_time, y_event,
        test_size=0.20,
        stratify=y_event,
        random_state=42,
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Preprocesar
    print("── Preprocesando...")
    preprocessor = build_preprocessor()
    X_train_sc = pd.DataFrame(preprocessor.fit_transform(X_train),
                               columns=X_train.columns)

    # 4. Armar DataFrame de supervivencia para lifelines
    df_train_surv = X_train_sc.copy()
    df_train_surv[DURATION_COL] = yt_train.values
    df_train_surv[EVENT_COL]    = ye_train.values

    # 5. Entrenar Cox PH
    print("── Entrenando CoxPHFitter...")
    cph = CoxPHFitter(**COX_PARAMS)
    cph.fit(df_train_surv, duration_col=DURATION_COL, event_col=EVENT_COL)
    print(f"   C-index (train, lifelines): {cph.concordance_index_:.4f}")

    # 6. Evaluar
    print("── Evaluando en test set...")
    metricas = evaluar(
        cph, preprocessor,
        X_test, yt_test, ye_test,
        X_train, yt_train, ye_train,
    )

    print(f"\n   C-index test:       {metricas['c_index_test']}")
    print(f"   C-index train:      {metricas['c_index_train']}")
    print(f"   Overfit gap:        {metricas['overfit_gap']} (< 0.05 = OK)")
    print(f"   S(1) mediana:       {metricas['s1_mediana_test']}  "
          f"→ P(mora 30d) mediana: {round(1-metricas['s1_mediana_test'],4)}")
    print(f"   S(2) mediana:       {metricas['s2_mediana_test']}")
    print(f"   S(3) mediana:       {metricas['s3_mediana_test']}")
    print(f"   Brier t=1:          {metricas['brier_t1']}")
    print(f"   Brier t=2:          {metricas['brier_t2']}")
    print(f"   Brier t=3:          {metricas['brier_t3']}")
    print(f"   Umbral ALTO:        > {metricas['umbral_alerta_alta']}")
    print(f"   Umbral MEDIO:       > {metricas['umbral_alerta_media']}")

    # 7. Importancia de features
    fi_df = feature_importance(cph, list(X_train.columns))
    print(f"\n── Top 10 features (|coeficiente Cox|):")
    for _, row in fi_df.head(10).iterrows():
        direction = "↑riesgo" if row["coef"] > 0 else "↓riesgo"
        print(f"   {int(row['rank']):2d}. {row['feature']:40s} "
              f"HR={row['exp_coef']:.3f}  {direction}  ({row['importancia_pct']:.1f}%)")

    # 8. Guardar artefactos
    artifacts = {
        "preprocessor":   preprocessor,
        "cox_model":      cph,
        "feature_names":  list(X_train.columns),
    }
    with open(MODEL_PATH,   "wb") as f: pickle.dump(artifacts, f)
    with open(METRICS_PATH, "w")  as f: json.dump(metricas, f, indent=2)
    fi_df.to_csv(FIMP_PATH, index=False)

    print(f"\n── Artefactos guardados:")
    print(f"   Modelo:      {MODEL_PATH}")
    print(f"   Métricas:    {METRICS_PATH}")
    print(f"   Importancia: {FIMP_PATH}")


if __name__ == "__main__":
    main()
