"""
02_train_model.py
-----------------
Entrena un modelo de clasificación binaria para predecir atraso_30
(alerta temprana: ¿el cliente entrará en mora en los próximos 30 días?).

Modelo: GradientBoostingClassifier (scikit-learn, sin dependencias externas)
  - Alternativa recomendada en producción: LightGBM o XGBoost

Pipeline:
  1. Carga features.csv
  2. Preprocesamiento (escalado numérico, codificación)
  3. Split train/test estratificado
  4. Búsqueda de hiperparámetros con cross-validation
  5. Evaluación: AUC-ROC, precisión, recall, F1, matriz de confusión
  6. Guarda modelo en src/modelo/modelo_sat.pkl
  7. Guarda métricas en src/modelo/metricas.json
  8. Guarda importancia de features en data/processed/feature_importance.csv
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH   = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "src", "modelo")
MODEL_PATH  = os.path.join(MODEL_DIR, "modelo_sat.pkl")
METRICS_PATH= os.path.join(MODEL_DIR, "metricas.json")
FIMP_PATH   = os.path.join(BASE_DIR, "data", "processed", "feature_importance.csv")

TARGET = "atraso_30"   # Alerta más temprana — máxima utilidad preventiva

# Features que NO entran al modelo (targets adicionales, metadatos, id)
EXCLUIR = [
    "id_solicitud", "periodo_desembolso", "nombre_departamento_particular",
    "atraso_60", "atraso_90",   # targets secundarios
    TARGET,                      # el target mismo
]

# Hiperparámetros del GBM (resultado de búsqueda previa en este dominio)
GBM_PARAMS = {
    "n_estimators":      300,
    "learning_rate":     0.05,
    "max_depth":         4,
    "min_samples_split": 50,
    "min_samples_leaf":  20,
    "subsample":         0.8,
    "max_features":      "sqrt",
    "random_state":      42,
}


def cargar_features(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y  = df[TARGET]
    X  = df.drop(columns=EXCLUIR, errors="ignore")
    # Eliminar columnas no numéricas residuales
    X  = X.select_dtypes(include=[np.number])
    print(f"   Features numéricas: {X.shape[1]}")
    print(f"   Registros: {len(X):,}")
    print(f"   Target '{TARGET}': {y.sum():,} positivos ({y.mean()*100:.1f}%)")
    return X, y


def build_pipeline() -> Pipeline:
    """Pipeline: imputación → escalado → GBM."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingClassifier(**GBM_PARAMS)),
    ])


def evaluar(pipeline, X_test, y_test, X_train, y_train) -> dict:
    """Calcula métricas completas sobre test set."""
    y_prob_test  = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test  = pipeline.predict(X_test)
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]

    auc_test  = roc_auc_score(y_test,  y_prob_test)
    auc_train = roc_auc_score(y_train, y_prob_train)
    ap_score  = average_precision_score(y_test, y_prob_test)
    cm        = confusion_matrix(y_test, y_pred_test)
    report    = classification_report(y_test, y_pred_test, output_dict=True)

    metricas = {
        "auc_roc_test":   round(auc_test,  4),
        "auc_roc_train":  round(auc_train, 4),
        "avg_precision":  round(ap_score,  4),
        "overfit_gap":    round(auc_train - auc_test, 4),
        "precision_class1": round(report["1"]["precision"], 4),
        "recall_class1":    round(report["1"]["recall"],    4),
        "f1_class1":        round(report["1"]["f1-score"],  4),
        "support_class1":   int(report["1"]["support"]),
        "confusion_matrix": cm.tolist(),
        "total_test":       len(y_test),
        "target": TARGET,
        "modelo": "GradientBoostingClassifier",
        "parametros": GBM_PARAMS,
    }
    return metricas


def calcular_umbral_optimo(pipeline, X_test, y_test) -> float:
    """
    Encuentra el umbral de probabilidad que maximiza F1.
    Útil para calibrar las alertas ALTO/MEDIO/BAJO.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    idx_opt   = np.argmax(f1_scores[:-1])
    return float(thresholds[idx_opt])


def feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    gbm = pipeline.named_steps["model"]
    fi = pd.DataFrame({
        "feature":   feature_names,
        "importance": gbm.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    return fi


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Cargar
    print("── Cargando features...")
    X, y = cargar_features(FEAT_PATH)

    # 2. Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Cross-validation 5-fold en train
    print("── Validación cruzada (5-fold)...")
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"   AUC CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Scores por fold: {[round(s,4) for s in cv_scores]}")

    # 4. Entrenar en todo el train set
    print("── Entrenando modelo final...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluar
    print("── Evaluando...")
    metricas = evaluar(pipeline, X_test, y_test, X_train, y_train)
    metricas["cv_auc_mean"] = round(cv_scores.mean(), 4)
    metricas["cv_auc_std"]  = round(cv_scores.std(),  4)

    umbral_optimo = calcular_umbral_optimo(pipeline, X_test, y_test)
    metricas["umbral_optimo_f1"] = round(umbral_optimo, 4)

    # Definir umbrales de alerta (basados en percentiles del prob score)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    metricas["umbral_alerta_alta"]  = round(float(np.percentile(y_prob_test, 75)), 4)
    metricas["umbral_alerta_media"] = round(float(np.percentile(y_prob_test, 40)), 4)

    print(f"\n   AUC-ROC test:     {metricas['auc_roc_test']}")
    print(f"   AUC-ROC train:    {metricas['auc_roc_train']}")
    print(f"   Overfit gap:      {metricas['overfit_gap']} (< 0.05 = OK)")
    print(f"   Avg Precision:    {metricas['avg_precision']}")
    print(f"   Recall (mora=1):  {metricas['recall_class1']}")
    print(f"   Precision (mora=1):{metricas['precision_class1']}")
    print(f"   Umbral óptimo F1: {metricas['umbral_optimo_f1']}")
    print(f"   Umbral ALTO:      > {metricas['umbral_alerta_alta']}")
    print(f"   Umbral MEDIO:     > {metricas['umbral_alerta_media']}")

    cm = metricas["confusion_matrix"]
    print(f"\n   Matriz de confusión (test):")
    print(f"   TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"   FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # 6. Feature importance
    fi_df = feature_importance(pipeline, list(X.columns))
    print(f"\n── Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        print(f"   {int(row['rank']):2d}. {row['feature']:45s} {row['importance_pct']:5.1f}%")

    # 7. Guardar todo
    with open(MODEL_PATH,   "wb")  as f: pickle.dump(pipeline, f)
    with open(METRICS_PATH, "w")   as f: json.dump(metricas, f, indent=2)
    fi_df.to_csv(FIMP_PATH, index=False)

    print(f"\n── Artefactos guardados:")
    print(f"   Modelo:      {MODEL_PATH}")
    print(f"   Métricas:    {METRICS_PATH}")
    print(f"   Importancia: {FIMP_PATH}")


if __name__ == "__main__":
    main()
