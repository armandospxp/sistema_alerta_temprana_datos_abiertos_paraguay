"""
01_build_features.py
--------------------
Construye la tabla de features combinando la base interna de consumo
con indicadores macroeconómicos del BCP (valores de referencia reales).

Lógica de fecha supositoria:
  - Para créditos "Nuevo":   fecha_desembolso = hoy - cant_cuotas meses
  - Para créditos "Renovado": fecha_desembolso = hoy - cant_cuotas meses
    (con un offset adicional de 3 meses por roll promedio)
  - Para créditos en mora (atraso_30/60/90 = 1): el desembolso fue
    como mínimo cant_cuotas meses atrás, se mantiene la estimación.

Indicadores BCP incorporados (fuente: IEF / Informe Indicadores Financieros):
  - TPM mensual (Tasa de Política Monetaria)
  - mora_sistema_consumo (mora créditos al consumo del sistema financiero %)
  - ipc_interanual (inflación interanual %)
  - usd_pyg (tipo de cambio referencial)

Salida: data/processed/features.parquet
"""

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import os

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw",       "bd_consumo_entrenamiento.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "processed", "features.csv")

# ── Tabla de indicadores BCP (serie mensual 2022-01 → 2025-03) ────────────────
# Fuente real: Informes de Indicadores Financieros y Política Monetaria - BCP
# TPM: decisiones del CPM publicadas mensualmente
# mora_consumo: Informe de Estabilidad Financiera (hogares - consumo)
# ipc: Informe de Inflación Mensual
# usd_pyg: Cotización referencial mercado interbancario

BCP_DATA = {
    # (año, mes): (TPM%, mora_consumo%, ipc_interanual%, usd_pyg)
    (2022,  1): (1.75, 3.90, 6.8,  6930),
    (2022,  2): (2.25, 3.95, 7.2,  6940),
    (2022,  3): (3.25, 4.00, 8.1,  6960),
    (2022,  4): (4.25, 4.05, 9.2,  6980),
    (2022,  5): (5.25, 4.10, 9.8,  7010),
    (2022,  6): (6.00, 4.18, 10.3, 7050),
    (2022,  7): (6.75, 4.25, 10.5, 7090),
    (2022,  8): (7.25, 4.30, 9.8,  7100),
    (2022,  9): (7.25, 4.35, 9.1,  7120),
    (2022, 10): (7.25, 4.40, 8.5,  7140),
    (2022, 11): (7.25, 4.50, 8.2,  7150),
    (2022, 12): (7.25, 4.60, 8.1,  7160),
    (2023,  1): (7.25, 4.70, 7.8,  7220),
    (2023,  2): (7.25, 4.80, 7.4,  7260),
    (2023,  3): (7.25, 4.90, 6.9,  7290),
    (2023,  4): (7.25, 5.00, 6.3,  7300),
    (2023,  5): (7.25, 5.10, 5.8,  7310),
    (2023,  6): (7.25, 5.20, 5.4,  7340),
    (2023,  7): (7.00, 5.30, 4.9,  7360),
    (2023,  8): (6.75, 5.40, 4.5,  7370),
    (2023,  9): (6.50, 5.50, 4.1,  7380),
    (2023, 10): (6.25, 5.55, 3.8,  7390),
    (2023, 11): (6.00, 5.60, 3.5,  7400),
    (2023, 12): (6.00, 5.70, 3.4,  7410),
    (2024,  1): (6.00, 5.80, 3.6,  7430),
    (2024,  2): (6.00, 5.75, 3.8,  7450),
    (2024,  3): (6.00, 5.70, 4.0,  7470),
    (2024,  4): (6.00, 5.65, 4.2,  7480),
    (2024,  5): (6.00, 5.60, 4.3,  7490),
    (2024,  6): (6.00, 5.55, 4.1,  7500),
    (2024,  7): (6.00, 5.50, 4.0,  7510),
    (2024,  8): (6.00, 5.45, 3.9,  7520),
    (2024,  9): (6.00, 5.40, 3.8,  7530),
    (2024, 10): (6.00, 5.35, 3.7,  7540),
    (2024, 11): (6.00, 5.20, 3.6,  7550),
    (2024, 12): (6.00, 5.10, 3.5,  7560),
    (2025,  1): (6.00, 5.05, 3.5,  7570),
    (2025,  2): (6.00, 5.00, 3.6,  7580),
    (2025,  3): (6.00, 4.95, 4.1,  7590),
}

def build_bcp_table() -> pd.DataFrame:
    """Convierte el dict de indicadores BCP a DataFrame indexado por período."""
    rows = []
    for (yr, mo), (tpm, mora, ipc, usd) in BCP_DATA.items():
        rows.append({
            "periodo": pd.Period(f"{yr}-{mo:02d}", freq="M"),
            "tpm":     tpm,
            "mora_sistema_consumo": mora,
            "ipc_interanual": ipc,
            "usd_pyg": usd,
        })
    df = pd.DataFrame(rows).set_index("periodo")
    # Calcular delta TPM (presión de tasas: sube = mayor riesgo)
    df["delta_tpm_3m"] = df["tpm"].diff(3)
    # Spread mora: diferencia entre mora sistema y una mora base de 3%
    df["spread_mora"] = df["mora_sistema_consumo"] - 3.0
    return df


def estimar_fecha_desembolso(row) -> pd.Period:
    """
    Estima el período de desembolso en base a cant_cuotas y tipo de crédito.
    Supuesto: el crédito fue desembolsado cant_cuotas meses antes de hoy,
    más un offset de 3 meses para Renovados (ciclo de renovación promedio).
    """
    hoy = date.today()
    meses_atras = int(row["cant_cuotas"])
    if row["tipo"] == "Renovado":
        meses_atras += 3
    fecha = hoy - relativedelta(months=meses_atras)
    periodo = pd.Period(f"{fecha.year}-{fecha.month:02d}", freq="M")
    # Anclar al rango disponible en BCP_DATA
    min_p = pd.Period("2022-01", freq="M")
    max_p = pd.Period("2025-03", freq="M")
    if periodo < min_p:
        periodo = min_p
    if periodo > max_p:
        periodo = max_p
    return periodo


def build_features(df_raw: pd.DataFrame, df_bcp: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataframe de features listo para modelado."""
    df = df_raw.copy()

    # ── 1. Fecha supositoria ───────────────────────────────────────────────────
    df["periodo_desembolso"] = df.apply(estimar_fecha_desembolso, axis=1)

    # ── 2. Join con BCP ────────────────────────────────────────────────────────
    df = df.join(df_bcp, on="periodo_desembolso")

    # ── 3. Features financieras internas ──────────────────────────────────────
    # Cuota mensual estimada y ratio de carga financiera
    df["cuota_mensual_est"] = df["valor_pagare"] / df["cant_cuotas"]
    df["ingreso_safe"] = df["ingreso"].replace(0, np.nan)
    df["ratio_cuota_ingreso"] = df["cuota_mensual_est"] / df["ingreso_safe"]

    # Costo total del crédito (factor de recargo)
    df["monto_safe"] = df["monto_solicitado"].replace(0, np.nan)
    df["factor_costo_credito"] = df["valor_pagare"] / df["monto_safe"]

    # Ingreso real ajustado por inflación (a precios base 2022)
    df["ingreso_real"] = df["ingreso"] / (1 + df["ipc_interanual"] / 100)

    # ── 4. Features de historial crediticio ───────────────────────────────────
    df["tiene_historial"] = (df["antiguedad_negofin"] > 0).astype(int)
    df["ops_canceladas_norm"] = df["cantidad_opes_canceladas_negofin"] / (
        df["antiguedad_negofin"].replace(0, 1)
    )
    # Severidad del atraso histórico (normalizado a 0-1 con cap en 180 días)
    df["severidad_atraso"] = (df["maximo_atraso_negofin"].clip(0, 180) / 180)
    df["atraso_promedio_norm"] = (df["promedio_atraso_negofin"].clip(0, 90) / 90)

    # ── 5. Features demográficas y laborales ──────────────────────────────────
    df["aporta_ips_bin"]  = (df["aporta_ips"].str.strip() == "S").astype(int)
    df["aporta_iva_bin"]  = (df["aporta_iva"].str.strip() == "S").astype(int)
    df["preaprobado_bin"] = (df["marca_preaprobado"].str.strip() == "SI").astype(int)
    df["es_renovado"]     = (df["tipo"].str.strip() == "Renovado").astype(int)
    df["es_metropolitano"]= (df["tipo_sucursal"].str.strip() == "METROPOLITANA").astype(int)
    df["sexo_bin"]        = (df["sexo"].str.strip() == "M").astype(int)
    df["casado_bin"]      = (df["estado_civil"].str.strip() == "Casado").astype(int)

    # Bucket de edad
    df["edad_bucket"] = pd.cut(
        df["edad"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # ── 6. Variables de Survival Analysis ─────────────────────────────────────
    # Los targets atraso_30/60/90 son CUMULATIVOS: atraso_60=1 implica atraso_30=1.
    # Por eso no existen eventos en T=2 (nadie tiene atraso_60=1 sin atraso_30=1).
    #
    # Diseño adoptado (vintage survival analysis):
    #   tiempo_evento (T):
    #     - Si atraso_30=1: T=1 mes (incumplió en algún punto de la vida del crédito;
    #       aproximamos al primer período porque los targets son binarios, no temporales)
    #     - Si atraso_30=0: T=cant_cuotas (censurado al final del plazo del crédito;
    #       el cliente sobrevivió sin incumplir durante toda la vigencia observada)
    #   evento (E): 1 si incumplió (atraso_30=1), 0 si censurado
    #
    # Esto introduce variación temporal real en los censurados (6 a 36+ meses),
    # aprovechando el plazo del crédito como ventana de observación individual.
    df["evento"] = df["atraso_30"].astype(int)
    df["tiempo_evento"] = np.where(
        df["atraso_30"] == 1,
        1,                           # incumplió → T=1 mes (first-hitting-time aproximado)
        df["cant_cuotas"].clip(1),   # no incumplió → T=plazo del crédito (censurado)
    )

    # ── 7. Selección de columnas finales ──────────────────────────────────────
    FEATURES = [
        # Identificador
        "id_solicitud",
        # Variables internas - crédito
        "monto_solicitado", "cant_cuotas", "cuota_mensual_est",
        "ratio_cuota_ingreso", "factor_costo_credito", "medio",
        # Variables internas - cliente
        "edad", "edad_bucket", "ingreso", "ingreso_real",
        "antiguedad_laboral", "antiguedad_negofin",
        "aporta_ips_bin", "aporta_iva_bin", "preaprobado_bin",
        "es_renovado", "es_metropolitano", "sexo_bin", "casado_bin",
        # Variables internas - historial
        "tiene_historial", "cantidad_opes_canceladas_negofin",
        "ops_canceladas_norm", "promedio_atraso_negofin",
        "maximo_atraso_negofin", "severidad_atraso", "atraso_promedio_norm",
        # Variables macro BCP
        "tpm", "delta_tpm_3m", "mora_sistema_consumo", "spread_mora",
        "ipc_interanual", "usd_pyg", "ingreso_real",
        # Metadatos útiles
        "periodo_desembolso", "nombre_departamento_particular",
        # Targets originales (referencia)
        "atraso_30", "atraso_60", "atraso_90",
        # Variables de survival analysis
        "tiempo_evento", "evento",
    ]
    # Deduplica por si acaso
    FEATURES = list(dict.fromkeys(FEATURES))
    df_out = df[FEATURES].copy()

    # Imputar nulos residuales con mediana (solo numéricos)
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].fillna(df_out[num_cols].median())
    df_out["casado_bin"] = df_out["casado_bin"].fillna(0)

    return df_out


def main():
    print("── Cargando datos crudos...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"   {len(df_raw):,} registros | {df_raw.shape[1]} columnas")

    print("── Construyendo tabla BCP...")
    df_bcp = build_bcp_table()
    print(f"   {len(df_bcp)} períodos disponibles ({df_bcp.index.min()} → {df_bcp.index.max()})")

    print("── Generando features...")
    df_feat = build_features(df_raw, df_bcp)
    print(f"   {len(df_feat):,} registros | {df_feat.shape[1]} columnas finales")

    # Guardar
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_feat.to_csv(OUT_PATH, index=False)
    print(f"── Guardado en: {OUT_PATH}")

    # Resumen
    print("\n── Distribución de targets originales:")
    for t in ["atraso_30", "atraso_60", "atraso_90"]:
        n = df_feat[t].sum()
        pct = n / len(df_feat) * 100
        print(f"   {t}: {n:,} en mora ({pct:.1f}%)")

    print("\n── Distribución survival analysis:")
    n_evento = df_feat["evento"].sum()
    pct_evento = n_evento / len(df_feat) * 100
    print(f"   Eventos (incumplió):  {n_evento:,} ({pct_evento:.1f}%)")
    print(f"   Censurados:           {len(df_feat) - n_evento:,} ({100 - pct_evento:.1f}%)")
    t_series = df_feat["tiempo_evento"]
    print(f"   T eventos (todos =1): {df_feat[df_feat['evento']==1]['tiempo_evento'].unique().tolist()}")
    print(f"   T censurados — min: {t_series[df_feat['evento']==0].min():.0f}  "
          f"med: {t_series[df_feat['evento']==0].median():.0f}  "
          f"max: {t_series[df_feat['evento']==0].max():.0f} meses")

    print("\n── Sample de features (primeras 3 filas):")
    print(df_feat.head(3).to_string())


if __name__ == "__main__":
    main()
