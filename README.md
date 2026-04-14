# SAT Crédito — Sistema de Alerta Temprana

Sistema de scoring de riesgo crediticio para cartera de consumo,
vinculando datos internos con indicadores macroeconómicos del BCP.

## Estructura del proyecto

```
sat_credito/
├── data/
│   ├── raw/                    # Datos originales (no modificar)
│   │   └── bd_consumo_muestra.csv
│   └── processed/              # Datos generados por los scripts
│       ├── features.parquet    # Features listos para modelado
│       ├── alertas_sat.csv     # Salida: alertas por cliente
│       └── feature_importance.csv
├── notebooks/                  # Análisis exploratorio (EDA)
├── src/
│   ├── 01_build_features.py    # Construcción de features + enriquecimiento BCP
│   ├── 02_train_model.py       # Entrenamiento y evaluación del modelo
│   ├── 03_score_alertas.py     # Scoring sobre nuevos datos
│   └── modelo/
│       ├── modelo_sat.pkl      # Modelo serializado
│       └── metricas.json       # Métricas de evaluación
└── README.md
```

## Flujo de ejecución

```bash
# 1. Construir features (fecha supositoria + indicadores BCP)
python src/01_build_features.py

# 2. Entrenar modelo
python src/02_train_model.py

# 3. Generar alertas
python src/03_score_alertas.py

# Opcional: score sobre nueva base
python src/03_score_alertas.py --input nueva_base.parquet --output alertas_julio.csv
```

## Lógica de fecha supositoria

Como la base no contiene fecha de desembolso, se estima así:

- **Crédito Nuevo**: `fecha_desembolso = hoy − cant_cuotas meses`
- **Crédito Renovado**: `fecha_desembolso = hoy − (cant_cuotas + 3) meses`
  (offset de 3 meses por ciclo promedio de renovación)

Esto permite vincular cada crédito al contexto macroeconómico
del período en que fue originado.

## Indicadores BCP incorporados

| Variable               | Fuente BCP                            | Frecuencia |
|------------------------|---------------------------------------|------------|
| `tpm`                  | Tasa de Política Monetaria            | Mensual    |
| `mora_sistema_consumo` | Informe de Estabilidad Financiera     | Bimestral  |
| `ipc_interanual`       | Informe de Inflación                  | Mensual    |
| `usd_pyg`              | Cotización referencial interbancaria  | Diaria/mes |

**Actualización**: Editar el dict `BCP_DATA` en `01_build_features.py`
con los nuevos valores mensuales del BCP.

## Niveles de alerta

| Nivel | Criterio               | Acción sugerida                     |
|-------|------------------------|-------------------------------------|
| ALTO  | prob_mora ≥ p75        | Contacto preventivo inmediato       |
| MEDIO | prob_mora ≥ p40        | Monitoreo reforzado                 |
| BAJO  | prob_mora < p40        | Seguimiento rutinario               |

## Variable objetivo

`atraso_30`: indicador binario de si el cliente entró en mora ≥30 días.
Es la señal de alerta más temprana y maximiza el tiempo de reacción.

## Dependencias

```
pandas
numpy
scikit-learn
python-dateutil
pyarrow       # para parquet
```

```bash
pip install pandas numpy scikit-learn python-dateutil pyarrow
```
