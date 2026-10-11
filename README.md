# TC5061 · Código de experimentación base con MLflow

Alumno: Fernando Mejía Laguna · Curso: TC5061.10 – MLOps

Script de entrenamiento parametrizado (Regresión Logística sobre UCI Adult / Census Income, predice si el
ingreso es >50K) con registro de parámetros, métricas, artefactos y modelo en MLflow Tracking.

## Contenido
- `train.py` – script de entrenamiento con argparse e instrumentación MLflow.
- `run_experiments.sh` – lanza las 8 configuraciones comparadas.
- `requirements.txt` – dependencias con versiones fijas (Python 3.13).
- `data/` – dataset UCI Adult (`adult.data`, `adult.test`).
- `notebook_exploracion.ipynb` – notebook de la semana anterior, solo para exploración (EDA).

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución
```bash
python train.py                                   # valores por defecto
python train.py --C 0.1 --penalty l1 --seed 7 --run-name mi-corrida
python train.py --help                            # lista de argumentos
./run_experiments.sh                              # las 8 corridas del experimento
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la regularización |
| `--penalty` | l2 | `l1` o `l2` |
| `--max-iter` | 1000 | Iteraciones máximas |
| `--class-weight` | balanced | `balanced` o `none` |
| `--ohe-min-frequency` | 20 | Frecuencia mínima por categoría en el One-Hot |
| `--test-size` | 0.20 | Proporción de prueba |
| `--seed` | 42 | Semilla aleatoria |
| `--experiment-name` | TC5061-adult-income-experimentacion | Experimento MLflow |

## Ver resultados en MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```
Cada corrida registra: parámetros (`log_param`), métricas de train y test (accuracy, precision, recall, f1,
roc_auc), artefactos (`evaluation.png`, `classification_report.txt`, `metrics.json`) y el modelo
(`mlflow.sklearn.log_model`).

## Reproducibilidad
La semilla (`--seed`) controla `random`, `numpy`, la partición estratificada train/test y el
`random_state` del modelo. Dos ejecuciones con los mismos argumentos producen métricas idénticas.
