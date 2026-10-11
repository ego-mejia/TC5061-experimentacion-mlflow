"""
train.py
Entrenamiento parametrizado de un clasificador (Regresión Logística) sobre UCI Adult
(Census Income), instrumentado con MLflow Tracking.

Uso:
    python train.py                      # corre con los valores por defecto
    python train.py --C 0.1 --penalty l1 --seed 7
    python train.py --help
"""

import argparse
import os
import random
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")  # backend sin ventanas: el script corre en terminal
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
NUM_FEATURES = ["age", "education_num", "hours_per_week"]
LOG_FEATURES = ["capital_gain", "capital_loss"]  # se les aplica log1p en el pipeline
CAT_FEATURES = ["workclass", "marital_status", "occupation", "relationship", "race", "sex",
                "native_country"]
FEATURES = NUM_FEATURES + LOG_FEATURES + CAT_FEATURES
TARGET = "target"
MISSING_COLS = ["workclass", "occupation", "native_country"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena una Regresión Logística sobre UCI Adult y registra la corrida en MLflow.")
    g = p.add_argument_group("hiperparámetros del modelo")
    g.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización (menor = más regularización). Default: 1.0")
    g.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                   help="Tipo de regularización. Default: l2")
    g.add_argument("--max-iter", type=int, default=1000,
                   help="Iteraciones máximas del solver. Default: 1000")
    g.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                   help="'balanced' pondera las clases por su frecuencia inversa. Default: balanced")
    g = p.add_argument_group("preprocesamiento y partición")
    g.add_argument("--ohe-min-frequency", type=int, default=20,
                   help="Categorías con menos registros se agrupan como 'infrecuente'. Default: 20")
    g.add_argument("--test-size", type=float, default=0.20,
                   help="Proporción del conjunto de prueba. Default: 0.20")
    g.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria (partición y modelo). Default: 42")
    g = p.add_argument_group("datos y MLflow")
    g.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                   help="Carpeta con adult.data y adult.test. Default: ./data")
    g.add_argument("--experiment-name", default="TC5061-adult-income-experimentacion",
                   help="Nombre del experimento de MLflow. Default: TC5061-adult-income-experimentacion")
    g.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                   help="URI de tracking de MLflow. Default: sqlite:///mlflow.db")
    g.add_argument("--run-name", default=None, help="Nombre de la corrida (opcional).")
    return p.parse_args()


def set_seed(seed):
    # Fija todas las fuentes de aleatoriedad del proceso
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(data_dir):
    # skipinitialspace elimina el espacio que sigue a cada coma en los archivos originales
    train = pd.read_csv(os.path.join(data_dir, "adult.data"), names=COLUMNS, skipinitialspace=True)
    test = pd.read_csv(os.path.join(data_dir, "adult.test"), names=COLUMNS, skipinitialspace=True,
                       skiprows=1)
    return pd.concat([train, test], ignore_index=True)


def clean_data(df_raw):
    # Mismas decisiones de limpieza del notebook baseline (tarea 01)
    df = df_raw.copy()
    str_cols = df.select_dtypes(exclude="number").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    df[TARGET] = (df["income"].str.rstrip(".") == ">50K").astype(int)
    df = df.drop(columns=["income", "education", "fnlwgt"])
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df = df.replace("?", np.nan)
    df[MISSING_COLS] = df[MISSING_COLS].fillna("Unknown")
    assert df.isna().sum().sum() == 0
    df[CAT_FEATURES] = df[CAT_FEATURES].astype("category")
    return df


def build_pipeline(args):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scaler", StandardScaler())]), LOG_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=args.ohe_min_frequency, sparse_output=False),
         CAT_FEATURES),
    ])
    classifier = LogisticRegression(
        C=args.C, penalty=args.penalty, solver="liblinear",  # liblinear soporta l1 y l2
        max_iter=args.max_iter,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.seed)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba)}


def plot_evaluation(model, X_test, y_test, path):
    # Matriz de confusión + curva ROC sobre el conjunto de prueba
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test),
                                            display_labels=["<=50K", ">50K"], cmap="Blues",
                                            values_format=",", ax=axes[0])
    axes[0].set_title("Matriz de confusión (test)")
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1], name="LogReg")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.5)")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Curva ROC (test)")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def log_to_mlflow(args, model, train_m, test_m, X_train, X_test, y_test, n_rows):
    mlflow.set_tags({"model_type": "LogisticRegression", "dataset": "UCI Adult (Census Income)",
                     "author": "Fernando Mejía Laguna"})
    mlflow.log_params({
        "C": args.C, "penalty": args.penalty, "max_iter": args.max_iter,
        "class_weight": args.class_weight, "ohe_min_frequency": args.ohe_min_frequency,
        "test_size": args.test_size, "seed": args.seed, "solver": "liblinear",
        "n_rows_clean": n_rows, "n_train": len(X_train), "n_test": len(X_test),
    })
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
    mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})

    with tempfile.TemporaryDirectory() as tmp:
        plot_path = os.path.join(tmp, "evaluation.png")
        plot_evaluation(model, X_test, y_test, plot_path)
        mlflow.log_artifact(plot_path)
    mlflow.log_text(classification_report(y_test, model.predict(X_test),
                                          target_names=["<=50K", ">50K"], digits=4),
                    "classification_report.txt")
    mlflow.log_dict({"train": train_m, "test": test_m}, "metrics.json")
    mlflow.sklearn.log_model(model, name="model",
                             signature=infer_signature(X_train, model.predict(X_train)),
                             input_example=X_train.head(5))


def main():
    args = parse_args()
    set_seed(args.seed)

    df = clean_data(load_data(args.data_dir))
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=args.test_size, stratify=df[TARGET],
        random_state=args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        model = build_pipeline(args).fit(X_train, y_train)
        train_m, test_m = evaluate(model, X_train, y_train), evaluate(model, X_test, y_test)
        log_to_mlflow(args, model, train_m, test_m, X_train, X_test, y_test, len(df))
        print(f"run_id={run.info.run_id}")
        print("test: " + " | ".join(f"{k}={v:.4f}" for k, v in test_m.items()))


if __name__ == "__main__":
    main()
