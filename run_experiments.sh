#!/usr/bin/env bash
# Corre las 8 configuraciones del experimento (todas bajo el mismo experimento de MLflow).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY train.py --run-name 01-default
$PY train.py --run-name 02-C0.01        --C 0.01
$PY train.py --run-name 03-C0.1         --C 0.1
$PY train.py --run-name 04-C10          --C 10
$PY train.py --run-name 05-l1-C0.1      --penalty l1 --C 0.1
$PY train.py --run-name 06-sin-balanceo --class-weight none
$PY train.py --run-name 07-ohe-freq200  --ohe-min-frequency 200
$PY train.py --run-name 08-seed7        --seed 7
