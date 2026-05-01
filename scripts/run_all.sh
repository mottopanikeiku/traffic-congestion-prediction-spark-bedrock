#!/usr/bin/env bash
# Reproducible end-to-end run.
# Runs:
#   1. unit tests
#   2. full pipeline (profile + train + evaluate + alerts + charts)
#   3. report build
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export TRAFFIC_BEDROCK_MODE="${TRAFFIC_BEDROCK_MODE:-mock}"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"

echo "==> Running unit tests..."
python -m pytest tests/ -q

echo "==> Running end-to-end pipeline..."
python -m traffic.cli train

echo "==> Building report..."
python scripts/build_report.py

echo "==> Done. Outputs:"
echo "    Report:      reports/Final_Project_Report.docx"
echo "    Figures:     reports/figures/"
echo "    Metrics:     reports/results/model_results.csv"
echo "    Alerts:      reports/results/sample_alerts.json"
