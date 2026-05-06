# Traffic Congestion Prediction Using Apache Spark ML & Amazon Bedrock

CSP554 — Big Data Technologies, Final Project
Illinois Institute of Technology
*Aidan Ash · Paul Balea · Fatih Cetin · Atishay Jain · Tomas Rebelatto*

---

## 1. What this project does

End-to-end big-data pipeline that predicts hourly congestion on the
Metro Interstate I-94 corridor and turns each prediction into a
plain-language traffic alert.

```
┌─── Data sources ────┐    ┌─── Apache Spark ML ────┐    ┌── Amazon Bedrock ──┐
│  S3 (s3a://…)       │ →  │  profile → preprocess  │ →  │  Anthropic Claude  │
│  UCI HTTP download  │    │  Logistic / DT / RF    │    │  3 Haiku  (or mock)│
│  Synthetic fallback │    │  3-fold CV (optional)  │    └────────────────────┘
└─────────────────────┘    └────────────────────────┘             │
            ↓                          ↓                          ↓
   reports/results/profile_*.csv      models/<model>/    reports/results/sample_alerts.json
                              \         |        /
                               → reports/figures/*.png  →  reports/Final_Project_Report.docx
                              /                       \
              SageMaker Experiment ←——— pipeline tracking ———→ S3 artifact mirror
```

The pipeline:

1. Loads the **Metro Interstate Traffic Volume** dataset (S3 → local
   CSV → HTTP download → calibrated synthetic fallback).
2. **Profiles** the data with the Spark DataFrame API
   (`describe`, `groupBy`, `approxQuantile`, null audits, distributions).
3. Engineers temporal + meteorological features through a Spark ML
   `Pipeline` (`StringIndexer` → `OneHotEncoder` → `VectorAssembler`).
4. Trains and evaluates **Logistic Regression, Decision Tree, and
   Random Forest** classifiers on a 3-class congestion target
   (low / medium / high). 3-fold `CrossValidator` is available via
   `--cv`.
5. Sends each prediction through **Amazon Bedrock**
   (`anthropic.claude-3-haiku-20240307-v1:0`) to produce a 2–3 sentence
   plain-language alert. A deterministic **mock backend** is included so
   the project runs end-to-end without an AWS account.
6. **Mirrors artifacts to Amazon S3** when `--s3-bucket` / `--s3-upload`
   are set (or `TRAFFIC_S3_BUCKET=…` is exported), and **registers an
   Amazon SageMaker Experiment + Trial + TrialComponent** for the run
   when `--sagemaker-mode real` is set.
7. Renders ten profiling + evaluation visualizations and a complete
   **Final_Project_Report.docx**.

AWS-dependent features are switchable so the entire deliverable is
exercisable with zero AWS configuration: Bedrock and SageMaker include
mock modes, and S3 reads/uploads stay disabled unless a bucket is
explicitly configured.

---

## 2. Project layout

```
traffic_congestion_prediction/
├── README.md                         # this file
├── requirements.txt
├── data/
│   ├── raw/                          # downloaded / synthesised CSV
│   └── processed/                    # sampled feature CSV for plotting
├── models/                           # persisted Spark ML models
│   ├── logisticregression/
│   ├── decisiontree/
│   └── randomforest/
├── reports/
│   ├── Final_Project_Report.docx     # full graded deliverable
│   ├── figures/                      # 10 PNG charts
│   └── results/                      # profile.json, model_results.{json,csv},
│                                     # sample_alerts.json, run_summary.json,
│                                     # per_class_metrics.json, sagemaker_tracking.json
├── scripts/
│   ├── run_all.sh                    # convenience driver
│   ├── build_extras.py               # per-class metrics + architecture diagram
│   └── build_report.py               # rebuilds Final_Project_Report.docx
├── src/traffic/                      # 13 unit-tested modules
│   ├── __init__.py
│   ├── __main__.py                   # `python -m traffic` entry point
│   ├── bedrock_client.py             # mock + real Bedrock backends
│   ├── cli.py                        # argparse subcommands
│   ├── config.py                     # paths, model + Bedrock + Spark + S3 + SageMaker config
│   ├── data_loader.py                # S3 → local → HTTP → synthetic
│   ├── models.py                     # train + evaluate 3 classifiers, CV-ready
│   ├── pipeline.py                   # end-to-end orchestration
│   ├── preprocessing.py              # feature engineering pipeline
│   ├── profiling.py                  # DataFrame-API profiling
│   ├── sagemaker_tracker.py          # off / mock / real SageMaker Experiments
│   ├── spark_session.py              # SparkSession factory (with S3A connector)
│   ├── storage.py                    # local + S3 storage helpers
│   ├── synthetic_data.py             # offline-friendly fallback dataset
│   └── visualizations.py             # all 10 charts
└── tests/test_smoke.py               # 12-test pytest suite
```

---

## 3. Prerequisites

| Requirement | Version |
| ----------- | ------- |
| Python      | 3.10 +  |
| Java        | 11 / 17 recommended; Java 21 also works locally |
| Spark       | 3.5.x   |

```bash
pip install -r requirements.txt
```

The pipeline runs Spark in `local[*]` mode by default — no cluster is
required.

Windows note: Spark's local model persistence uses Hadoop's local
filesystem layer, so Windows runs need `winutils.exe` available under
`%HADOOP_HOME%\bin`. Use a Java version supported by Spark/Hadoop
(11, 17, or 21); Java 25 is not compatible with the Hadoop 3.3.x runtime
used by Spark 3.5.

---

## 4. Quickstart

### Run everything end-to-end (mock Bedrock, no AWS needed)

```bash
PYTHONPATH=src python -m traffic.cli all
```

This runs: download → load → profile → preprocess → train all 3 models →
evaluate → generate sample Bedrock alerts → render all 10 figures.

### Rebuild the .docx report

```bash
python scripts/build_extras.py     # per-class metrics + architecture diagram
python scripts/build_report.py     # writes reports/Final_Project_Report.docx
```

### Run the 12-test pytest suite

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

---

## 5. CLI reference

```
python -m traffic.cli <subcommand> [options]
```

### Subcommands

| Subcommand | Purpose |
| ---------- | ------- |
| `download` | Fetch (or synthesise) the raw CSV |
| `profile`  | Run profiling only (no training) |
| `train`    | Profile + preprocess + train 3 models + alerts + figures |
| `alert`    | Produce a single Bedrock alert from CLI inputs |
| `upload`   | Push existing local artifacts to S3 (no retraining) |
| `all`      | Alias for `train` |

### Common flags

```
--csv PATH                          Override raw CSV path
--aws-region REGION                 AWS region (env: AWS_REGION)

# Bedrock
--bedrock-mode {mock,real}          Default: mock (env: TRAFFIC_BEDROCK_MODE)
--bedrock-model MODEL_ID            Default: anthropic.claude-3-haiku-20240307-v1:0

# Amazon S3
--s3-bucket BUCKET                  Read raw CSV from s3a://BUCKET/PREFIX/raw/
                                    (env: TRAFFIC_S3_BUCKET)
--s3-prefix PREFIX                  Default: traffic-congestion-prediction
--s3-upload                         Mirror artifacts back to S3 after the run

# Amazon SageMaker
--sagemaker-mode {off,mock,real}    Default: off (env: TRAFFIC_SAGEMAKER_MODE)
--sagemaker-experiment NAME         Default: traffic-congestion-prediction

# Modeling
--cv                                Wrap each estimator in 3-fold CrossValidator
--skip-visualize                    Skip chart generation
```

### Examples

```bash
# Default offline run (mock Bedrock, no AWS)
PYTHONPATH=src python -m traffic.cli all

# Cross-validated training
PYTHONPATH=src python -m traffic.cli train --cv

# Read raw CSV from S3, mirror artifacts back to S3, log to SageMaker
PYTHONPATH=src python -m traffic.cli train \
    --s3-bucket my-bdt-bucket --s3-prefix csp554/traffic --s3-upload \
    --sagemaker-mode real

# One-shot alert from CLI inputs (real Bedrock)
PYTHONPATH=src python -m traffic.cli alert \
    --level high --hour 8 --day Monday --weather Rain --temp 12 \
    --bedrock-mode real

# Profile only
PYTHONPATH=src python -m traffic.cli profile

# Push existing local artifacts to S3 (no retraining)
PYTHONPATH=src python -m traffic.cli upload --s3-bucket my-bdt-bucket
```

---

## 6. Configuration via environment variables

Everything in `src/traffic/config.py` can be overridden via env vars.
The most useful ones:

| Variable                       | Effect |
| ------------------------------ | ------ |
| `TRAFFIC_BEDROCK_MODE`         | `mock` (default) or `real` |
| `TRAFFIC_BEDROCK_MODEL_ID`     | Override the Bedrock model id |
| `TRAFFIC_MODELS_DIR`           | Where to persist trained Spark models |
| `TRAFFIC_S3_BUCKET`            | S3 bucket for raw data + artifacts |
| `TRAFFIC_S3_PREFIX`            | S3 key prefix (default `traffic-congestion-prediction`) |
| `TRAFFIC_S3_UPLOAD`            | `1` to mirror artifacts to S3 after the run |
| `TRAFFIC_SAGEMAKER_MODE`       | `off` (default), `mock`, or `real` |
| `TRAFFIC_SAGEMAKER_EXPERIMENT` | SageMaker experiment name |
| `TRAFFIC_SAGEMAKER_TRIAL`      | SageMaker trial-name prefix |
| `TRAFFIC_USE_CV`               | `1` to enable 3-fold CV (same as `--cv`) |
| `AWS_REGION`                   | Region for Bedrock / S3 / SageMaker |
| `SPARK_MASTER`                 | Override Spark master (e.g. `spark://…`) |
| `SPARK_LOCAL_IP`               | Auto-set to `127.0.0.1`; override if needed |

---

## 7. AWS integrations

The project implements integration points for all four AWS services named
in the proposal. The shipped run is local/offline, with Bedrock and
SageMaker exercised in mock mode.

### Amazon Bedrock (real mode)

```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...   # or use an IAM role / SageMaker exec role
export AWS_SECRET_ACCESS_KEY=...

PYTHONPATH=src python -m traffic.cli train --bedrock-mode real
```

Requires Anthropic Claude 3 Haiku model access enabled in the AWS
account. Override the model id with `--bedrock-model` (any
Bedrock-hosted model with the Anthropic Messages API contract works).

### Amazon S3 (raw data + artifact mirror)

```bash
# Upload your CSV first (or just put the synthetic fallback there):
aws s3 cp data/raw/Metro_Interstate_Traffic_Volume.csv \
    s3://my-bdt-bucket/csp554/traffic/raw/

# Pipeline reads from s3a://… and mirrors artifacts back:
PYTHONPATH=src python -m traffic.cli train \
    --s3-bucket my-bdt-bucket --s3-prefix csp554/traffic --s3-upload
```

The SparkSession is built with the Hadoop S3A connector
(`org.apache.hadoop:hadoop-aws:3.3.4`) and the default AWS credential
provider chain, so no extra configuration is required.

### AWS IAM

Used by every `boto3` call (Bedrock, S3, SageMaker). The standard
credential chain — env vars, `~/.aws/credentials`, EC2/ECS task role,
SageMaker execution role — is honoured automatically.

### Amazon SageMaker (Experiment tracking)

Three modes, selected by `--sagemaker-mode {off,mock,real}` (env
`TRAFFIC_SAGEMAKER_MODE`):

* `off` (default) — no SageMaker calls.
* `mock` — write the same payload that would be sent to SageMaker into
  `reports/results/sagemaker_tracking.json` so the integration is
  verifiable offline (used in the shipped run).
* `real` — call `boto3.client('sagemaker')` to register an
  `Experiment` + `Trial` + `TrialComponent` per pipeline run, with
  hyperparameters and metrics attached. Requires the standard
  `AmazonSageMakerFullAccess`-style policy.

---

## 8. Cross-validation

Pass `--cv` to wrap each classifier in a 3-fold `CrossValidator` over a
small `ParamGridBuilder` grid:

* **Logistic Regression** — `regParam ∈ {0.0, 0.01, 0.1}`
  × `elasticNetParam ∈ {0.0, 0.5}`
* **Decision Tree** — `maxDepth ∈ {6, 8, 10, 12}`
* **Random Forest** — `numTrees ∈ {60, 100}` × `maxDepth ∈ {12, 15}`

The best parameter set is recorded in
`reports/results/model_results.json` under
`model_results[*].parameters`.

```bash
PYTHONPATH=src python -m traffic.cli train --cv
```

---

## 9. Artifacts produced by `train`

| Path | Description |
| ---- | ----------- |
| `data/raw/Metro_Interstate_Traffic_Volume.csv` | Raw dataset (real or synthetic) |
| `data/processed/processed_sample.csv` | 4 000-row feature sample for plotting |
| `models/<name>/` | Persisted Spark ML model (`logisticregression`, `decisiontree`, `randomforest`) |
| `reports/results/profile.json` | Full profiling output (machine-readable) |
| `reports/results/profile_*.csv` | Per-aggregate profiling tables (numeric, weather, holiday, hourly, dow, monthly) |
| `reports/results/class_counts.csv` | Distribution of the 3-class congestion label |
| `reports/results/model_results.json` | Per-model metrics + confusion matrix + parameters |
| `reports/results/model_results.csv` | Same metrics, human-readable |
| `reports/results/per_class_metrics.json` | Per-class precision / recall / F1 |
| `reports/results/sample_alerts.json` | 8 Bedrock alerts produced from test predictions |
| `reports/results/sagemaker_tracking.json` | SageMaker Experiment payload (mock or real) |
| `reports/results/run_summary.json` | Top-level run summary |
| `reports/figures/fig_*.png` | 10 figures embedded in the report |
| `reports/Final_Project_Report.docx` | Full graded report |

When `--s3-upload` is set, the same files are also mirrored to
`s3://{bucket}/{prefix}/{reports,models,...}/`.

---

## 10. Deliverables (what to upload to Canvas)

Everything required by the assignment is in this folder.

1. **`reports/Final_Project_Report.docx`** — the full report with the
   sections listed in the professor's note: Abstract, Introduction
   (motivation, research question, contributions), Literature Review,
   Methodology (architecture, data source, modules, feature engineering,
   modelling, AWS integrations, reproducibility), Results (profiling,
   aggregate metrics, per-class metrics, Bedrock alerts), Discussion
   (model comparison, RF vs DT, Bedrock value, limitations,
   implications, future work), Conclusion, References, Appendices A
   through E.
2. **Source code** — `src/traffic/*.py` (13 modules, ~1,800 lines),
   `tests/test_smoke.py` (12 unit tests), `scripts/*.py`, `requirements.txt`.
3. **Visualizations** — 10 PNG figures under `reports/figures/`.
4. **Result artifacts** — JSON/CSV files under `reports/results/`,
   including `sagemaker_tracking.json`.
5. **Trained models** — `models/{logisticregression,decisiontree,randomforest}/`.
6. **README** — this file.

---

## 11. Reproducibility

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q              # 12 passing tests
PYTHONPATH=src python -m traffic.cli train             # end-to-end
python scripts/build_extras.py
python scripts/build_report.py                         # rewrites the .docx
```

Random seeds are fixed at the dataset, train/test split, and
model-estimator levels (`MODEL_CFG.seed = 42`). The synthetic-data
fallback is also seeded, so two runs on the same machine produce
identical metrics.

---

## 12. Expected metrics (shipped run)

Run on a calibrated synthetic dataset (52,551 rows, 2012-10-02 →
2018-09-30, Minneapolis-realistic temperatures) — same code path as the
real UCI CSV, just a different data source.

| Model              | Accuracy | Precision (w) | Recall (w) | F1 (w) | Train time |
| ------------------ | -------- | ------------- | ---------- | ------ | ---------- |
| Logistic Regression| 0.5570   | 0.5232        | 0.5570     | 0.5298 | ~6 s       |
| Decision Tree      | 0.9126   | 0.9134        | 0.9126     | 0.9121 | ~1 s       |
| **Random Forest**  | **0.9161** | **0.9179**  | **0.9161** | **0.9149** | ~9 s |

Random Forest leads on weighted F1 by ~0.3 percentage points over the
single Decision Tree, which lines up with the "ensembles dominate" claim
in the literature review (Chen & Guestrin, 2016).

---

## 13. Notes on the dataset

The pipeline first attempts to read the real Metro Interstate Traffic
Volume dataset from S3 (`s3a://{bucket}/{prefix}/raw/...`), then from
the local `data/raw/` folder, then from the UCI machine-learning
repository. If all three are unreachable, a deterministic synthetic
generator is used. The synthetic generator is calibrated against the
published dataset's distributional characteristics: seasonal Minneapolis
temperature curve (mean 8 °C, range -19 °C to +33 °C), hourly +
weekday rush-hour seasonality, weather-frequency distribution,
holiday markers, and traffic-volume range.

To regenerate using the real UCI CSV, ensure `archive.ics.uci.edu` is
reachable (or upload the CSV to S3 and set `--s3-bucket`), delete
`data/raw/Metro_Interstate_Traffic_Volume.csv`, and rerun
`python -m traffic.cli all`.
