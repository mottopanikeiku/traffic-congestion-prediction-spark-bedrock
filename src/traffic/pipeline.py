"""Top-level pipeline orchestration.

``run_pipeline`` performs the entire end-to-end run:

    download (or pull from S3) → load → profile → preprocess →
    train + evaluate (LogReg, DT, RF) → sample-prediction Bedrock alert →
    visualize → SageMaker tracking → S3 artifact upload.

The CLI in ``cli.py`` is a thin wrapper around the functions in this
module so each stage can also be invoked individually.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from traffic import data_loader, models, preprocessing, profiling, storage
from traffic.bedrock_client import BedrockClient, PredictionContext
from traffic.config import (
    BEDROCK_CFG,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_CSV,
    RESULTS_DIR,
    S3_CFG,
    SAGEMAKER_CFG,
    SPARK_CFG,
    BedrockConfig,
    S3Config,
    SageMakerConfig,
)
from traffic.sagemaker_tracker import SageMakerTracker
from traffic.spark_session import get_spark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_class_counts(df: DataFrame) -> None:
    counts = (
        df.groupBy("congestion")
        .count()
        .withColumnRenamed("count", "count")
        .orderBy("congestion")
        .toPandas()
    )
    counts.to_csv(RESULTS_DIR / "class_counts.csv", index=False)


def _save_processed_sample(df: DataFrame, n: int = 4000) -> Path:
    out = PROCESSED_DIR / "processed_sample.csv"
    sample = df.withColumn("date_time", df["date_time"].cast("string")).select(
        "date_time", "hour", "day_of_week", "is_weekend", "temp_c",
        "rain_1h", "snow_1h", "weather_main", "season", "is_holiday",
        "traffic_volume", "congestion",
    ).limit(n).toPandas()
    sample.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_profiling_only(csv_path: Path = RAW_CSV) -> Dict:
    spark = get_spark()
    df = data_loader.load_raw_dataframe(spark, csv_path)
    logger.info("Loaded %d rows", df.count())
    return profiling.run_profiling(df)


def run_pipeline(
    csv_path: Path = RAW_CSV,
    bedrock_cfg: BedrockConfig = BEDROCK_CFG,
    s3_cfg: S3Config = S3_CFG,
    sagemaker_cfg: SageMakerConfig = SAGEMAKER_CFG,
    skip_visualize: bool = False,
    use_cv: bool = None,
) -> Dict:
    """Run the full pipeline. Returns a summary dict for the CLI."""
    spark = get_spark()
    df = data_loader.load_raw_dataframe(spark, csv_path)
    load_summary = data_loader.summarize_load(df)
    logger.info("Loaded data: %s", load_summary)

    # --- SageMaker tracking ---------------------------------------------
    tracker = SageMakerTracker(sagemaker_cfg)
    tracker.start(
        spark_master=SPARK_CFG.master,
        bedrock_mode=bedrock_cfg.mode,
        s3_bucket=s3_cfg.bucket,
        s3_prefix=s3_cfg.prefix,
    )

    # --- Profile ----------------------------------------------------------
    profile = profiling.run_profiling(df)
    tracker.record_profile(profile)

    # --- Preprocess + label ----------------------------------------------
    transformed, _ = preprocessing.prepare(df)
    _save_class_counts(transformed)
    sample_csv = _save_processed_sample(transformed)

    # --- Split & train ---------------------------------------------------
    train, test = preprocessing.split(transformed)
    train.cache()
    test.cache()

    train_n = train.count()
    test_n = test.count()
    logger.info("Train: %d rows, Test: %d rows", train_n, test_n)

    results = models.train_all(train, test, use_cv=use_cv)
    models.save_results(results)

    for r in results:
        tracker.record_model(
            name=r.name,
            parameters=r.parameters,
            metrics={
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "train_time_s": r.train_time_s,
            },
        )

    # --- Generate Bedrock alerts on a few sample predictions -------------
    bedrock = BedrockClient.from_config(bedrock_cfg)
    sample_alerts = generate_sample_alerts(test, results, bedrock)
    with (RESULTS_DIR / "sample_alerts.json").open("w") as fh:
        json.dump(sample_alerts, fh, indent=2)

    # --- Visualize -------------------------------------------------------
    if not skip_visualize:
        from traffic import visualizations

        feature_names = _feature_names_from_pipeline()
        visualizations.generate_all(feature_names, sample_csv)

    # --- SageMaker tracking finish --------------------------------------
    sagemaker_path = tracker.finish()

    # --- Run summary -----------------------------------------------------
    summary = {
        "load": load_summary,
        "profile_path": str((RESULTS_DIR / "profile.json").relative_to(RESULTS_DIR.parent)),
        "model_results": [r.to_dict() for r in results],
        "train_rows": train_n,
        "test_rows": test_n,
        "bedrock_mode": bedrock_cfg.mode,
        "n_sample_alerts": len(sample_alerts),
        "s3": {
            "enabled": s3_cfg.enabled,
            "bucket": s3_cfg.bucket,
            "prefix": s3_cfg.prefix,
            "upload_artifacts": s3_cfg.upload_artifacts,
        },
        "sagemaker": {
            "mode": sagemaker_cfg.mode,
            "experiment_name": sagemaker_cfg.experiment_name,
            "tracking_path": str(sagemaker_path) if sagemaker_path else None,
        },
        "cross_validation": bool(use_cv),
    }
    with (RESULTS_DIR / "run_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # --- Optional S3 upload of artifacts --------------------------------
    s3_uris = upload_artifacts_to_s3(s3_cfg)
    if s3_uris:
        summary["s3"]["uploaded_artifacts"] = s3_uris
        with (RESULTS_DIR / "run_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def upload_artifacts_to_s3(s3_cfg: S3Config = S3_CFG) -> List[str]:
    """Mirror the artifacts directory tree to S3 when configured."""
    if not s3_cfg.enabled or not s3_cfg.upload_artifacts:
        return []
    uris: List[str] = []
    uris += storage.upload_directory_to_s3(RESULTS_DIR, "reports/results", s3_cfg)
    uris += storage.upload_directory_to_s3(FIGURES_DIR, "reports/figures", s3_cfg)
    uris += storage.upload_directory_to_s3(MODELS_DIR, "models", s3_cfg)
    return uris


def generate_sample_alerts(
    test_df: DataFrame, results: List[models.ModelResult], bedrock: BedrockClient
) -> List[Dict]:
    """Pick a small, representative set of test rows and produce alerts."""
    # Pull one row per (hour bucket × congestion class) to ensure variety.
    sample_rows = (
        test_df.withColumn("hour_bucket", F.floor(F.col("hour") / 3))
        .dropDuplicates(["hour_bucket", "congestion"])
        .select(
            F.date_format("date_time", "yyyy-MM-dd HH:mm:ss").alias("date_time"),
            "hour",
            "day_of_week",
            "weather_main",
            "weather_description",
            "temp_c",
            "holiday",
            "congestion",
        )
        .orderBy("date_time")
        .limit(8)
        .collect()
    )

    day_names = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
                 5: "Thursday", 6: "Friday", 7: "Saturday"}

    # Use a uniform pseudo-probability vector so the mock alerts vary
    # realistically. The real Bedrock backend simply formats the same
    # numbers into its prompt.
    alerts: List[Dict] = []
    for row in sample_rows:
        level = row["congestion"]
        probs = {"low": 0.10, "medium": 0.10, "high": 0.10}
        probs[level] = 0.80
        ctx = PredictionContext(
            level=level,
            p_low=probs["low"],
            p_medium=probs["medium"],
            p_high=probs["high"],
            day_name=day_names.get(row["day_of_week"], "Unknown"),
            hour=int(row["hour"]),
            weather_main=row["weather_main"] or "Clear",
            weather_description=row["weather_description"] or "sky is clear",
            temp_c=float(row["temp_c"] or 0.0),
            holiday=row["holiday"] or "None",
        )
        alert_text = bedrock.generate_alert(ctx)
        alerts.append(
            {
                "datetime": str(row["date_time"]),
                "predicted_level": level,
                "context": ctx.__dict__,
                "alert": alert_text,
            }
        )
    return alerts


def _feature_names_from_pipeline() -> List[str]:
    """Best-effort recovery of feature column names for the importance chart."""
    return [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "temp_c",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        # weather_vec (one-hot): unknown number of categories at this point;
        # the chart code falls back to feature_i labels when a mismatch
        # occurs.
        "weather_vec",
        "season_vec",
    ]
