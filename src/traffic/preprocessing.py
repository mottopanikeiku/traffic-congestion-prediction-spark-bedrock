"""Feature engineering and Spark ML preprocessing pipeline.

The pipeline takes the raw Spark DataFrame and produces a `features` vector
plus a `label` index suitable for any of the three classifiers in
`models.py`.

Steps:
1. Derive temporal features: hour, day_of_week, month, season, is_weekend.
2. Derive holiday flag.
3. Convert temperature from Kelvin to Celsius (more interpretable).
4. Bucket traffic volume into a 3-class congestion label
   (low / medium / high) using the thresholds from `ModelConfig`.
5. StringIndexer + OneHotEncoder for categorical columns
   (weather_main, season).
6. VectorAssembler to assemble the final feature vector.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from traffic.config import MODEL_CFG

logger = logging.getLogger(__name__)


CONGESTION_LABELS = ["low", "medium", "high"]


def add_derived_columns(df: DataFrame) -> DataFrame:
    """Add hour / day_of_week / month / season / is_weekend / holiday flags."""
    df = df.withColumn("hour", F.hour("date_time"))
    df = df.withColumn("day_of_week", F.dayofweek("date_time"))  # 1=Sun..7=Sat
    df = df.withColumn("month", F.month("date_time"))
    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0),
    )
    df = df.withColumn(
        "season",
        F.when(F.col("month").isin(12, 1, 2), "winter")
        .when(F.col("month").isin(3, 4, 5), "spring")
        .when(F.col("month").isin(6, 7, 8), "summer")
        .otherwise("fall"),
    )
    df = df.withColumn(
        "is_holiday",
        F.when(F.col("holiday") == "None", 0).otherwise(1),
    )
    df = df.withColumn("temp_c", F.col("temp") - 273.15)
    return df


def add_congestion_label(df: DataFrame) -> DataFrame:
    """Bucket traffic_volume into low/medium/high (string)."""
    cfg = MODEL_CFG
    return df.withColumn(
        "congestion",
        F.when(F.col("traffic_volume") <= cfg.low_max, "low")
        .when(F.col("traffic_volume") >= cfg.high_min, "high")
        .otherwise("medium"),
    )


def build_pipeline() -> Pipeline:
    """Construct the Spark ML preprocessing pipeline (no estimator)."""
    weather_indexer = StringIndexer(
        inputCol="weather_main", outputCol="weather_idx", handleInvalid="keep"
    )
    season_indexer = StringIndexer(
        inputCol="season", outputCol="season_idx", handleInvalid="keep"
    )
    label_indexer = StringIndexer(
        inputCol="congestion",
        outputCol="label",
        stringOrderType="alphabetAsc",  # high=0, low=1, medium=2  (deterministic)
        handleInvalid="error",
    )

    weather_ohe = OneHotEncoder(
        inputCols=["weather_idx", "season_idx"],
        outputCols=["weather_vec", "season_vec"],
    )

    feature_cols: List[str] = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "temp_c",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_vec",
        "season_vec",
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="keep"
    )

    return Pipeline(
        stages=[weather_indexer, season_indexer, label_indexer, weather_ohe, assembler]
    )


def prepare(df: DataFrame) -> Tuple[DataFrame, PipelineModel]:
    """Apply derived columns + label + fitted preprocessing pipeline.

    Returns the transformed DataFrame and the fitted PipelineModel so the
    same transformations can be reapplied to test data and to single-row
    prediction requests at inference time.
    """
    df = add_derived_columns(df)
    df = add_congestion_label(df)

    pipeline = build_pipeline()
    model = pipeline.fit(df)
    transformed = model.transform(df)
    return transformed, model


def split(df: DataFrame, train_fraction: float = MODEL_CFG.train_fraction):
    return df.randomSplit([train_fraction, 1.0 - train_fraction], seed=MODEL_CFG.seed)
