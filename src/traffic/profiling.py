"""Data profiling routines built on the Spark DataFrame API.

These run distributed across the cluster (`describe`, `groupBy`,
`approxQuantile`) so the profiling stage scales with the data instead of
collecting everything to the driver.

Outputs:
* Summary statistics for every numeric column (CSV + JSON).
* Null-count audit per column.
* Categorical distributions (weather_main, holiday).
* Hourly / day-of-week / month aggregates of traffic volume.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from traffic.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def numeric_summary(df: DataFrame) -> DataFrame:
    """Return Spark's `describe()` for the numeric columns we care about."""
    return df.describe(["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"])


def null_counts(df: DataFrame) -> Dict[str, int]:
    """Return a dict of column -> null count."""
    exprs = [
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]
    row = df.agg(*exprs).first().asDict()
    return {k: int(v) for k, v in row.items()}


def categorical_distribution(df: DataFrame, column: str, top_n: int = 20) -> DataFrame:
    """Top-N category counts ordered by frequency (descending)."""
    return (
        df.groupBy(column)
        .agg(F.count("*").alias("count"))
        .orderBy(F.col("count").desc())
        .limit(top_n)
    )


def hourly_volume(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("hour", F.hour("date_time"))
        .groupBy("hour")
        .agg(
            F.avg("traffic_volume").alias("avg_volume"),
            F.expr("percentile_approx(traffic_volume, 0.5)").alias("median_volume"),
            F.count("*").alias("n_records"),
        )
        .orderBy("hour")
    )


def day_of_week_volume(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("dow", F.dayofweek("date_time"))  # 1=Sun..7=Sat
        .groupBy("dow")
        .agg(
            F.avg("traffic_volume").alias("avg_volume"),
            F.count("*").alias("n_records"),
        )
        .orderBy("dow")
    )


def monthly_volume(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("month", F.month("date_time"))
        .groupBy("month")
        .agg(F.avg("traffic_volume").alias("avg_volume"))
        .orderBy("month")
    )


def weather_volume(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("weather_main")
        .agg(
            F.avg("traffic_volume").alias("avg_volume"),
            F.count("*").alias("n_records"),
        )
        .orderBy(F.col("avg_volume").desc())
    )


def quantiles(df: DataFrame, column: str = "traffic_volume") -> Dict[str, float]:
    qs = df.approxQuantile(column, [0.05, 0.25, 0.5, 0.75, 0.95], 0.001)
    return {"p05": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4]}


def run_profiling(df: DataFrame, out_dir: Path = RESULTS_DIR) -> Dict:
    """Run the full profiling suite and persist results to `out_dir`.

    Returns a dict suitable for direct dumping into the report appendix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.cache()

    profile: Dict = {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "schema": [{"name": f.name, "type": f.dataType.simpleString()} for f in df.schema.fields],
    }

    # Numeric summary
    numeric = numeric_summary(df)
    numeric_pd = numeric.toPandas()
    numeric_pd.to_csv(out_dir / "profile_numeric_summary.csv", index=False)
    profile["numeric_summary"] = numeric_pd.to_dict(orient="records")

    # Nulls
    profile["null_counts"] = null_counts(df)

    # Quantiles
    profile["traffic_volume_quantiles"] = quantiles(df, "traffic_volume")
    profile["temp_quantiles"] = quantiles(df, "temp")

    # Categorical distributions
    weather_pd = categorical_distribution(df, "weather_main").toPandas()
    weather_pd.to_csv(out_dir / "profile_weather_main.csv", index=False)
    profile["weather_main_distribution"] = weather_pd.to_dict(orient="records")

    holiday_pd = categorical_distribution(df, "holiday").toPandas()
    holiday_pd.to_csv(out_dir / "profile_holiday.csv", index=False)
    profile["holiday_distribution"] = holiday_pd.to_dict(orient="records")

    # Time-based aggregates
    hourly_pd = hourly_volume(df).toPandas()
    hourly_pd.to_csv(out_dir / "profile_hourly_volume.csv", index=False)
    profile["hourly_volume"] = hourly_pd.to_dict(orient="records")

    dow_pd = day_of_week_volume(df).toPandas()
    dow_pd.to_csv(out_dir / "profile_dow_volume.csv", index=False)
    profile["dow_volume"] = dow_pd.to_dict(orient="records")

    monthly_pd = monthly_volume(df).toPandas()
    monthly_pd.to_csv(out_dir / "profile_monthly_volume.csv", index=False)
    profile["monthly_volume"] = monthly_pd.to_dict(orient="records")

    weather_avg_pd = weather_volume(df).toPandas()
    weather_avg_pd.to_csv(out_dir / "profile_weather_volume.csv", index=False)
    profile["weather_volume"] = weather_avg_pd.to_dict(orient="records")

    with (out_dir / "profile.json").open("w") as fh:
        json.dump(profile, fh, indent=2, default=str)

    df.unpersist()
    logger.info("Profiling artifacts written to %s", out_dir)
    return profile
