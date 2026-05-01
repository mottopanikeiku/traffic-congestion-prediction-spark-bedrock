"""Quick smoke tests to verify that the public API is importable and that
the offline integrations (mock Bedrock, mock SageMaker, S3 helpers)
behave correctly. Real Spark logic is exercised end-to-end via
``scripts/run_all.sh``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_imports():
    import traffic  # noqa: F401
    from traffic import (  # noqa: F401
        bedrock_client,
        cli,
        config,
        data_loader,
        models,
        pipeline,
        preprocessing,
        profiling,
        sagemaker_tracker,
        spark_session,
        storage,
        synthetic_data,
        visualizations,
    )


def test_mock_bedrock_alert():
    from traffic.bedrock_client import BedrockClient, PredictionContext
    from traffic.config import BedrockConfig

    client = BedrockClient.from_config(BedrockConfig(mode="mock"))
    ctx = PredictionContext(
        level="high",
        p_low=0.05,
        p_medium=0.15,
        p_high=0.80,
        day_name="Monday",
        hour=8,
        weather_main="Rain",
        weather_description="moderate rain",
        temp_c=12.0,
        holiday="None",
    )
    text = client.generate_alert(ctx)
    assert "Monday" in text
    assert "08:00" in text
    assert "Heavy" in text or "heavy" in text.lower()
    assert "rain" in text.lower()


def test_mock_bedrock_low_level():
    from traffic.bedrock_client import BedrockClient, PredictionContext
    from traffic.config import BedrockConfig

    client = BedrockClient.from_config(BedrockConfig(mode="mock"))
    ctx = PredictionContext(
        level="low",
        p_low=0.85,
        p_medium=0.10,
        p_high=0.05,
        day_name="Sunday",
        hour=4,
        weather_main="Clear",
        weather_description="sky is clear",
        temp_c=18.0,
        holiday="None",
    )
    text = client.generate_alert(ctx)
    assert "Light" in text
    assert "Sunday" in text


def test_holiday_clause_renders():
    from traffic.bedrock_client import BedrockClient, PredictionContext
    from traffic.config import BedrockConfig

    client = BedrockClient.from_config(BedrockConfig(mode="mock"))
    ctx = PredictionContext(
        level="medium",
        p_low=0.20,
        p_medium=0.60,
        p_high=0.20,
        day_name="Thursday",
        hour=10,
        weather_main="Clear",
        weather_description="sky is clear",
        temp_c=22.0,
        holiday="Thanksgiving Day",
    )
    text = client.generate_alert(ctx)
    assert "Thanksgiving Day" in text


def test_synthetic_data_generation(tmp_path):
    from traffic.synthetic_data import generate_dataset

    target = tmp_path / "synthetic.csv"
    generate_dataset(target, seed=7)
    assert target.exists()
    content = target.read_text().splitlines()
    assert len(content) > 100  # header + many rows
    assert content[0].split(",")[0] == "holiday"


def test_synthetic_data_temperature_range(tmp_path):
    """Synthetic temps must stay inside a realistic Minneapolis envelope."""
    import csv as _csv
    from traffic.synthetic_data import generate_dataset

    target = tmp_path / "synthetic.csv"
    generate_dataset(target, seed=11)
    temps = []
    with target.open() as fh:
        for row in _csv.DictReader(fh):
            temps.append(float(row["temp"]))
    # Real UCI dataset temp range, K: ~243 (lowest cold snap) – ~310 (hottest summer hour).
    assert min(temps) > 240, f"min temp {min(temps)} K is below realistic range"
    assert max(temps) < 320, f"max temp {max(temps)} K is above realistic range"


def test_s3_config_disabled_by_default():
    from traffic.config import S3Config

    cfg = S3Config(bucket=None)
    assert not cfg.enabled


def test_s3_config_uri_builder():
    from traffic.config import S3Config

    cfg = S3Config(bucket="my-bucket", prefix="bdt/traffic")
    assert cfg.enabled
    assert cfg.uri("raw", "Metro_Interstate_Traffic_Volume.csv") == (
        "s3a://my-bucket/bdt/traffic/raw/Metro_Interstate_Traffic_Volume.csv"
    )
    assert cfg.key("reports", "results", "profile.json") == (
        "bdt/traffic/reports/results/profile.json"
    )


def test_resolve_raw_csv_falls_back_to_local(tmp_path):
    from traffic.config import S3Config
    from traffic.storage import resolve_raw_csv

    local = tmp_path / "metro.csv"
    local.write_text("dummy")
    cfg = S3Config(bucket=None)
    assert resolve_raw_csv(local, cfg) == str(local)


def test_sagemaker_tracker_off_mode_is_noop(tmp_path):
    from traffic.config import SageMakerConfig
    from traffic.sagemaker_tracker import SageMakerTracker

    tracker = SageMakerTracker(SageMakerConfig(mode="off"))
    tracker.start(spark_master="local[*]", bedrock_mode="mock")
    tracker.record_model("LR", {"maxIter": 50}, {"f1": 0.5})
    out = tracker.finish()
    assert out is None


def test_sagemaker_tracker_mock_mode_writes_payload():
    from traffic.config import RESULTS_DIR, SageMakerConfig
    from traffic.sagemaker_tracker import SageMakerTracker

    tracker = SageMakerTracker(SageMakerConfig(mode="mock", experiment_name="unit-test"))
    tracker.start(spark_master="local[*]", bedrock_mode="mock", s3_bucket=None)
    tracker.record_profile({"row_count": 1000, "column_count": 9,
                            "traffic_volume_quantiles": {"p50": 3000.0}})
    tracker.record_model(
        "RandomForest",
        parameters={"numTrees": 120, "maxDepth": 15},
        metrics={"accuracy": 0.93, "f1": 0.92},
    )
    out = tracker.finish()
    assert out is not None and out.exists()
    payload = json.loads(out.read_text())
    assert payload["experiment_name"] == "unit-test"
    assert payload["mode"] == "mock"
    assert payload["models"][0]["name"] == "RandomForest"
    assert payload["profile_summary"]["row_count"] == 1000


def test_models_module_exposes_cv():
    """The CrossValidator code path must be wired up.

    We spin up a tiny Spark session because Spark ML estimators have to
    be constructed inside an active SparkContext. The session is reused
    across pytest collection.
    """
    from traffic import models
    from traffic.spark_session import get_spark

    get_spark()  # warm up SparkContext
    triples = models._build_estimators_and_grids()
    names = [t[0] for t in triples]
    assert names == ["LogisticRegression", "DecisionTree", "RandomForest"]
    for _, _, grid in triples:
        assert isinstance(grid, list) and len(grid) >= 2
