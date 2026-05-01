"""Command-line interface for the traffic-congestion pipeline.

Examples:

    # End-to-end (mock Bedrock, no AWS, local CSV)
    PYTHONPATH=src python -m traffic.cli all

    # Train + 3-fold cross-validation on each model
    PYTHONPATH=src python -m traffic.cli train --cv

    # Train using a CSV stored in S3, mirror artifacts back to S3,
    # and register a SageMaker Experiment trial for the run
    PYTHONPATH=src python -m traffic.cli train \\
        --s3-bucket my-bucket --s3-prefix bdt/traffic --s3-upload \\
        --sagemaker-mode real

    # One-shot Bedrock alert from CLI inputs (real backend)
    PYTHONPATH=src python -m traffic.cli alert \\
        --level high --hour 8 --day Monday --weather Rain --temp 12 \\
        --bedrock-mode real

    # Profiling only (no model training)
    PYTHONPATH=src python -m traffic.cli profile

    # Push existing local artifacts to S3 without retraining
    PYTHONPATH=src python -m traffic.cli upload --s3-bucket my-bucket
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from traffic import data_loader, pipeline
from traffic.bedrock_client import BedrockClient, PredictionContext
from traffic.config import (
    BEDROCK_CFG,
    RAW_CSV,
    S3_CFG,
    SAGEMAKER_CFG,
    BedrockConfig,
    S3Config,
    SageMakerConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("traffic.cli")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _build_bedrock_cfg(args) -> BedrockConfig:
    cfg = BedrockConfig()
    if getattr(args, "bedrock_mode", None):
        cfg.mode = args.bedrock_mode
    if getattr(args, "bedrock_model", None):
        cfg.model_id = args.bedrock_model
    if getattr(args, "aws_region", None):
        cfg.region_name = args.aws_region
    return cfg


def _build_s3_cfg(args) -> S3Config:
    cfg = S3Config()
    if getattr(args, "s3_bucket", None) is not None:
        cfg.bucket = args.s3_bucket or None
    if getattr(args, "s3_prefix", None):
        cfg.prefix = args.s3_prefix
    if getattr(args, "s3_upload", False):
        cfg.upload_artifacts = True
    if getattr(args, "aws_region", None):
        cfg.region_name = args.aws_region
    # Propagate to module singleton so SparkSession + storage layer see it.
    S3_CFG.bucket = cfg.bucket
    S3_CFG.prefix = cfg.prefix
    S3_CFG.upload_artifacts = cfg.upload_artifacts
    S3_CFG.region_name = cfg.region_name
    return cfg


def _build_sagemaker_cfg(args) -> SageMakerConfig:
    cfg = SageMakerConfig()
    if getattr(args, "sagemaker_mode", None):
        cfg.mode = args.sagemaker_mode
    if getattr(args, "sagemaker_experiment", None):
        cfg.experiment_name = args.sagemaker_experiment
    if getattr(args, "aws_region", None):
        cfg.region_name = args.aws_region
    SAGEMAKER_CFG.mode = cfg.mode
    SAGEMAKER_CFG.experiment_name = cfg.experiment_name
    SAGEMAKER_CFG.region_name = cfg.region_name
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def cmd_download(args) -> int:
    target = Path(args.csv) if args.csv else RAW_CSV
    data_loader.download_dataset(target)
    print(f"Dataset ready at {target}")
    return 0


def cmd_profile(args) -> int:
    summary = pipeline.run_profiling_only(Path(args.csv) if args.csv else RAW_CSV)
    print(f"Profiling complete: {summary['row_count']:,} rows")
    return 0


def cmd_train(args) -> int:
    s3_cfg = _build_s3_cfg(args)
    sm_cfg = _build_sagemaker_cfg(args)
    summary = pipeline.run_pipeline(
        Path(args.csv) if args.csv else RAW_CSV,
        bedrock_cfg=_build_bedrock_cfg(args),
        s3_cfg=s3_cfg,
        sagemaker_cfg=sm_cfg,
        skip_visualize=args.skip_visualize,
        use_cv=args.cv if args.cv else None,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_alert(args) -> int:
    cfg = _build_bedrock_cfg(args)
    client = BedrockClient.from_config(cfg)
    ctx = PredictionContext(
        level=args.level,
        p_low=args.p_low,
        p_medium=args.p_medium,
        p_high=args.p_high,
        day_name=args.day,
        hour=args.hour,
        weather_main=args.weather,
        weather_description=args.weather_desc,
        temp_c=args.temp,
        holiday=args.holiday,
    )
    print(client.generate_alert(ctx))
    return 0


def cmd_upload(args) -> int:
    s3_cfg = _build_s3_cfg(args)
    s3_cfg.upload_artifacts = True
    if not s3_cfg.enabled:
        print("--s3-bucket (or TRAFFIC_S3_BUCKET) is required for upload.", file=sys.stderr)
        return 2
    uris = pipeline.upload_artifacts_to_s3(s3_cfg)
    print(f"Uploaded {len(uris)} files to s3://{s3_cfg.bucket}/{s3_cfg.prefix}/")
    for uri in uris[:20]:
        print(f"  {uri}")
    if len(uris) > 20:
        print(f"  ... and {len(uris) - 20} more")
    return 0


def cmd_all(args) -> int:
    return cmd_train(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def _add_common_csv(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=None, help="Optional path to the raw CSV")


def _add_common_aws(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--aws-region", default=None,
        help="AWS region for Bedrock / S3 / SageMaker (env: AWS_REGION)",
    )


def _add_common_s3(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--s3-bucket", default=None,
        help="S3 bucket for raw data + artifacts (env: TRAFFIC_S3_BUCKET)",
    )
    p.add_argument(
        "--s3-prefix", default=None,
        help="S3 key prefix (default: traffic-congestion-prediction)",
    )
    p.add_argument(
        "--s3-upload", action="store_true",
        help="Mirror local artifacts back to S3 after the run",
    )


def _add_common_bedrock(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bedrock-mode", choices=("mock", "real"), default=None,
        help="Override Bedrock backend (env: TRAFFIC_BEDROCK_MODE)",
    )
    p.add_argument(
        "--bedrock-model", default=None,
        help="Bedrock model id (default: anthropic.claude-3-haiku-20240307-v1:0)",
    )


def _add_common_sagemaker(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sagemaker-mode", choices=("off", "mock", "real"), default=None,
        help="SageMaker tracking mode (env: TRAFFIC_SAGEMAKER_MODE)",
    )
    p.add_argument(
        "--sagemaker-experiment", default=None,
        help="SageMaker experiment name (default: traffic-congestion-prediction)",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="traffic", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download (or generate) the raw dataset")
    _add_common_csv(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_pr = sub.add_parser("profile", help="Run data profiling only")
    _add_common_csv(p_pr)
    _add_common_aws(p_pr)
    _add_common_s3(p_pr)
    p_pr.set_defaults(func=cmd_profile)

    p_tr = sub.add_parser(
        "train",
        help="Run profiling + preprocessing + train all 3 models + alerts + figures",
    )
    _add_common_csv(p_tr)
    _add_common_aws(p_tr)
    _add_common_bedrock(p_tr)
    _add_common_s3(p_tr)
    _add_common_sagemaker(p_tr)
    p_tr.add_argument("--skip-visualize", action="store_true", help="Skip chart generation")
    p_tr.add_argument("--cv", action="store_true", help="Train each model with 3-fold CV")
    p_tr.set_defaults(func=cmd_train)

    p_al = sub.add_parser("alert", help="Generate a single Bedrock alert from CLI inputs")
    _add_common_aws(p_al)
    _add_common_bedrock(p_al)
    p_al.add_argument("--level", choices=("low", "medium", "high"), required=True)
    p_al.add_argument("--p-low", type=float, default=0.1)
    p_al.add_argument("--p-medium", type=float, default=0.1)
    p_al.add_argument("--p-high", type=float, default=0.8)
    p_al.add_argument("--day", default="Monday")
    p_al.add_argument("--hour", type=int, default=8)
    p_al.add_argument("--weather", default="Clear")
    p_al.add_argument("--weather-desc", default="sky is clear")
    p_al.add_argument("--temp", type=float, default=15.0)
    p_al.add_argument("--holiday", default="None")
    p_al.set_defaults(func=cmd_alert)

    p_up = sub.add_parser("upload", help="Push local reports/ + models/ to S3")
    _add_common_aws(p_up)
    _add_common_s3(p_up)
    p_up.set_defaults(func=cmd_upload)

    p_all = sub.add_parser(
        "all",
        help="End-to-end: download → profile → train → alerts → charts (alias for train)",
    )
    _add_common_csv(p_all)
    _add_common_aws(p_all)
    _add_common_bedrock(p_all)
    _add_common_s3(p_all)
    _add_common_sagemaker(p_all)
    p_all.add_argument("--skip-visualize", action="store_true")
    p_all.add_argument("--cv", action="store_true")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
