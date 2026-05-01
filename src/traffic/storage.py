"""Local + S3 storage abstraction.

The pipeline produces a few categories of artifact (raw CSV, profiling
JSON/CSV, trained Spark ML models, figures, alerts, and the final report).
Every stage writes to the local filesystem first (so the offline
reproducibility story stays intact), and then ``maybe_upload_artifact``
optionally mirrors each file to ``s3://{bucket}/{prefix}/...`` when
``S3_CFG.enabled and S3_CFG.upload_artifacts`` is true.

For *reading* the raw dataset, ``resolve_raw_csv`` returns either a
local path or an ``s3a://`` URI that Spark can consume directly through
the Hadoop S3A connector.

The S3 boto3 client is imported lazily so users without ``boto3``
installed can still run the offline pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from traffic.config import RAW_CSV, S3_CFG, S3Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _boto3_client(cfg: S3Config):  # pragma: no cover - exercised when AWS configured
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 mode. Install with `pip install boto3`."
        ) from exc
    return boto3.client("s3", region_name=cfg.region_name)


def _object_exists(client, bucket: str, key: str) -> bool:  # pragma: no cover
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_raw_csv(local_default: Path = RAW_CSV, cfg: S3Config = S3_CFG) -> str:
    """Return the path Spark should read for the raw CSV.

    Priority:
        1. If S3 is configured AND the object exists, return its s3a:// URI.
        2. Otherwise return the local path.

    The returned string is guaranteed to be readable by ``spark.read.csv``.
    """
    if cfg.enabled:  # pragma: no branch
        try:
            client = _boto3_client(cfg)
            key = cfg.key("raw", local_default.name)
            if _object_exists(client, cfg.bucket, key):
                uri = cfg.uri("raw", local_default.name)
                logger.info("Using S3 raw CSV at %s", uri)
                return uri
            logger.info("S3 configured but %s not present; using local CSV.", key)
        except Exception as exc:  # pragma: no cover
            logger.warning("S3 lookup failed (%s); falling back to local CSV.", exc)
    return str(local_default)


def upload_local_to_s3(
    local_path: Path,
    s3_subkey: str,
    cfg: S3Config = S3_CFG,
) -> Optional[str]:
    """Upload a single local file to s3://{bucket}/{prefix}/{s3_subkey}.

    Returns the s3:// URI on success, ``None`` if S3 is disabled or upload
    is suppressed.
    """
    if not cfg.enabled or not cfg.upload_artifacts:
        return None
    try:  # pragma: no cover - exercised when AWS configured
        client = _boto3_client(cfg)
        key = cfg.key(s3_subkey)
        client.upload_file(str(local_path), cfg.bucket, key)
        uri = f"s3://{cfg.bucket}/{key}"
        logger.info("Uploaded %s -> %s", local_path, uri)
        return uri
    except Exception as exc:  # pragma: no cover
        logger.warning("S3 upload failed for %s: %s", local_path, exc)
        return None


def upload_directory_to_s3(
    local_dir: Path,
    s3_subkey: str,
    cfg: S3Config = S3_CFG,
    pattern: str = "*",
) -> List[str]:
    """Recursively upload every file under ``local_dir`` into S3."""
    if not cfg.enabled or not cfg.upload_artifacts:
        return []
    uploaded: List[str] = []
    for path in sorted(local_dir.rglob(pattern)):
        if path.is_file():
            rel = path.relative_to(local_dir)
            uri = upload_local_to_s3(path, f"{s3_subkey}/{rel.as_posix()}", cfg)
            if uri:
                uploaded.append(uri)
    return uploaded


def upload_artifacts(
    pairs: Iterable[Tuple[Path, str]], cfg: S3Config = S3_CFG
) -> List[str]:
    """Upload a list of (local_path, s3_subkey) pairs and return the URIs."""
    uris: List[str] = []
    for local_path, subkey in pairs:
        if local_path is None:
            continue
        uri = upload_local_to_s3(local_path, subkey, cfg)
        if uri:
            uris.append(uri)
    return uris
