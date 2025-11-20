import os
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)
_DVC_WARNING_EMITTED = False


def _dvc_base_cmd() -> list[str]:
    global _DVC_WARNING_EMITTED
    dvc_path = shutil.which("dvc")
    if dvc_path:
        return ["dvc"]
    if not _DVC_WARNING_EMITTED:
        logger.warning("`dvc` CLI not found on PATH; using python -m dvc fallback.")
        _DVC_WARNING_EMITTED = True
    return [sys.executable, "-m", "dvc"]


def _run_dvc_cmd(args: list[str]) -> subprocess.CompletedProcess:
    cmd = _dvc_base_cmd() + args
    return subprocess.run(cmd, check=True, capture_output=True)


def _read_dvc_remote_url(remote_name: str) -> Optional[str]:
    config_path = Path(".dvc") / "config"
    if not config_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(str(config_path))
    target = f'remote "{remote_name}"'
    for section in parser.sections():
        normalized = section.strip("'")
        if normalized == target:
            if parser.has_option(section, "url"):
                return parser.get(section, "url")
    return None


def setup_dvc_remote() -> bool:
    """Ensure DVC S3 remote is configured; idempotent and safe to rerun."""
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        logger.info("Initializing local DVC metadata directory.")
        _run_dvc_cmd(["init", "--no-scm"])

    remote_name = os.getenv("DVC_REMOTE_NAME", "myremote")
    bucket_url = os.getenv("DVC_BUCKET_URL") or _read_dvc_remote_url(remote_name)
    default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    default_endpoint = f"https://s3.{default_region}.amazonaws.com" if default_region != "us-east-1" else "https://s3.amazonaws.com"
    endpoint = os.getenv("DVC_ENDPOINT", default_endpoint)
    if not bucket_url:
        logger.warning("DVC_BUCKET_URL not set; skipping remote configuration.")
        return False

    try:
        remotes_output = _run_dvc_cmd(["remote", "list"]).stdout.decode()
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        logger.error("Unable to list DVC remotes: %s", exc.stderr)
        raise

    if remote_name not in remotes_output:
        _run_dvc_cmd(["remote", "add", "-d", remote_name, bucket_url])
        logger.info("Added DVC remote %s -> %s", remote_name, bucket_url)
    else:
        current_url = _read_dvc_remote_url(remote_name)
        if current_url != bucket_url:
            _run_dvc_cmd(["remote", "modify", remote_name, "url", bucket_url])
            logger.info("Updated DVC remote %s url to %s", remote_name, bucket_url)
        else:
            logger.info("DVC remote %s already configured.", remote_name)

    _run_dvc_cmd(["remote", "modify", remote_name, "endpointurl", endpoint])
    logger.info("DVC remote %s endpoint updated to %s", remote_name, endpoint)
    return True


def test_s3_connection(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
) -> bool:
    """Connectivity check to the specified bucket; returns True if HeadBucket passes."""
    bucket_url = os.getenv("DVC_BUCKET_URL")
    if not bucket_url:
        cfg_path = Path(".dvc") / "config"
        if cfg_path.exists():
            parser = configparser.ConfigParser()
            parser.read(str(cfg_path))
            section = f'remote "{os.getenv("DVC_REMOTE_NAME", "myremote")}"'
            if parser.has_option(section, "url"):
                bucket_url = parser.get(section, "url")
    bucket_name = urlparse(bucket_url).netloc if bucket_url else None
    aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = aws_secret_access_key or os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )
    region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

    if not bucket_name:
        logger.warning("No DVC_BUCKET_URL set; skipping S3 connectivity test.")
        return False

    endpoint_override = os.getenv("AWS_S3_ENDPOINT") or os.getenv("DVC_ENDPOINT")

    def _try_connect(endpoint_url: Optional[str]) -> bool:
        try:
            client_kwargs = {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "region_name": region_name,
            }
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            s3 = boto3.client("s3", **client_kwargs)
            s3.head_bucket(Bucket=bucket_name)
            logger.info(
                "S3 connection successful for bucket %s (endpoint=%s).",
                bucket_name,
                endpoint_url or "default",
            )
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please set environment variables.")
        except EndpointConnectionError as exc:
            logger.error(
                "S3 endpoint %s unreachable: %s",
                endpoint_url or "default",
                exc,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            logger.error(
                "S3 connection failed for bucket %s (code=%s). Verify permissions and bucket name.",
                bucket_name,
                code,
            )
        except BotoCoreError as exc:
            logger.error("S3 connection failed for bucket %s: %s", bucket_name, exc)
        return False

    if _try_connect(endpoint_override):
        return True
    if endpoint_override:
        logger.info("Retrying S3 connection with default endpoint.")
        return _try_connect(None)
    return False
