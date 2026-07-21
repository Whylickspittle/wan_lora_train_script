#!/usr/bin/env python3
"""R2 bucket setup helper.

Reads Cloudflare R2 credentials from a .env file, checks whether the target
bucket exists, and creates it if necessary.

Usage:
    python setup_r2_bucket.py --bucket test01 --env-file /workspace/nexisgen/.env
    python setup_r2_bucket.py --hotkey 5xxx... --env-file /workspace/nexisgen/.env
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

logger = logging.getLogger("setup_r2_bucket")

_R2_ACCOUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_valid_r2_account_id(value: str) -> bool:
    return bool(_R2_ACCOUNT_ID_RE.fullmatch(value.strip().lower()))


def build_s3_client(env_file: Path | None) -> boto3.client:
    """Build boto3 S3 client for R2 from environment variables."""
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv()

    # Prefer R2_* variables, fall back to AWS_* for compatibility.
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = (
        os.environ.get("R2_WRITE_ACCESS_KEY", "").strip()
        or os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    )
    secret_key = (
        os.environ.get("R2_WRITE_SECRET_KEY", "").strip()
        or os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    region = os.environ.get("R2_REGION", "auto").strip() or "auto"

    if not account_id:
        raise ValueError("R2_ACCOUNT_ID not found in env")
    if not is_valid_r2_account_id(account_id):
        raise ValueError("R2_ACCOUNT_ID must be 32 lowercase hex characters")
    if not access_key:
        raise ValueError("R2_WRITE_ACCESS_KEY / AWS_ACCESS_KEY_ID not found in env")
    if not secret_key:
        raise ValueError("R2_WRITE_SECRET_KEY / AWS_SECRET_ACCESS_KEY not found in env")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    logger.info("connecting to R2 endpoint: %s", endpoint)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def bucket_exists(s3: boto3.client, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchBucket", "NotFound"}:
            return False
        raise


def create_bucket(s3: boto3.client, bucket: str) -> bool:
    try:
        s3.create_bucket(Bucket=bucket)
        logger.info("created bucket: %s", bucket)
        return True
    except ClientError as exc:
        logger.error("failed to create bucket %s: %s", bucket, exc)
        return False


def ensure_bucket(bucket: str, env_file: Path | None = None) -> bool:
    """High-level helper: ensure bucket exists, creating it if necessary."""
    s3 = build_s3_client(env_file)
    if bucket_exists(s3, bucket):
        logger.info("bucket already exists: %s", bucket)
        return True

    logger.info("bucket does not exist, creating: %s", bucket)
    if create_bucket(s3, bucket):
        if bucket_exists(s3, bucket):
            logger.info("bucket verified: %s", bucket)
            return True
        logger.error("bucket creation reported success but bucket is not accessible")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure an R2 bucket exists")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name")
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Miner hotkey (bucket defaults to lowercase hotkey)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file with R2 credentials",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.bucket and not args.hotkey:
        logger.error("provide --bucket or --hotkey")
        return 1

    bucket = args.bucket if args.bucket else args.hotkey.strip().lower()
    return 0 if ensure_bucket(bucket, args.env_file) else 1


if __name__ == "__main__":
    sys.exit(main())
