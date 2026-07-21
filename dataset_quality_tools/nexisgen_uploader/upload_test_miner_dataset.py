#!/usr/bin/env python3
"""Test upload of a small Nexisgen-format dataset to R2.

This is a relaxed version of upload_miner_dataset.py that does NOT enforce the
400-sample protocol requirement. It is meant only for testing the upload mechanics
with a handful of clips.

Layout uploaded:
    {interval_id}/dataset.parquet
    {interval_id}/clips/{clip_id}.mp4
    {interval_id}/frames/{clip_id}.jpg
    {interval_id}/manifest.json       # uploaded LAST
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env from the script's directory by default. The --env-file argument can
# override this with a specific path (e.g. /workspace/nexisgen/.env).
load_dotenv()

_R2_ACCOUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("upload_test_miner_dataset")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket_name_for_hotkey(hotkey: str) -> str:
    return hotkey.strip().lower()


def is_valid_r2_account_id(value: str) -> bool:
    return bool(_R2_ACCOUNT_ID_RE.fullmatch(value.strip().lower()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a small test Nexis dataset to R2")
    parser.add_argument("--interval", type=int, required=True, help="Interval ID / top-level prefix")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hotkey", type=str, required=True)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--account-id", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--jurisdiction", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file containing R2 credentials (default: .env in current directory)",
    )
    return parser.parse_args()


def env_or_raise(name: str, override: str | None = None, cli_flag: str | None = None) -> str:
    value = (override or "").strip() or os.environ.get(name, "").strip()
    if not value:
        flag_hint = f" or pass {cli_flag}" if cli_flag else ""
        raise ValueError(f"Missing required value: set {name} env var{flag_hint}")
    return value


def build_s3_client(account_id: str, region: str, access_key: str, secret_key: str, jurisdiction: str | None):
    if not is_valid_r2_account_id(account_id):
        raise ValueError("R2 account_id must be 32 lowercase hex characters")
    suffix = f".{jurisdiction.strip().lower()}" if jurisdiction else ""
    endpoint = f"https://{account_id}{suffix}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def upload_file(s3, bucket: str, key: str, local_path: Path) -> None:
    try:
        s3.upload_file(str(local_path), bucket, key)
    except ClientError as exc:
        raise RuntimeError(f"Failed to upload {local_path} to s3://{bucket}/{key}: {exc}") from exc


def main() -> int:
    args = parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)
        logger.info("loaded env file: %s", args.env_file)

    bucket = args.bucket if args.bucket else bucket_name_for_hotkey(args.hotkey)
    logger.info("target bucket: %s", bucket)
    logger.info("target prefix: %d/", args.interval)
    logger.info("dataset dir: %s", args.dataset_dir.resolve())

    dataset_path = args.dataset_dir / "dataset.parquet"
    manifest_path = args.dataset_dir / "manifest.json"
    clips_dir = args.dataset_dir / "clips"
    frames_dir = args.dataset_dir / "frames"

    for p in (dataset_path, manifest_path, clips_dir, frames_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Load manifest (relaxed: just JSON, no 400-record enforcement)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("manifest interval_id=%s record_count=%s", manifest.get("interval_id"), manifest.get("record_count"))

    # Verify dataset.parquet sha256 matches manifest
    actual_dataset_hash = sha256_file(dataset_path)
    if actual_dataset_hash != manifest.get("dataset_sha256"):
        logger.warning(
            "dataset.parquet sha256 mismatch: manifest=%s actual=%s",
            manifest.get("dataset_sha256"),
            actual_dataset_hash,
        )
        logger.error("re-run the generator or fix manifest.json")
        return 1
    logger.info("dataset.parquet sha256 matches manifest")

    # Load records and verify clip/frame hashes
    records = pq.read_table(dataset_path).to_pylist()
    logger.info("loaded %d records", len(records))

    logger.info("verifying clip/frame hashes...")
    failures = []
    for i, row in enumerate(records):
        clip_path = clips_dir / Path(row["clip_uri"]).name
        frame_path = frames_dir / Path(row["first_frame_uri"]).name
        if not clip_path.is_file():
            failures.append(f"row {i}: missing clip {clip_path}")
            continue
        if not frame_path.is_file():
            failures.append(f"row {i}: missing frame {frame_path}")
            continue
        if sha256_file(clip_path) != row["clip_sha256"]:
            failures.append(f"row {i}: clip_sha256 mismatch {clip_path}")
        if sha256_file(frame_path) != row["first_frame_sha256"]:
            failures.append(f"row {i}: frame_sha256 mismatch {frame_path}")

    if failures:
        for f in failures:
            logger.error(f)
        return 1
    logger.info("all clip/frame hashes verified")

    if args.dry_run:
        logger.info("DRY RUN complete - no files uploaded")
        return 0

    account_id = env_or_raise("R2_ACCOUNT_ID", args.account_id, "--account-id")
    region = (args.region or os.environ.get("R2_REGION", "auto")).strip() or "auto"
    access_key = env_or_raise("R2_WRITE_ACCESS_KEY")
    secret_key = env_or_raise("R2_WRITE_SECRET_KEY")

    s3 = build_s3_client(account_id, region, access_key, secret_key, args.jurisdiction)
    prefix = f"{args.interval}"

    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("bucket exists and credentials work")
    except ClientError as exc:
        logger.warning("head_bucket failed (can be ignored): %s", exc)

    # Upload dataset.parquet
    dataset_key = f"{prefix}/dataset.parquet"
    logger.info("uploading %s ...", dataset_key)
    upload_file(s3, bucket, dataset_key, dataset_path)

    # Upload clips and frames
    tasks = []
    for row in records:
        tasks.append((f"{prefix}/{row['clip_uri']}", clips_dir / Path(row["clip_uri"]).name))
        tasks.append((f"{prefix}/{row['first_frame_uri']}", frames_dir / Path(row["first_frame_uri"]).name))

    logger.info("uploading %d assets with %d workers ...", len(tasks), args.max_workers)
    uploaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_key = {executor.submit(upload_file, s3, bucket, key, path): key for key, path in tasks}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                future.result()
                uploaded += 1
            except Exception as exc:
                logger.error("upload failed for %s: %s", key, exc)
                failed += 1

    if failed:
        logger.error("%d assets failed; aborting before manifest upload", failed)
        return 1

    logger.info("uploaded %d assets successfully", uploaded)

    # Upload manifest LAST
    manifest_key = f"{prefix}/manifest.json"
    logger.info("uploading %s (completion signal) ...", manifest_key)
    upload_file(s3, bucket, manifest_key, manifest_path)

    logger.info("DONE: s3://%s/%s/ is ready", bucket, prefix)
    logger.info("remember to run: nexis commit-credentials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
