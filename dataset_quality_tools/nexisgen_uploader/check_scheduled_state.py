#!/usr/bin/env python3
"""Check scheduled upload results in R2 and local state."""

import argparse
from pathlib import Path
import boto3
from dotenv import load_dotenv
import os


def main():
    parser = argparse.ArgumentParser(description="Check scheduled upload results in R2")
    parser.add_argument("--bucket", type=str, default="test01", help="R2 bucket name")
    parser.add_argument("--prefixes", type=str, default="3,4", help="Comma-separated interval prefixes to check")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to .env file with R2 credentials")
    parser.add_argument("--state-file", type=Path, default=None, help="Path to .schedule_state.json")
    args = parser.parse_args()

    load_dotenv() if args.env_file is None else load_dotenv(args.env_file, override=True)

    account_id = os.environ['R2_ACCOUNT_ID']
    endpoint = f'https://{account_id}.r2.cloudflarestorage.com'
    s3 = boto3.client('s3', endpoint_url=endpoint, region_name='auto',
                      aws_access_key_id=os.environ['R2_WRITE_ACCESS_KEY'],
                      aws_secret_access_key=os.environ['R2_WRITE_SECRET_KEY'])

    for prefix in args.prefixes.split(','):
        prefix = prefix.strip()
        print(f"\n--- s3://{args.bucket}/{prefix}/ ---")
        paginator = s3.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=args.bucket, Prefix=f'{prefix}/'):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
                print(f"  {obj['Key']} ({obj['Size']} bytes)")
        print(f"Total objects in {prefix}/: {len(keys)}")

    print("\n--- State file ---")
    state_path = args.state_file or Path(__file__).parent / ".schedule_state.json"
    if state_path.exists():
        print(state_path.read_text())
    else:
        print(f"state file not found: {state_path}")


if __name__ == "__main__":
    main()
