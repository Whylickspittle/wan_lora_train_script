# Nexisgen Automated Uploader

Automated Nexisgen dataset generation and R2 upload scheduler.

This module produces small Nexisgen-format interval packages and uploads them to a Cloudflare R2 bucket on a schedule. It is intended to be driven by Linux cron or systemd timer.

---

## Files

| File | Purpose |
|---|---|
| `prepare_test_clips.py` | Generate a few 1280×704 @ 24fps test clips and first frames |
| `generate_test_dataset.py` | Build `dataset.parquet` + `manifest.json` from clips/frames |
| `upload_test_miner_dataset.py` | Upload one interval to R2 (relaxed, no 400-sample enforcement) |
| `run_scheduled_upload.py` | Runner that orchestrates generate → upload → state update |
| `check_scheduled_state.py` | Verify uploaded intervals in R2 and local state |
| `crontab.test` | 5-minute interval test crontab (stops after 2 runs) |
| `crontab.production` | Every-20-hours production crontab |

---

## Quick Start

### 1. Prepare test clips

```bash
cd dataset_quality_tools/nexisgen_uploader
python prepare_test_clips.py --output ./interval_1
```

This creates `interval_1/clips/` and `interval_1/frames/`.

### 2. Configure R2 credentials

Create or reuse a `.env` file with:

```bash
R2_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_REGION=auto
R2_WRITE_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_WRITE_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Test one manual upload

```bash
python run_scheduled_upload.py \
    --hotkey 5EJGfSvRcEGVQtqDuU7YYwuZRHmaktf6JEZDeFPyeXksiHrm \
    --bucket test01 \
    --env-file /path/to/.env \
    --max-runs 1
```

### 4. Run the 5-minute cron test

Edit `crontab.test`, replace `HOTKEY` with your miner hotkey, then:

```bash
sudo service cron start   # if cron is not running
crontab crontab.test
```

The job will run every 5 minutes and stop after 2 successful uploads. Verify with:

```bash
python check_scheduled_state.py --bucket test01 --prefixes 2,3 --env-file /path/to/.env
```

When done, remove the test cron job:

```bash
crontab -r
```

---

## Production Setup

1. Replace the test clip source with your real data pipeline.
2. Edit `crontab.production`, replace `HOTKEY`, and remove `--bucket test01` so the script uses your hotkey-named bucket.
3. Install the crontab:

```bash
crontab crontab.production
```

4. Remember to run `nexis commit-credentials` in the `nexisgen` repo so validators can read your R2 bucket.

---

## Notes

- `upload_test_miner_dataset.py` is a relaxed test uploader. It does **not** enforce the Nexisgen 400-sample requirement.
- `run_scheduled_upload.py` copies clips/frames from `--source-interval-dir` for each new interval. In production, replace this step with real clip generation.
- `--max-runs` is only for testing; omit it in production.
