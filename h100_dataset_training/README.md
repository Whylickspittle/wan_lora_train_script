# H100 Dataset Training

Edit one file:

```text
h100_dataset_training/config.json
```

Then run these from the `production_release` folder:

```powershell
python h100_dataset_training/00_download_model.py
python h100_dataset_training/00_prepare_huggingface_dataset.py
python h100_dataset_training/01_preflight_dataset.py
python h100_dataset_training/02_train_dataset.py
python h100_dataset_training/03_infer_latest_checkpoint.py
python h100_dataset_training/04_make_quality_report.py
```

Skip `00_prepare_huggingface_dataset.py` if the dataset is already local and `config.json` points to its `manifest.jsonl`.

Final report:

```text
runs/<active_dataset>/report/report.html
```

The checkpoint path is handled automatically through:

```text
runs/<active_dataset>/latest_checkpoint.txt
```

## Large Dataset Memory Rules

This workflow keeps the dataset on disk. It reads `manifest.jsonl`, opens one batch of videos, trains on that batch, then moves on. It does not pre-load all videos into GPU memory or system RAM.

For safest H100 runs, keep these defaults in `config.json`:

```json
"train_batch_size": 1,
"num_workers": 0,
"pin_memory": false
```

Diagnostics will show decoded-batch RAM estimates and total scanned video storage in:

```text
runs/<active_dataset>/diagnostics/diagnostics_report.html
```

If the dataset comes from Hugging Face, leave `huggingface_dataset.streaming` set to `true`. The preparation script streams samples out to local video files and writes a local manifest for training.
