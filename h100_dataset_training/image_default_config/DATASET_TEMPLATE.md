# Dataset Template

Use this structure for each dataset:

```text
J:/datasets/<dataset_id>/
  manifest.jsonl
  videos/
    clip_000001.mp4
    clip_000002.mp4
```

`manifest.jsonl` example:

```json
{"id":"clip_000001","video":"J:/datasets/<dataset_id>/videos/clip_000001.mp4","prompt":"A person walks through a bright modern room while the camera slowly pans left."}
{"id":"clip_000002","video":"J:/datasets/<dataset_id>/videos/clip_000002.mp4","prompt":"A close-up of a product rotating on a table with soft studio lighting."}
```

Good captions should mention:

- main subject,
- motion,
- camera movement,
- setting,
- lighting,
- style or quality details.

Avoid captions that are too short, generic, or unrelated to the video.

## Memory-Safe Layout

Do not place all video data inside one giant archive for training. Keep each dataset as a folder of individual video files plus one manifest:

```text
J:/datasets/dataset_a/
  manifest.jsonl
  videos/
    clip_000001.mp4
    clip_000002.mp4
```

The training script reads one batch at a time from these files. This prevents the whole dataset from being loaded into system RAM or GPU RAM.
