#!/usr/bin/env python3
"""Keep one clip per source video in vbench_review80 (earliest node).

Clips are named clip_<source_id>_node<start>.mp4. For each source_id we keep the
clip with the smallest start (the opening segment) and MOVE the rest to
<category>/dup_removed/ (reversible). Root review_manifest.jsonl /
review_scores.csv / audit_aesthetic_motion.csv are rewritten to the kept set,
with originals backed up to *.predupe.
"""

from __future__ import annotations

import collections
import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path("vbench_review80")
PAT = re.compile(r"clip_(\d+)_node([0-9.]+)")


def kept_basenames() -> set[str]:
    keep: set[str] = set()
    moved = 0
    for cat in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "logs"):
        clips_dir = cat / "clips"
        groups: dict[str, list[tuple[float, Path]]] = collections.defaultdict(list)
        for c in clips_dir.glob("*.mp4"):
            m = PAT.search(c.stem)
            if not m:
                keep.add(c.name)  # unparseable -> keep, don't lose it
                continue
            groups[m.group(1)].append((float(m.group(2)), c))
        dup_dir = cat / "dup_removed"
        for src_id, items in groups.items():
            items.sort(key=lambda t: t[0])  # earliest node first
            keep.add(items[0][1].name)
            for _, path in items[1:]:
                dup_dir.mkdir(exist_ok=True)
                shutil.move(str(path), str(dup_dir / path.name))
                moved += 1
    print(f"moved {moved} duplicate clips to <category>/dup_removed/")
    return keep


def filter_jsonl(path: Path, keep: set[str]) -> int:
    if not path.exists():
        return 0
    backup = path.with_suffix(path.suffix + ".predupe")
    if not backup.exists():
        shutil.copy2(path, backup)
    rows = [json.loads(l) for l in backup.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if Path(r["video"]).name in keep]
    with path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    return len(kept)


def filter_csv(path: Path, keep: set[str]) -> int:
    if not path.exists():
        return 0
    backup = path.with_suffix(path.suffix + ".predupe")
    if not backup.exists():
        shutil.copy2(path, backup)
    with backup.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = [r for r in reader if r.get("clip") in keep]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    keep = kept_basenames()
    print(f"kept {len(keep)} clips (one per source video)")
    m = filter_jsonl(ROOT / "review_manifest.jsonl", keep)
    s = filter_csv(ROOT / "review_scores.csv", keep)
    a = filter_csv(ROOT / "audit_aesthetic_motion.csv", keep)
    print(f"review_manifest.jsonl: {m} rows")
    print(f"review_scores.csv: {s} rows")
    print(f"audit_aesthetic_motion.csv: {a} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
