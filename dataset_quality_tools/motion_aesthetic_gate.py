#!/usr/bin/env python3
"""Final RAFT + aesthetic gate for the motion-scenery acquisition workflow.

run_vbench_dataset.py selects Top-K clips per keyword using CLIP proxies, but
its motion signal is mean_delta (a weak pixel proxy, r~0.18 vs RAFT). This gate
is the SECOND funnel stage: it rescoring the *already-selected* clips with the
exact VBench RAFT optical-flow motion (score_motion_strength.py) and the exact
VBench LAION aesthetic (vbench_exact_scorer.py), then keeps only clips that are
genuinely dynamic-but-not-chaotic and visually clean:

    motion_mean in [--motion-lo, --motion-hi]   (default 33..80)
    aesthetic_quality >= --aesthetic-floor       (default 0.58)

It reads every <dataset-root>/*/selected_manifest.jsonl (preserving each clip's
prompt), stages the selected clips into one dir, scores them, filters, copies
the survivors into <output>/clips/, and writes:
    <output>/final_manifest.jsonl   ({id, video, prompt}) for the survivors
    <output>/gate_scores.csv        every selected clip + scores + keep/reason

Original per-keyword outputs are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


def collect_selected(dataset_root: Path) -> list[dict]:
    """Read all */selected_manifest.jsonl; return [{path, prompt, staged_name}]."""
    items: list[dict] = []
    for man in sorted(dataset_root.glob("*/selected_manifest.jsonl")):
        subdir = man.parent
        for line in man.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = (subdir / rec["video"]).resolve()
            if not vid.exists():
                print(f"  missing clip (skip): {vid}", file=sys.stderr)
                continue
            staged = f"{_safe(subdir.name)}__{vid.name}"
            items.append({"path": vid, "prompt": rec.get("prompt", ""),
                          "staged_name": staged, "id": Path(staged).stem})
    return items


def stage(items: list[dict], staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for it in items:
        dst = staging / it["staged_name"]
        try:
            dst.symlink_to(it["path"])
        except OSError:
            shutil.copy2(it["path"], dst)


def run_scorer(script: str, staging: Path, out_csv: Path) -> int:
    cmd = [sys.executable, str(HERE / script), "-i", str(staging), "-o", str(out_csv), "--resume"]
    print("  $", " ".join(cmd))
    return subprocess.run(cmd).returncode


def load_csv(path: Path, key: str, cols: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[Path(r[key]).name] = {c: r.get(c, "") for c in cols}
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Final RAFT + aesthetic gate")
    ap.add_argument("--dataset-root", type=Path, default=Path("./vbench_dataset"),
                    help="root holding <keyword>/selected_manifest.jsonl dirs")
    ap.add_argument("--output", type=Path, default=Path("./motion_scenery_final"))
    ap.add_argument("--motion-lo", type=float, default=33.0)
    ap.add_argument("--motion-hi", type=float, default=80.0)
    ap.add_argument("--aesthetic-floor", type=float, default=0.58)
    ap.add_argument("--require-dynamic", action="store_true",
                    help="use VBench's binary dynamic flag (motion>thres) as the motion "
                         "criterion instead of the motion_lo floor; motion_hi still guards "
                         "against chaos. Matches how VBench scores dynamic_degree.")
    ap.add_argument("--prompt-prefix", default="a cinematic 4k video of",
                    help="used only when a selected_manifest entry has no prompt")
    args = ap.parse_args()

    items = collect_selected(args.dataset_root)
    if not items:
        print(f"No selected clips under {args.dataset_root}/*/selected_manifest.jsonl", file=sys.stderr)
        return 1
    print(f"Collected {len(items)} selected clips from {args.dataset_root}")

    args.output.mkdir(parents=True, exist_ok=True)
    staging = args.output / "_staging"
    stage(items, staging)

    motion_csv = args.output / "gate_motion.csv"
    aes_csv = args.output / "gate_aesthetic.csv"
    if run_scorer("score_motion_strength.py", staging, motion_csv) != 0:
        print("RAFT motion scoring failed", file=sys.stderr); return 1
    if run_scorer("vbench_exact_scorer.py", staging, aes_csv) != 0:
        print("aesthetic scoring failed", file=sys.stderr); return 1

    motion = load_csv(motion_csv, "video", ["motion_mean", "dynamic"])
    aes = load_csv(aes_csv, "video", ["aesthetic_quality"])

    clips_out = args.output / "clips"
    clips_out.mkdir(exist_ok=True)
    scores_csv = args.output / "gate_scores.csv"
    manifest = args.output / "final_manifest.jsonl"

    kept = 0
    sf = scores_csv.open("w", newline="", encoding="utf-8")
    sw = csv.DictWriter(sf, fieldnames=["staged_name", "motion_mean", "aesthetic_quality",
                                        "keep", "reason"])
    sw.writeheader()
    mf = manifest.open("w", encoding="utf-8")

    for it in items:
        name = it["staged_name"]
        mrec = motion.get(name, {})
        mm = fnum(mrec.get("motion_mean"))
        dyn = str(mrec.get("dynamic", "")).strip()
        aq = fnum(aes.get(name, {}).get("aesthetic_quality"))
        reasons = []
        if mm is None:
            reasons.append("no-motion-score")
        else:
            if args.require_dynamic:
                if dyn not in ("1", "1.0"):
                    reasons.append(f"not-dynamic({mm:.0f})")
            elif mm < args.motion_lo:
                reasons.append(f"motion<{args.motion_lo:.0f}({mm:.0f})")
            if mm > args.motion_hi:
                reasons.append(f"motion>{args.motion_hi:.0f}({mm:.0f})")
        if aq is None:
            reasons.append("no-aes-score")
        elif aq < args.aesthetic_floor:
            reasons.append(f"aes<{args.aesthetic_floor:.2f}({aq:.3f})")
        keep = not reasons
        sw.writerow({"staged_name": name,
                     "motion_mean": f"{mm:.3f}" if mm is not None else "",
                     "aesthetic_quality": f"{aq:.4f}" if aq is not None else "",
                     "keep": int(keep), "reason": "" if keep else "; ".join(reasons)})
        if keep:
            dst = clips_out / name
            shutil.copy2(it["path"], dst)
            prompt = it["prompt"] or f"{args.prompt_prefix} {Path(name).stem}"
            mf.write(json.dumps({"id": it["id"], "video": f"clips/{name}",
                                 "prompt": prompt}, ensure_ascii=False) + "\n")
            kept += 1
    sf.close(); mf.close()
    shutil.rmtree(staging, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"selected in : {len(items)}")
    print(f"kept (gate) : {kept}  ({kept/len(items):.1%})")
    print(f"clips  -> {clips_out}")
    print(f"manifest -> {manifest}")
    print(f"scores -> {scores_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
