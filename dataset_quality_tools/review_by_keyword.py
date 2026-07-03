#!/usr/bin/env python3
"""Per-keyword breakdown of the motion-scenery full-exam review.

Joins full_exam.csv with the RAFT/aesthetic CSVs (for the binary `dynamic`
flag and the gate verdict), groups the staged clips by their keyword prefix
(`<keyword>__<clip>.mp4`), and reports for each keyword:

  n            clips scored
  dyn=1        clips VBench judges dynamic (motion>16.5)
  gate_pass    dynamic==1 AND aes>=floor AND motion<=hi  (the Stage-2 gate)
  aes/motion   median of aesthetic_quality and motion_mean_RAFT

Also prints the gate-rejection reason histogram so you can see whether the
gate drops clips for being too static vs too ugly vs chaotic.
"""
from __future__ import annotations
import argparse, csv, statistics as st
from collections import defaultdict
from pathlib import Path


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path, key, cols):
    out = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[Path(r[key]).name] = {c: r.get(c, "") for c in cols}
    return out


def keyword_of(name: str) -> str:
    return name.split("__", 1)[0] if "__" in name else "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default="motion_scenery_review/full_exam.csv")
    ap.add_argument("--motion", default="motion_scenery_review/review_motion.csv")
    ap.add_argument("--aes", default="motion_scenery_review/review_aes.csv")
    ap.add_argument("--aesthetic-floor", type=float, default=0.55)
    ap.add_argument("--motion-hi", type=float, default=80.0)
    args = ap.parse_args()

    exam = list(csv.DictReader(open(args.exam, encoding="utf-8")))
    motion = load(args.motion, "video", ["motion_mean", "dynamic"])
    aes = load(args.aes, "video", ["aesthetic_quality", "dynamic"])

    groups = defaultdict(list)
    for r in exam:
        groups[keyword_of(r["video"])].append(r)

    reasons_hist = defaultdict(int)
    print(f"\n{'='*92}")
    print(f"{'keyword':<46}{'n':>4}{'dyn=1':>7}{'gate':>6}{'aes_md':>8}{'mot_md':>8}")
    print("=" * 92)
    tot_n = tot_dyn = tot_gate = 0
    for kw in sorted(groups):
        rows = groups[kw]
        n = len(rows)
        dyn = gate = 0
        aess, mots = [], []
        for r in rows:
            name = r["video"]
            mm = fnum(motion.get(name, {}).get("motion_mean"))
            dflag = str(motion.get(name, {}).get("dynamic", "")).strip() in ("1", "1.0")
            aq = fnum(aes.get(name, {}).get("aesthetic_quality"))
            if aq is not None:
                aess.append(aq)
            if mm is not None:
                mots.append(mm)
            if dflag:
                dyn += 1
            rs = []
            if not dflag:
                rs.append("static")
            if mm is not None and mm > args.motion_hi:
                rs.append("chaotic")
            if aq is not None and aq < args.aesthetic_floor:
                rs.append("ugly")
            if not rs:
                gate += 1
            else:
                for x in rs:
                    reasons_hist[x] += 1
        tot_n += n; tot_dyn += dyn; tot_gate += gate
        amd = f"{st.median(aess):.3f}" if aess else "-"
        mmd = f"{st.median(mots):.1f}" if mots else "-"
        print(f"{kw[:45]:<46}{n:>4}{dyn:>7}{gate:>6}{amd:>8}{mmd:>8}")
    print("-" * 92)
    print(f"{'TOTAL':<46}{tot_n:>4}{tot_dyn:>7}{tot_gate:>6}")
    print(f"\ngate pass rate: {tot_gate}/{tot_n} = {tot_gate/tot_n:.1%}"
          f"   (floor aes>={args.aesthetic_floor}, motion<={args.motion_hi})")
    print("rejection reasons (a clip can have several):")
    for k, v in sorted(reasons_hist.items(), key=lambda x: -x[1]):
        print(f"  {k:<10}{v}")


if __name__ == "__main__":
    main()
