#!/usr/bin/env python3
"""Summarize a full_exam CSV: per-column distribution + auto-flag bad clips."""
import csv, sys, statistics as st
from pathlib import Path

NUM = ["motion_mean_RAFT","motion_median_RAFT","motion_max_RAFT","moving_ratio",
       "aesthetic_quality","mean_delta","p95_delta","subject_consistency",
       "background_consistency","temporal_flickering","sharpness",
       "flicker_luma_std","black_ratio","white_ratio"]

def fnum(x):
    try: return float(x)
    except: return None

def pct(v,xs):
    xs=sorted(xs);
    import bisect; return bisect.bisect_left(xs,v)/len(xs)

def main(path):
    rows=list(csv.DictReader(open(path,encoding="utf-8")))
    n=len(rows)
    print(f"\n{'='*70}\n{Path(path).name}   clips={n}\n{'='*70}")
    print(f"{'metric':<24}{'min':>9}{'median':>9}{'mean':>9}{'max':>9}")
    cols={}
    for c in NUM:
        vals=[fnum(r[c]) for r in rows if fnum(r[c]) is not None]
        cols[c]=vals
        if vals:
            print(f"{c:<24}{min(vals):>9.3f}{st.median(vals):>9.3f}{sum(vals)/len(vals):>9.3f}{max(vals):>9.3f}")
    # auto-flags
    print(f"\n--- AUTO-FLAGS ---")
    flagged=[]
    for r in rows:
        g=lambda c: fnum(r[c])
        reasons=[]
        if g("aesthetic_quality") is not None and g("aesthetic_quality")<0.55: reasons.append(f"aes<0.55({g('aesthetic_quality'):.3f})")
        if g("motion_mean_RAFT") is not None and g("motion_mean_RAFT")>80: reasons.append(f"over-motion({g('motion_mean_RAFT'):.0f})")
        if g("sharpness") is not None and g("sharpness")<0.012: reasons.append(f"soft/blurry({g('sharpness'):.4f})")
        if g("flicker_luma_std") is not None and g("flicker_luma_std")>0.10: reasons.append(f"flicker({g('flicker_luma_std'):.3f})")
        if g("black_ratio") is not None and g("black_ratio")>0.05: reasons.append(f"black({g('black_ratio'):.3f})")
        if g("white_ratio") is not None and g("white_ratio")>0.05: reasons.append(f"white({g('white_ratio'):.3f})")
        if g("subject_consistency") is not None and g("subject_consistency")<0.40: reasons.append(f"chaotic-subj({g('subject_consistency'):.3f})")
        if reasons:
            flagged.append((r["video"],reasons))
    print(f"{len(flagged)}/{n} clips flagged:")
    for v,rs in sorted(flagged):
        print(f"  {v:<48} {', '.join(rs)}")
    clean=[r["video"] for r in rows if r["video"] not in {v for v,_ in flagged}]
    print(f"\nCLEAN (no flags): {len(clean)}/{n}")

if __name__=="__main__":
    for p in sys.argv[1:]:
        main(p)
