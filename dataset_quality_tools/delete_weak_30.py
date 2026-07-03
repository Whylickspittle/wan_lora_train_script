import csv
import json
import shutil
from pathlib import Path

ROOT = Path("/workspace/wan_lora_train_script/dataset_quality_tools/motion_scenery_final_merged_300")

NAMES = [
    "clip_19226446_node5.80.mp4",
    "clip_34430016_node3.63.mp4",
    "clip_34570951_node10.74.mp4",
    "clouds__clouds_drifting_fast_over_a_mountain_ridge__cinematic_4K__clip_36680639_node2.64.mp4",
    "clouds__clouds_drifting_fast_over_a_mountain_ridge__cinematic_4K__clip_38369533_node2.40.mp4",
    "clouds__low_clouds_rolling_through_a_forested_valley_at_dawn__cinematic_4K__clip_34626144_node12.27.mp4",
    "geo_wonders__clouds_drifting_over_the_grand_canyon__drone_orbit__cinematic_4K__clip_7624524_node7.42.mp4",
    "geo_wonders__clouds_drifting_over_the_grand_canyon__drone_orbit__cinematic_4K__clip_9320555_node7.04.mp4",
    "geo_wonders__clouds_drifting_over_the_grand_canyon__drone_orbit__cinematic_4K__clip_13778739_node7.95.mp4",
    "geo_wonders__clouds_drifting_over_the_grand_canyon__drone_orbit__cinematic_4K__clip_34305802_node0.16.mp4",
    "other_motion__autumn_leaves_falling_in_a_forest__cinematic_4K__clip_5597104_node3.93.mp4",
    "other_motion__autumn_leaves_falling_in_a_forest__cinematic_4K__clip_5597105_node30.18.mp4",
    "other_motion__autumn_leaves_falling_in_a_forest__cinematic_4K__clip_10025696_node11.59.mp4",
    "other_motion__autumn_leaves_falling_in_a_forest__cinematic_4K__clip_35575372_node18.64.mp4",
    "river__river_flowing_through_a_green_forest_canyon__cinematic_4K__clip_34430018_node2.27.mp4",
    "snow_fog__fog_rolling_over_a_green_valley_at_sunrise__cinematic_4K__clip_36578171_node12.00.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_14536361_node7.99.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_30526780_node9.13.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_31381766_node5.44.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_31381783_node3.41.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_35552213_node7.33.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_35779924_node8.41.mp4",
    "waterfall__powerful_waterfall_in_a_tropical_rainforest__cinematic_4K__clip_28398821_node20.08.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_6473352_node13.45.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_6709779_node13.67.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_20310320_node2.84.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_31209343_node5.02.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_35523150_node2.78.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_37446512_node1.28.mp4",
    "wind__tall_grass_swaying_in_the_wind_on_a_green_hillside__cinematic_4K__clip_16415529_node16.93.mp4",
]

scores = {Path(r["video"]).name: r for r in csv.DictReader(open(ROOT / "final_scores_for_review.csv"))}

TO_DELETE = set()
for n in NAMES:
    r = scores.get(n)
    if not r:
        continue
    mean = float(r["motion_mean"])
    median = float(r["motion_median"])
    if min(mean, median) < 25.0:
        TO_DELETE.add(n)

print(f"Will delete {len(TO_DELETE)} / {len(NAMES)} clips (min(mean,median) < 25)")
for n in sorted(TO_DELETE):
    print(f"  DEL {n}")
kept = [n for n in NAMES if n not in TO_DELETE]
print(f"\nKeep {len(kept)} / {len(NAMES)}:")
for n in kept:
    print(f"  KEEP {n}")

# Delete files
for subdir in ["clips", "clips_motion_ok", "clips_aesthetic_exception"]:
    d = ROOT / subdir
    if not d.exists():
        continue
    for n in TO_DELETE:
        p = d / n
        if p.exists():
            p.unlink()

# Update CSVs
for csv_name in ["final_scores_for_review.csv", "final_scores_motion_ok.csv", "final_scores_aesthetic_exception.csv", "final_scores_dropped.csv"]:
    p = ROOT / csv_name
    if not p.exists():
        continue
    rows = []
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if Path(r.get("video", "")).name not in TO_DELETE:
                rows.append(r)
    shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"{csv_name}: {len(rows)} rows kept")

# Update manifests
for manifest_name in ["manifest_motion_ok.jsonl", "manifest_aesthetic_exception.jsonl"]:
    p = ROOT / manifest_name
    if not p.exists():
        continue
    kept_recs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if Path(rec.get("video", "")).name not in TO_DELETE:
                kept_recs.append(rec)
    shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    with p.open("w", encoding="utf-8") as f:
        for rec in kept_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{manifest_name}: {len(kept_recs)} entries kept")
