import csv
import json
import shutil
from pathlib import Path

ROOT = Path("/workspace/wan_lora_train_script/dataset_quality_tools/motion_scenery_final_merged_300")

DELETE = {
    # subject mismatch
    "geo_wonders__wind_sweeping_across_uyuni_salt_flat_reflections__cinematic_4K__clip_19121716_node6.41.mp4",
    "geo_wonders__wind_sweeping_across_uyuni_salt_flat_reflections__cinematic_4K__clip_32200340_node7.15.mp4",
    "geo_wonders__wind_sweeping_across_uyuni_salt_flat_reflections__cinematic_4K__clip_34476227_node0.34.mp4",
    "other_motion__autumn_leaves_falling_in_a_forest__cinematic_4K__clip_5894807_node9.26.mp4",
    "other_motion__steam_rising_from_a_geyser_hot_spring__cinematic_4K__clip_9953665_node5.80.mp4",
    "waterfall__powerful_waterfall_in_a_tropical_rainforest__cinematic_4K__clip_15457478_node3.85.mp4",
    "waterfall__powerful_waterfall_in_a_tropical_rainforest__cinematic_4K__clip_28848488_node0.76.mp4",
    "waterfall__powerful_waterfall_in_a_tropical_rainforest__cinematic_4K__clip_32106621_node3.34.mp4",
    "waterfall__powerful_waterfall_in_a_tropical_rainforest__cinematic_4K__clip_35792542_node7.12.mp4",
    "wind__palm_trees_swaying_in_strong_wind_before_a_storm__cinematic_4K__clip_19437867_node1.80.mp4",
    "wind__wind_blowing_through_a_golden_wheat_field_at_sunset__cinematic_4K__clip_37709664_node0.24.mp4",
    # duplicates
    "geo_wonders__wind_sweeping_across_uyuni_salt_flat_reflections__cinematic_4K__clip_37899338_node0.13.mp4",
    "geo_wonders__wind_sweeping_across_uyuni_salt_flat_reflections__cinematic_4K__clip_37899376_node15.26.mp4",
    "river__mountain_stream_cascading_over_mossy_rocks__slow_motion__cinematic_4K__clip_11263445_node38.85.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_35519148_node3.64.mp4",
    "snow_fog__snow_falling_gently_in_a_pine_forest__cinematic_4K__clip_35519241_node0.89.mp4",
}


def delete_files():
    for subdir in ["clips", "clips_motion_ok", "clips_aesthetic_exception"]:
        d = ROOT / subdir
        if not d.exists():
            continue
        for name in DELETE:
            p = d / name
            if p.exists():
                p.unlink()
                print(f"deleted: {p}")


def filter_csv(src: Path, keep_fn):
    rows = []
    fieldnames = []
    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if keep_fn(row):
                rows.append(row)
    shutil.copy(src, src.with_suffix(src.suffix + ".bak"))
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def basename_from_row(row: dict) -> str:
    return Path(row.get("video", "")).name


def update_csvs():
    for csv_name in ["final_scores_for_review.csv", "final_scores_motion_ok.csv", "final_scores_aesthetic_exception.csv", "final_scores_dropped.csv"]:
        p = ROOT / csv_name
        if not p.exists():
            continue
        n = filter_csv(p, lambda r: basename_from_row(r) not in DELETE)
        print(f"{csv_name}: {n} rows kept")


def update_manifest(src: Path):
    kept = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            name = Path(rec.get("video", "")).name
            if name not in DELETE:
                kept.append(rec)
    shutil.copy(src, src.with_suffix(src.suffix + ".bak"))
    with src.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(kept)


def update_manifests():
    for name in ["manifest_motion_ok.jsonl", "manifest_aesthetic_exception.jsonl"]:
        p = ROOT / name
        if not p.exists():
            continue
        n = update_manifest(p)
        print(f"{name}: {n} entries kept")


if __name__ == "__main__":
    delete_files()
    update_csvs()
    update_manifests()
    print("done")
