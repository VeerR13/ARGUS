"""
download_coco_vehicles.py — Download COCO val2017 images containing
motorcycle (3), bus (5), truck (7) and convert to YOLO format.

Pulls ~500 images with rare classes, adds to dataset/train/ and dataset/val/.
No API key needed — uses public COCO URLs.
"""

import json
import os
import shutil
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter

COCO_ANN_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/val2017/"

# COCO category IDs we care about
COCO_CATS  = {3: "motorcycle", 5: "bus", 7: "truck", 2: "car"}
# Map to our class IDs
OUR_IDS    = {3: 1, 5: 2, 7: 3, 2: 0}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

DATASET_DIR   = Path("dataset")
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
VAL_IMG_DIR   = DATASET_DIR / "val"   / "images"
VAL_LBL_DIR   = DATASET_DIR / "val"   / "labels"

ANN_CACHE = Path("coco_cache/instances_val2017.json")
MAX_PER_CLASS = 400   # cap per rare class to avoid bloat
VAL_RATIO     = 0.15


def download_annotations():
    ANN_CACHE.parent.mkdir(exist_ok=True)
    if ANN_CACHE.exists():
        print("  Annotations already cached.")
        return

    zip_path = Path("coco_cache/annotations.zip")
    print("  Downloading COCO annotations (~240 MB)...")
    urllib.request.urlretrieve(
        COCO_ANN_URL,
        zip_path,
        reporthook=lambda b, bs, t: print(f"\r  {min(b*bs,t)/1e6:.0f}/{t/1e6:.0f} MB", end="", flush=True)
    )
    print()

    import zipfile
    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2017.json", "coco_cache/")
    shutil.move("coco_cache/annotations/instances_val2017.json", ANN_CACHE)
    print("  Done.")


def load_coco_image_map(ann_path: Path):
    """Returns: {image_id: {'file_name':..., 'boxes':[...]}}"""
    print("  Parsing COCO annotations...")
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    image_data = defaultdict(lambda: {"file_name": "", "width": 0, "height": 0, "boxes": []})

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in COCO_CATS:
            continue
        img_id = ann["image_id"]
        img    = id_to_img[img_id]
        image_data[img_id]["file_name"] = img["file_name"]
        image_data[img_id]["width"]     = img["width"]
        image_data[img_id]["height"]    = img["height"]

        # COCO bbox: [x, y, w, h] in pixels
        x, y, w, h = ann["bbox"]
        cx = (x + w/2) / img["width"]
        cy = (y + h/2) / img["height"]
        nw = w / img["width"]
        nh = h / img["height"]
        image_data[img_id]["boxes"].append((OUR_IDS[cat_id], cx, cy, nw, nh))

    return image_data


def select_images(image_data: dict) -> list:
    """Select images prioritising rare classes, capped per class."""
    # Count which images have each class
    class_counts = Counter()
    selected = {}

    # First pass: prioritise motorcycle, bus, truck
    for img_id, data in image_data.items():
        classes = {b[0] for b in data["boxes"]}
        rare = classes & {1, 2, 3}  # motorcycle, bus, truck
        if rare:
            selected[img_id] = data
            for c in rare:
                class_counts[c] += 1

    # Trim if over cap
    final = {}
    counts = Counter()
    for img_id, data in selected.items():
        classes = {b[0] for b in data["boxes"]}
        rare = classes & {1, 2, 3}
        if any(counts[c] >= MAX_PER_CLASS for c in rare):
            continue
        final[img_id] = data
        for c in rare:
            counts[c] += 1

    print(f"  Selected {len(final)} images from COCO val2017")
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {counts.get(cls_id, 0)} images containing it")

    return list(final.items())


def download_and_write(items: list):
    n_val   = max(1, int(len(items) * VAL_RATIO))
    val_ids = set(list(range(0, len(items), max(1, len(items)//n_val)))[:n_val])

    added_train = added_val = skipped = 0

    for i, (img_id, data) in enumerate(items):
        fname  = data["file_name"]
        is_val = i in val_ids

        img_dir = VAL_IMG_DIR if is_val else TRAIN_IMG_DIR
        lbl_dir = VAL_LBL_DIR if is_val else TRAIN_LBL_DIR
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        stem    = Path(fname).stem
        out_img = img_dir / f"coco_{stem}.jpg"
        out_lbl = lbl_dir / f"coco_{stem}.txt"

        if out_img.exists():
            skipped += 1
            continue

        url = COCO_IMG_BASE + fname
        try:
            urllib.request.urlretrieve(url, out_img)
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
            continue

        lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in data["boxes"]]
        out_lbl.write_text("\n".join(lines))

        if is_val:
            added_val += 1
        else:
            added_train += 1

        if (added_train + added_val) % 50 == 0:
            print(f"  Progress: {added_train+added_val}/{len(items)-skipped} downloaded")

    print(f"\n  Done: +{added_train} train, +{added_val} val  ({skipped} skipped/cached)")


def main():
    download_annotations()
    image_data = load_coco_image_map(ANN_CACHE)
    items = select_images(image_data)
    print(f"\n  Downloading {len(items)} images from COCO...")
    download_and_write(items)

    # Final count
    print("\n=== Dataset totals after COCO import ===")
    from boost_classes import count_labels
    counts = count_labels(TRAIN_LBL_DIR)
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {counts.get(cls_id, 0)} annotations")


if __name__ == "__main__":
    main()
