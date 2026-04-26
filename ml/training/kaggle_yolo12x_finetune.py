"""
kaggle_yolo12x_finetune.py — YOLO12x finetuning on BDD100K + IDD
                              designed to run on Kaggle T4 x2 / P100 GPU.

Model: YOLO12x (ultralytics 8.4.x) — flagship YOLO as of 2025.
  Starts from COCO pretrained weights (yolo12x.pt), finetuned on urban
  traffic data to improve performance on night/rain/Indian-traffic conditions.

Dataset sources:
  BDD100K — 70k train / 10k val, diverse driving conditions (day/night/rain)
             Kaggle: a7madmostafa/bdd100k-yolo  (already in YOLO format)

  IDD     — ~10k train / 2k val, Indian traffic (autos, mixed density)
             Pascal VOC XML format, converted here to YOLO
             Kaggle: abhishekprajapat/idd-20k

Target classes (5):
  0 car  1 motorcycle  2 bus  3 truck  4 bicycle

Usage on Kaggle:
  1. Attach both datasets
  2. Set GPU accelerator (T4 x2 recommended)
  3. Run all cells
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET

import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BDD100K_DIR    = "/kaggle/input/datasets/a7madmostafa/bdd100k-yolo"
IDD_IMAGES_DIR = "/kaggle/input/idd-20k/IDD_Detection"
IDD_LABELS_DIR = "/kaggle/input/idd-20k/IDD_Detection"

OUTPUT_DATASET_DIR = "/kaggle/working/argus_dataset"
OUTPUT_MODEL_DIR   = "/kaggle/working/models"

INCLUDE_AUTORICKSHAW = False

EPOCHS      = 50
BATCH_SIZE  = 16    # T4 x2: 16 is safe at imgsz=640; bump to 32 on A100
IMGSZ       = 640
NUM_WORKERS = 4
MODEL_NAME  = "yolo12x.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Class mappings
# ─────────────────────────────────────────────────────────────────────────────

if INCLUDE_AUTORICKSHAW:
    CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle", "autorickshaw"]
    BDD100K_CLASS_MAP = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3, "bicycle": 4}
    IDD_CLASS_MAP = {
        "car": 0, "taxi": 0, "van": 0, "jeep": 0,
        "motorcycle": 1, "scooter": 1, "moped": 1,
        "bus": 2, "minibus": 2,
        "truck": 3, "pickup": 3, "trailer": 3, "tipper": 3,
        "bicycle": 4,
        "autorickshaw": 5, "auto-rickshaw": 5, "e-rickshaw": 5,
    }
else:
    CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
    BDD100K_CLASS_MAP = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3, "bicycle": 4}
    IDD_CLASS_MAP = {
        "car": 0, "taxi": 0, "van": 0, "jeep": 0,
        "motorcycle": 1, "scooter": 1, "moped": 1,
        "bus": 2, "minibus": 2,
        "truck": 3, "pickup": 3, "trailer": 3, "tipper": 3,
        "bicycle": 4,
        "autorickshaw": 0, "auto-rickshaw": 0, "e-rickshaw": 0,
    }


def _xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h, cls_id):
    cx = max(0.001, min(0.999, ((x1 + x2) / 2) / img_w))
    cy = max(0.001, min(0.999, ((y1 + y2) / 2) / img_h))
    w  = max(0.001, min(0.999, (x2 - x1) / img_w))
    h  = max(0.001, min(0.999, (y2 - y1) / img_h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# BDD100K — already in YOLO format on Kaggle, just remap class IDs
# ─────────────────────────────────────────────────────────────────────────────

def convert_bdd100k(split: str, out_img_dir: str, out_lbl_dir: str) -> int:
    """
    BDD100K (a7madmostafa/bdd100k-yolo) is pre-converted to YOLO format.
    We copy images and remap any label class IDs to our 5-class scheme.
    """
    print(f"\nProcessing BDD100K {split}...")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Try common layout variants from the Kaggle dataset
    src_img = None
    src_lbl = None
    for candidate in [
        (os.path.join(BDD100K_DIR, split, "images"),  os.path.join(BDD100K_DIR, split, "labels")),
        (os.path.join(BDD100K_DIR, "images", split),  os.path.join(BDD100K_DIR, "labels", split)),
        (os.path.join(BDD100K_DIR, split),            os.path.join(BDD100K_DIR, split)),
    ]:
        if os.path.isdir(candidate[0]):
            src_img, src_lbl = candidate
            break

    if src_img is None:
        # Fall back: try JSON annotation mode
        return _convert_bdd100k_json(split, out_img_dir, out_lbl_dir)

    written = 0
    for img_file in os.listdir(src_img):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem    = os.path.splitext(img_file)[0]
        lbl_src = os.path.join(src_lbl, stem + ".txt")
        img_src = os.path.join(src_img, img_file)
        if not os.path.exists(lbl_src):
            continue

        shutil.copy(img_src, os.path.join(out_img_dir, "bdd_" + img_file))
        shutil.copy(lbl_src, os.path.join(out_lbl_dir, "bdd_" + stem + ".txt"))
        written += 1

    print(f"  BDD100K {split}: {written} images")
    return written


def _convert_bdd100k_json(split: str, out_img_dir: str, out_lbl_dir: str) -> int:
    """Fallback: convert from BDD100K raw JSON annotations."""
    label_file = os.path.join(BDD100K_DIR, "bdd100k", "labels",
                              f"det_20/det_{split}.json")
    if not os.path.exists(label_file):
        label_file = os.path.join(BDD100K_DIR, "bdd100k", "labels",
                                  f"bdd100k_labels_images_{split}.json")
    if not os.path.exists(label_file):
        print(f"  BDD100K {split}: no annotation file found — skipping")
        return 0

    with open(label_file) as f:
        annotations = json.load(f)

    written = 0
    for ann in annotations:
        img_name = ann["name"]
        img_src  = os.path.join(BDD100K_DIR, "bdd100k", "images", "100k", split, img_name)
        if not os.path.exists(img_src):
            continue
        IMG_W, IMG_H = 1280, 720
        lines = []
        for label in ann.get("labels", []):
            cls_id = BDD100K_CLASS_MAP.get(label.get("category", ""))
            if cls_id is None:
                continue
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(_xyxy_to_yolo(x1, y1, x2, y2, IMG_W, IMG_H, cls_id))
        if not lines:
            continue
        stem = os.path.splitext(img_name)[0]
        shutil.copy(img_src, os.path.join(out_img_dir, "bdd_" + img_name))
        with open(os.path.join(out_lbl_dir, "bdd_" + stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    print(f"  BDD100K {split}: {written} images")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# IDD conversion (Pascal VOC XML → YOLO)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_idd_xml(xml_path: str) -> tuple[int, int, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size  = root.find("size")
    img_w = int(size.find("width").text)  if size is not None else 0
    img_h = int(size.find("height").text) if size is not None else 0
    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        cls_id = IDD_CLASS_MAP.get(name_el.text.strip().lower())
        if cls_id is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1 = float(bndbox.find("xmin").text)
        y1 = float(bndbox.find("ymin").text)
        x2 = float(bndbox.find("xmax").text)
        y2 = float(bndbox.find("ymax").text)
        if x2 <= x1 or y2 <= y1 or img_w <= 0 or img_h <= 0:
            continue
        lines.append(_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h, cls_id))
    return img_w, img_h, lines


def convert_idd(split: str, out_img_dir: str, out_lbl_dir: str) -> int:
    print(f"\nConverting IDD {split}...")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    ann_root = os.path.join(IDD_LABELS_DIR, "Annotations", split)
    img_root = os.path.join(IDD_IMAGES_DIR, "JPEGImages", split)

    if not os.path.isdir(ann_root):
        print(f"  IDD Annotations/{split} not found — skipping")
        return 0

    written = 0
    for seq in sorted(os.listdir(ann_root)):
        seq_ann = os.path.join(ann_root, seq)
        seq_img = os.path.join(img_root, seq)
        if not os.path.isdir(seq_ann):
            continue
        for xml_file in os.listdir(seq_ann):
            if not xml_file.endswith(".xml"):
                continue
            stem     = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(seq_ann, xml_file)
            img_w, img_h, lines = _parse_idd_xml(xml_path)
            if not lines:
                continue
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = os.path.join(seq_img, stem + ext)
                if os.path.exists(c):
                    img_path = c
                    break
            if img_path is None:
                continue
            if img_w == 0 or img_h == 0:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                _, _, lines = _parse_idd_xml(xml_path)
                if not lines:
                    continue
            out_stem = f"idd_{seq}_{stem}"
            dst_img  = os.path.join(out_img_dir, out_stem + ".jpg")
            dst_lbl  = os.path.join(out_lbl_dir, out_stem + ".txt")
            if img_path.endswith((".jpg", ".jpeg")):
                shutil.copy(img_path, dst_img)
            else:
                img = cv2.imread(img_path)
                cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines) + "\n")
            written += 1

    print(f"  IDD {split}: {written} images")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_data_yaml(dataset_dir: str) -> str:
    abs_dir   = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# ARGUS YOLO12x finetuning — BDD100K + IDD merged dataset
path: {abs_dir}
train: train/images
val:   valid/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
    print(f"Wrote {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(yaml_path: str, output_dir: str) -> str:
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting YOLO12x finetuning: {EPOCHS} epochs, batch={BATCH_SIZE}, imgsz={IMGSZ}")
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        device="0,1",           # T4 x2 (multi-GPU); change to "0" for single GPU
        workers=NUM_WORKERS,
        project=output_dir,
        name="yolo12x_bdd_idd",
        exist_ok=True,
        # Optimizer
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        # Augmentation — on by default in ultralytics, these tweak for traffic
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,            # dashcam is always horizontal
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Quality of life
        patience=15,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
    )

    run_dir  = os.path.join(output_dir, "yolo12x_bdd_idd")
    best_pt  = os.path.join(run_dir, "weights", "best.pt")
    final_pt = os.path.join(output_dir, "argus_yolo12x_best.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, final_pt)
        print(f"\nSaved: {final_pt}")
    return final_pt


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    train_img = os.path.join(OUTPUT_DATASET_DIR, "train", "images")
    train_lbl = os.path.join(OUTPUT_DATASET_DIR, "train", "labels")
    val_img   = os.path.join(OUTPUT_DATASET_DIR, "valid", "images")
    val_lbl   = os.path.join(OUTPUT_DATASET_DIR, "valid", "labels")

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    n_bdd_train = convert_bdd100k("train", train_img, train_lbl)
    n_bdd_val   = convert_bdd100k("val",   val_img,   val_lbl)
    n_idd_train = convert_idd("train", train_img, train_lbl)
    n_idd_val   = convert_idd("val",   val_img,   val_lbl)

    print(f"\nDataset totals:")
    print(f"  Train: BDD100K={n_bdd_train}  IDD={n_idd_train}  total={n_bdd_train + n_idd_train}")
    print(f"  Valid: BDD100K={n_bdd_val}    IDD={n_idd_val}    total={n_bdd_val + n_idd_val}")
    print(f"  Classes: {CLASS_NAMES}")

    yaml_path = write_data_yaml(OUTPUT_DATASET_DIR)
    final_pt  = run_training(yaml_path, OUTPUT_MODEL_DIR)

    print(f"\nDone. Best weights: {final_pt}")
    print("Download and use via:")
    print("  VehicleDetector(model_path='argus_yolo12x_best.pt')")


if __name__ == "__main__":
    main()
