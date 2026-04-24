"""
kaggle_rfdetr_finetune.py — RF-DETR finetuning on BDD100K + IDD
                             designed to run on Kaggle T4/P100 GPU.

Dataset sources:
  BDD100K — 70k train / 10k val, 5 vehicle classes
             Download: https://bdd-data.berkeley.edu/ (registration required)
             Upload as private Kaggle dataset: your-username/bdd100k-raw

  IDD     — ~10k train / 2k val, mixed Indian traffic including autorickshaws
             Pascal VOC XML format (IDD-Detection, NOT the segmentation variant)
             Download: https://idd.insaan.iiit.ac.in/ (registration required)
             Upload as private Kaggle dataset: your-username/idd-detection
             Expected layout: IDD_Detection/Annotations/{train,val}/{seq}/*.xml
                              IDD_Detection/JPEGImages/{train,val}/{seq}/*.jpg

Target classes (5):
  0 car  1 motorcycle  2 bus  3 truck  4 bicycle

Autorickshaw (IDD-specific) is mapped to car by default; set
INCLUDE_AUTORICKSHAW=True to add a 6th class for better precision.

Usage on Kaggle:
  1. Attach both datasets to the notebook
  2. Set dataset paths below
  3. Run all cells (GPU accelerator on)
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Set these to your Kaggle input paths
BDD100K_IMAGES_DIR = "/kaggle/input/bdd100k-raw/bdd100k/images/100k"
BDD100K_LABELS_DIR = "/kaggle/input/bdd100k-raw/bdd100k/labels"
IDD_IMAGES_DIR     = "/kaggle/input/idd-detection/IDD_Detection"
IDD_LABELS_DIR     = "/kaggle/input/idd-detection/IDD_Detection"  # same root

OUTPUT_DATASET_DIR = "/kaggle/working/argus_dataset"
OUTPUT_MODEL_DIR   = "/kaggle/working/models"

INCLUDE_AUTORICKSHAW = False   # True → 6 classes; False → map autorickshaw → car

EPOCHS     = 50
BATCH_SIZE = 8      # T4 GPU safe with RF-DETR base at 560px
NUM_WORKERS = 4

# ─────────────────────────────────────────────────────────────────────────────
# Class mappings
# ─────────────────────────────────────────────────────────────────────────────

if INCLUDE_AUTORICKSHAW:
    CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle", "autorickshaw"]
    BDD100K_CLASS_MAP = {
        "car": 0, "motorcycle": 1, "bus": 2, "truck": 3, "bicycle": 4,
    }
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
    BDD100K_CLASS_MAP = {
        "car": 0, "motorcycle": 1, "bus": 2, "truck": 3, "bicycle": 4,
    }
    IDD_CLASS_MAP = {
        "car": 0, "taxi": 0, "van": 0, "jeep": 0,
        "motorcycle": 1, "scooter": 1, "moped": 1,
        "bus": 2, "minibus": 2,
        "truck": 3, "pickup": 3, "trailer": 3, "tipper": 3,
        "bicycle": 4,
        "autorickshaw": 0,   # map to car — 3-wheelers behave like slow cars in lanes
        "auto-rickshaw": 0,
        "e-rickshaw": 0,
    }


def _xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h, cls_id):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    cx = max(0.001, min(0.999, cx))
    cy = max(0.001, min(0.999, cy))
    w  = max(0.001, min(0.999, w))
    h  = max(0.001, min(0.999, h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# BDD100K conversion
# BDD100K images are 1280x720, annotations in JSON format
# ─────────────────────────────────────────────────────────────────────────────

def convert_bdd100k(split: str, out_img_dir: str, out_lbl_dir: str) -> int:
    """Convert BDD100K split ('train'/'val') to YOLO format."""
    print(f"\nConverting BDD100K {split}...")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    label_file = os.path.join(BDD100K_LABELS_DIR,
                              f"det_20/det_{split}.json")
    if not os.path.exists(label_file):
        # Try alternate path layout
        label_file = os.path.join(BDD100K_LABELS_DIR,
                                  f"bdd100k_labels_images_{split}.json")

    with open(label_file) as f:
        annotations = json.load(f)

    written = 0
    for ann in annotations:
        img_name = ann["name"]
        img_src  = os.path.join(BDD100K_IMAGES_DIR, split, img_name)
        if not os.path.exists(img_src):
            continue

        IMG_W, IMG_H = 1280, 720   # BDD100K is always 1280×720
        lines = []

        for label in ann.get("labels", []):
            category = label.get("category", "")
            cls_id   = BDD100K_CLASS_MAP.get(category)
            if cls_id is None:
                continue
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1 = box2d["x1"], box2d["y1"]
            x2, y2 = box2d["x2"], box2d["y2"]
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(_xyxy_to_yolo(x1, y1, x2, y2, IMG_W, IMG_H, cls_id))

        if not lines:
            continue

        stem     = os.path.splitext(img_name)[0]
        dst_img  = os.path.join(out_img_dir, "bdd_" + img_name)
        dst_lbl  = os.path.join(out_lbl_dir, "bdd_" + stem + ".txt")

        shutil.copy(img_src, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    print(f"  BDD100K {split}: {written} images converted")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# IDD conversion
# IDD-Detection uses Pascal VOC XML annotations.
# Directory layout:
#   IDD_Detection/Annotations/{split}/{sequence}/{frame}.xml
#   IDD_Detection/JPEGImages/{split}/{sequence}/{frame}.jpg
# ─────────────────────────────────────────────────────────────────────────────

def _parse_idd_xml(xml_path: str) -> tuple[int, int, list[str]]:
    """
    Parse one IDD Pascal VOC XML file.
    Returns (img_w, img_h, yolo_lines).
    img_w/img_h come from <size> tag; fall back to cv2 read if absent.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
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
    """
    Walk IDD-Detection Pascal VOC tree and convert to YOLO format.
    Annotations: IDD_Detection/Annotations/{split}/{seq}/{frame}.xml
    Images:      IDD_Detection/JPEGImages/{split}/{seq}/{frame}.jpg
    """
    print(f"\nConverting IDD {split}...")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    ann_root = os.path.join(IDD_LABELS_DIR, "Annotations", split)
    img_root = os.path.join(IDD_IMAGES_DIR, "JPEGImages", split)

    if not os.path.isdir(ann_root):
        print(f"  IDD Annotations/{split} not found at {ann_root} — skipping")
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
            xml_path = os.path.join(seq_ann, xml_file)
            stem     = os.path.splitext(xml_file)[0]

            img_w, img_h, lines = _parse_idd_xml(xml_path)
            if not lines:
                continue

            # Find corresponding image (.jpg or .png)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(seq_img, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # If size was missing from XML, read from image
            if img_w == 0 or img_h == 0:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                # Re-parse with correct dimensions
                _, _, lines = _parse_idd_xml(xml_path)
                if not lines:
                    continue

            out_stem = f"idd_{seq}_{stem}"
            dst_img  = os.path.join(out_img_dir, out_stem + ".jpg")
            dst_lbl  = os.path.join(out_lbl_dir, out_stem + ".txt")

            if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):
                shutil.copy(img_path, dst_img)
            else:
                img = cv2.imread(img_path)
                cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines) + "\n")
            written += 1

    print(f"  IDD {split}: {written} images converted")
    return written


def write_data_yaml(dataset_dir: str) -> str:
    abs_dir   = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# ARGUS RF-DETR finetuning — BDD100K + IDD merged dataset
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

def run_training(dataset_dir: str, output_dir: str,
                 epochs: int, batch: int) -> str:
    from rfdetr import RFDETRBase

    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, "rfdetr_bdd_idd")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nStarting RF-DETR finetuning: {epochs} epochs, batch={batch}")
    model = RFDETRBase()
    model.train(
        dataset_file="yolo",
        dataset_dir=os.path.abspath(dataset_dir),
        output_dir=run_dir,
        epochs=epochs,
        batch_size=batch,
        lr=5e-5,            # lower LR — large dataset, avoid overshooting pretrained features
        lr_encoder=7.5e-5,
        device="cuda",
        class_names=CLASS_NAMES,
        progress_bar="tqdm",
        tensorboard=True,
        checkpoint_interval=5,
        early_stopping=True,
        early_stopping_patience=10,
        num_workers=NUM_WORKERS,
        multi_scale=True,
        expanded_scales=True,
    )

    final = os.path.join(output_dir, "rfdetr_bdd_idd_final.pth")
    for candidate in ["checkpoint_best_total.pth", "checkpoint_best_ap50.pth",
                       "checkpoint_last.pth"]:
        src = os.path.join(run_dir, candidate)
        if os.path.exists(src):
            shutil.copy(src, final)
            print(f"Saved: {final}  (from {candidate})")
            break
    return final


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

    # --- Convert BDD100K ---
    n_bdd_train = convert_bdd100k("train", train_img, train_lbl)
    n_bdd_val   = convert_bdd100k("val",   val_img,   val_lbl)

    # --- Convert IDD ---
    n_idd_train = convert_idd("train", train_img, train_lbl)
    n_idd_val   = convert_idd("val",   val_img,   val_lbl)

    print(f"\nDataset totals:")
    print(f"  Train: BDD100K={n_bdd_train}  IDD={n_idd_train}  "
          f"total={n_bdd_train + n_idd_train}")
    print(f"  Valid: BDD100K={n_bdd_val}    IDD={n_idd_val}    "
          f"total={n_bdd_val + n_idd_val}")
    print(f"  Classes: {CLASS_NAMES}")

    _ = write_data_yaml(OUTPUT_DATASET_DIR)

    final_model = run_training(OUTPUT_DATASET_DIR, OUTPUT_MODEL_DIR,
                               EPOCHS, BATCH_SIZE)

    print(f"\nDone. Checkpoint: {final_model}")
    print("Download from Kaggle output and place at:")
    print("  ARGUS/rf-detr-base.pth   (replace default pretrained weights)")
    print("Or pass via ml_pipeline.detection.VehicleDetector(pretrain_weights=<path>)")


if __name__ == "__main__":
    main()
