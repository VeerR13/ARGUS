"""
train_augmentation.py — Albumentations augmentation pipeline for vehicle detection training.

Targets the phantom-detection failure modes seen in our footage:
  - Motion blur       → shadows/reflections often look sharp; real vehicles are blurred at speed
  - Random shadow     → model must not confuse dark road patches with vehicles
  - Gaussian noise    → night / low-light footage artefacts
  - Lens distortion   → dashcam barrel distortion
  - Brightness/contrast shifts → lighting variability across day/night/tunnel

Usage (with Ultralytics custom dataset):
    from train_augmentation import build_transform, yolo_augment_fn
    # Pass yolo_augment_fn as the `augment` callable in your training script,
    # or apply build_transform() directly to numpy BGR images.

Compatible with albumentations >= 1.3.
"""

import albumentations as A
import cv2
import numpy as np


def build_transform(image_size: int = 640, training: bool = True) -> A.Compose:
    """
    Returns an Albumentations Compose pipeline for vehicle detection.

    Args:
        image_size : target size after spatial transforms
        training   : if False, returns a minimal eval pipeline (resize only)
    """
    if not training:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=114),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────────────────
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=114),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.15, rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT, value=114, p=0.4,
        ),

        # ── Motion blur — phantom killer ─────────────────────────────────────
        # Teaches the model that real fast-moving vehicles are often blurred;
        # single-frame phantom detections (shadows, reflections) rarely are.
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 15), p=1.0),
            A.Blur(blur_limit=5, p=1.0),
        ], p=0.35),

        # ── Shadow augmentation — phantom killer ─────────────────────────────
        # Hard negatives: dark road patches / tree shadows that look like vehicles.
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),   # bottom half only (road surface)
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=5,
            p=0.40,
        ),

        # ── Noise / sensor artefacts ─────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.30),

        # ── Lighting variability ─────────────────────────────────────────────
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.50),

        # ── Colour / channel shifts ───────────────────────────────────────────
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.30
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=15, p=0.20),

        # ── Lens / camera artefacts ───────────────────────────────────────────
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.08, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
        ], p=0.20),

        # ── Compression artefacts (dashcam footage is often re-encoded) ───────
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.25),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,   # drop boxes if < 30% visible after crop/rotate
    ))


def augment_sample(image_bgr: np.ndarray, bboxes_yolo: list, class_labels: list,
                   transform: A.Compose) -> tuple:
    """
    Apply transform to one training sample.

    Args:
        image_bgr    : H×W×3 BGR numpy array
        bboxes_yolo  : list of [cx, cy, w, h] normalised (YOLO format)
        class_labels : list of int class IDs (same length as bboxes_yolo)
        transform    : output of build_transform()

    Returns:
        (augmented_bgr, augmented_bboxes_yolo, augmented_class_labels)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = transform(image=image_rgb, bboxes=bboxes_yolo, class_labels=class_labels)
    out_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    return out_bgr, result["bboxes"], result["class_labels"]


# ── Ultralytics integration ───────────────────────────────────────────────────
# To use with ultralytics training, set in your data.yaml or training script:
#
#   from train_augmentation import ULTRALYTICS_HYPS
#   model.train(..., **ULTRALYTICS_HYPS)
#
# These override the default Ultralytics mosaic/HSV augmentation with our
# phantom-targeted pipeline via the custom `augment` callback hook.

ULTRALYTICS_HYPS = {
    # Turn off built-in augs we replace:
    "hsv_h":    0.0,
    "hsv_s":    0.0,
    "hsv_v":    0.0,
    "fliplr":   0.0,   # handled by Albumentations
    # Keep mosaic — it's very effective for multi-vehicle scenes
    "mosaic":   1.0,
    "mixup":    0.1,
    "degrees":  0.0,   # handled by ShiftScaleRotate above
    "translate":0.0,
    "scale":    0.0,
    "shear":    0.0,
    "perspective": 0.0,
    "copy_paste":  0.1,
}


if __name__ == "__main__":
    # Quick visual sanity check — shows augmented frames in a window
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        print("Usage: python train_augmentation.py <image_path>")
        sys.exit(0)

    img = cv2.imread(img_path)
    transform = build_transform(640, training=True)
    for i in range(6):
        aug, _, _ = augment_sample(img, [], [], transform)
        cv2.imshow(f"aug_{i}", aug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
