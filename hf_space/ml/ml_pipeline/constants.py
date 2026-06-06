"""Shared constants for the ARGUS ML pipeline."""

# COCO-80 class IDs for the 4 vehicle types — used with pretrained YOLO12x.
COCO_VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Class IDs for the finetuned ARGUS model (BDD100K + IDD, 5-class scheme).
# When YOLO12x is finetuned on a custom dataset the model re-indexes classes
# starting at 0, so these IDs differ from COCO.
FINETUNED_VEHICLE_CLASSES: dict[int, str] = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "bicycle",
}
