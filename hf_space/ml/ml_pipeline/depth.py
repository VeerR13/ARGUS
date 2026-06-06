"""
Relative depth estimation via Depth-Anything-V2-Small.

Depth map range: 0 = near camera, 1 = far away.
Used as a secondary TTC signal: a vehicle whose bbox depth percentile
drops quickly frame-over-frame is closing faster than pixel velocity alone
can indicate (useful when the vehicle is large and fills most of the frame).

Usage:
    estimator = DepthEstimator()               # loads model once
    depth_map  = estimator.estimate(frame_rgb) # H×W float32 [0,1]
    d = estimator.bbox_depth(depth_map, (l,t,r,b))  # scalar [0,1]
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline

_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Thin wrapper around DepthAnything V2 Small (~25 ms/frame on GPU)."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = hf_pipeline(
            "depth-estimation",
            model=_MODEL_ID,
            device=device,
        )

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame_rgb : H×W×3 uint8 array (RGB order)

        Returns
        -------
        depth_map : H×W float32, range [0, 1], 0 = near, 1 = far
        """
        result = self._pipe(Image.fromarray(frame_rgb))
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def bbox_depth(
        self,
        depth_map: np.ndarray,
        bbox_ltrb: tuple[int, int, int, int],
    ) -> float:
        """Median depth in the central 50% of the bounding box.

        Using the centre crop avoids background bleed at box edges.

        Returns a value in [0, 1]: lower means closer to the camera.
        """
        l, t, r, b = bbox_ltrb
        h, w = b - t, r - l
        cy, cx = (t + b) // 2, (l + r) // 2
        hh = max(h // 4, 1)
        hw = max(w // 4, 1)
        crop = depth_map[cy - hh : cy + hh, cx - hw : cx + hw]
        if crop.size == 0:
            return float(depth_map[cy, cx])
        return float(np.median(crop))
