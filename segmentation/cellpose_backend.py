from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

# cellpose import is intentionally inside functions where possible,
# to keep import-time failures localized (useful in mixed CPU/GPU setups).


@dataclass
class CellposeNucleiConfig:
    """
    Cellpose config for nuclei segmentation.

    Key design choices:
    - model_type defaults to 'nuclei' (robust for DAPI-like channels).
    - channels follow Cellpose convention:
        (0,0) = grayscale (single channel)
        (nuclei_channel, cytoplasm_channel) otherwise
    - diameter can be None (auto) but for consistent QC benchmarking
      you usually want to lock it once estimated.
    """
    model_type: str = "nuclei"
    use_gpu: Optional[bool] = None   # None -> auto
    diameter: Optional[float] = None
    channels: Tuple[int, int] = (0, 0)
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    stitch_threshold: float = 0.0
    batch_size: int = 8


class CellposeBackend:
    """
    A thin wrapper around Cellpose with stable I/O:
    input  : 2D image (H,W) float/uint
    output : labels (H,W) int32, + metadata dict

    This wrapper isolates Cellpose specifics so downstream QC code stays model-agnostic.
    """

    def __init__(self, cfg: CellposeNucleiConfig):
        self.cfg = cfg
        self._model = None
        self._resolved_gpu = None

    @staticmethod
    def _auto_gpu() -> bool:
        # Conservative auto-detect: let Cellpose decide if GPU is usable.
        # This avoids hard-coding torch/cuda logic here.
        try:
            from cellpose import core
            return bool(core.use_gpu())
        except Exception:
            return False

    def _get_model(self):
        if self._model is not None:
            return self._model

        if self.cfg.use_gpu is None:
            use_gpu = self._auto_gpu()
        else:
            use_gpu = bool(self.cfg.use_gpu)

        self._resolved_gpu = use_gpu

        from cellpose import models
        self._model = models.Cellpose(gpu=use_gpu, model_type=self.cfg.model_type)
        return self._model

    def segment_nuclei(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment nuclei from a single 2D image.

        Returns
        -------
        labels : (H,W) int32, 0=background, 1..N nuclei ids
        meta   : dict containing parameters and basic QC stats
        """
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image (H,W), got shape={img.shape}")

        model = self._get_model()

        # Cellpose works best with float32
        img_f = img.astype(np.float32, copy=False)

        masks, flows, styles, diams = model.eval(
            img_f,
            diameter=self.cfg.diameter,
            channels=self.cfg.channels,
            flow_threshold=self.cfg.flow_threshold,
            cellprob_threshold=self.cfg.cellprob_threshold,
            stitch_threshold=self.cfg.stitch_threshold,
            batch_size=self.cfg.batch_size,
        )

        labels = masks.astype(np.int32, copy=False)

        meta = {
            "model_type": self.cfg.model_type,
            "use_gpu": self._resolved_gpu,
            "diameter_in": self.cfg.diameter,
            "diameter_out": float(diams) if diams is not None else None,
            "flow_threshold": self.cfg.flow_threshold,
            "cellprob_threshold": self.cfg.cellprob_threshold,
            "n_nuclei": int(labels.max()),
            "img_shape": tuple(labels.shape),
        }
        return labels, meta
