from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class CellposeNucleiConfig:
    model_type: str = "nuclei"
    use_gpu: Optional[bool] = None
    diameter: Optional[float] = None
    channels: Tuple[int, int] = (0, 0)
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    stitch_threshold: float = 0.0
    batch_size: int = 8


class CellposeBackend:
    def __init__(self, cfg: CellposeNucleiConfig):
        self.cfg = cfg
        self._model = None
        self._resolved_gpu = None

    @staticmethod
    def _auto_gpu() -> bool:
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
        self._model = models.CellposeModel(
            gpu=use_gpu,
            model_type=self.cfg.model_type,
        )
        return self._model

    def segment_nuclei(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={img.shape}")

        model = self._get_model()

        img_f = img.astype(np.float32, copy=False)

        out = model.eval(
            img_f,
            diameter=self.cfg.diameter,
            channels=self.cfg.channels,
            flow_threshold=self.cfg.flow_threshold,
            cellprob_threshold=self.cfg.cellprob_threshold,
            stitch_threshold=self.cfg.stitch_threshold,
            batch_size=self.cfg.batch_size,
        )

        # Cellpose v3: (masks, flows, styles, diams)
        # Cellpose v4+: (masks, flows, styles)
        if len(out) == 4:
            masks, flows, styles, diams = out
        else:
            masks, flows, styles = out
            diams = None

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
