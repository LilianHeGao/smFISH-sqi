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
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        try:
            from cellpose import core
            return bool(core.use_gpu())
        except Exception:
            return False

    @staticmethod
    def _cp_major() -> int:
        try:
            import cellpose as _cp
            return int(_cp.__version__.split(".")[0])
        except Exception:
            return 3

    def _get_model(self):
        if self._model is not None:
            return self._model

        if self.cfg.use_gpu is None:
            use_gpu = self._auto_gpu()
        else:
            use_gpu = bool(self.cfg.use_gpu)

        self._resolved_gpu = use_gpu

        # models.Cellpose is the stable high-level wrapper that respects
        # model_type in both v3 and v4 (unlike CellposeModel which ignores
        # model_type in v4+ and silently loads cyto3 instead).
        from cellpose import models
        self._model = models.Cellpose(
            gpu=use_gpu,
            model_type=self.cfg.model_type,
        )
        return self._model

    def segment_nuclei(self, img: np.ndarray):
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={img.shape}")

        model = self._get_model()

        img_f = img.astype(np.float32, copy=False)

        eval_kwargs: dict = dict(
            diameter=self.cfg.diameter,
            flow_threshold=self.cfg.flow_threshold,
            cellprob_threshold=self.cfg.cellprob_threshold,
            stitch_threshold=self.cfg.stitch_threshold,
            batch_size=self.cfg.batch_size,
        )
        # channels parameter removed in v4+ for grayscale images
        if self._cp_major() < 4:
            eval_kwargs["channels"] = list(self.cfg.channels)

        out = model.eval(img_f, **eval_kwargs)

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
