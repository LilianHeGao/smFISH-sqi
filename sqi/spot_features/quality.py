from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityGateConfig:
    snr_weight: float = 0.4
    score_weight: float = 0.3
    symmetry_weight: float = 0.3
    snr_cap: float = 20.0
    permissive_thresh: float = 0.6
    conservative_thresh: float = 0.8


def compute_quality_scores(
    df: pd.DataFrame,
    cfg: QualityGateConfig = QualityGateConfig(),
) -> pd.DataFrame:
    """
    Add q_score, pass_permissive, pass_conservative columns.

    q_score = w_snr * clip(snr/snr_cap, 0, 1)
            + w_score * score
            + w_symmetry * (1 - ellipticity)

    NaN ellipticity treated as 0.5 (neutral).
    """
    out = df.copy()

    snr_norm = np.clip(df["snr"].values / cfg.snr_cap, 0, 1).astype(np.float32)

    score_vals = df["score"].values.astype(np.float32)

    ellip = df["ellipticity"].values.astype(np.float32)
    ellip = np.where(np.isfinite(ellip), ellip, 0.5)
    symmetry = np.exp(-np.abs(np.log(ellip))).astype(np.float32)

    q = (cfg.snr_weight * snr_norm
         + cfg.score_weight * score_vals
         + cfg.symmetry_weight * symmetry).astype(np.float32)

    out["q_score"] = q
    out["pass_permissive"] = q >= cfg.permissive_thresh
    out["pass_conservative"] = q >= cfg.conservative_thresh
    print(
    "[DEBUG] q_score:",
    np.percentile(q, [0, 5, 25, 50, 75, 95, 100])
)
    print(
    "[DEBUG] snr_norm:",
    np.percentile(snr_norm, [5, 50, 95])
)
    print(
        "[DEBUG] score:",
        np.percentile(score_vals, [5, 50, 95])
    )
    print(
        "[DEBUG] symmetry:",
        np.percentile(symmetry, [5, 50, 95])
    )
    return out