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
    permissive_thresh: float = 0.3
    conservative_snr_min: float = 5.0
    conservative_ellip_max: float = 1.4


def compute_quality_scores(
    df: pd.DataFrame,
    cfg: QualityGateConfig = QualityGateConfig(),
) -> pd.DataFrame:
    """
    Add q_score, pass_permissive, pass_conservative columns.

    q_score is normalized to [0, 1]:
        raw = w_snr * clip(snr/snr_cap, 0, 1)
            + w_score * score
            + w_symmetry * symmetry
        q_score = raw / (w_snr + w_score + w_symmetry)

    pass_permissive: soft gate on q_score >= permissive_thresh.
    pass_conservative: hard physical gate (snr >= 5 AND ellipticity <= 1.4).
    """
    out = df.copy()

    snr_norm = np.clip(df["snr"].values / cfg.snr_cap, 0, 1).astype(np.float32)

    score_vals = np.clip(df["score"].values, 0, 1).astype(np.float32)

    ellip = df["ellipticity"].values.astype(np.float32)
    ellip = np.where(np.isfinite(ellip), ellip, 0.5)
    symmetry = np.clip(1.0 - ellip, 0, 1).astype(np.float32)

    raw = (cfg.snr_weight * snr_norm
           + cfg.score_weight * score_vals
           + cfg.symmetry_weight * symmetry)

    w_sum = cfg.snr_weight + cfg.score_weight + cfg.symmetry_weight
    q = np.clip(raw / w_sum, 0, 1).astype(np.float32)

    out["q_score"] = q
    out["pass_permissive"] = q >= cfg.permissive_thresh

    # Hard physical gate
    out["pass_conservative"] = (
        (df["snr"] >= cfg.conservative_snr_min)
        & (df["ellipticity"] <= cfg.conservative_ellip_max)
    )

    return out
