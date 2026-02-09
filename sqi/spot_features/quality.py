from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityGateConfig:
    snr_weight: float = 0.3
    score_weight: float = 0.2
    symmetry_weight: float = 0.5
    snr_cap: float = 20.0
    permissive_thresh: float = 0.3
    conservative_snr_min: float = 3.0
    conservative_ellip_max: float = 1.6


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

    snr = df["snr"].values.astype(np.float32)

    # 1. 先处理非有限值
    snr = np.nan_to_num(snr, nan=0.0, neginf=0.0, posinf=cfg.snr_cap)

    # 2. 负 SNR 没有信息量，直接截断
    snr = np.clip(snr, 0.0, None)

    # 3. 再做 log normalization
    snr_norm = np.log1p(snr) / np.log1p(cfg.snr_cap)
    snr_norm = np.clip(snr_norm, 0.0, 1.0).astype(np.float32)


    score_raw = df["score"].values.astype(np.float32)
    score_norm = np.tanh(score_raw / 10.0).astype(np.float32)

    ellip = df["ellipticity"].values.astype(np.float32)
    ellip = np.where(np.isfinite(ellip), ellip, 1.0)
    symmetry = np.exp(-np.abs(np.log(ellip))).astype(np.float32)

    raw = (cfg.snr_weight * snr_norm
           + cfg.score_weight * score_norm
           + cfg.symmetry_weight * symmetry)

    q = np.clip(raw, 0, 1).astype(np.float32)

    out["q_score"] = q
    out["pass_permissive"] = q >= cfg.permissive_thresh

    # Hard physical gate
    out["pass_conservative"] = (
        (df["snr"] >= cfg.conservative_snr_min)
        & (df["ellipticity"] <= cfg.conservative_ellip_max)
    )
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
        np.percentile(score_norm, [5, 50, 95])
    )
    print(
        "[DEBUG] symmetry:",
        np.percentile(symmetry, [5, 50, 95])
    )

    return out
