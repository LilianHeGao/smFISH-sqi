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
    conservative_snr_percentile: float = 50.0
    conservative_ellip_max: float = 1.6


def compute_quality_scores(
    df: pd.DataFrame,
    cfg: QualityGateConfig = QualityGateConfig(),
) -> pd.DataFrame:
    """
    Per-channel quality scoring.

    Within each channel:
      - SNR normalized via log1p, per-channel
      - Score normalized via tanh, per-channel
      - Symmetry shared (ellipticity-based)
      - q_score = weighted sum, clipped to [0, 1]
      - pass_permissive: q_score >= threshold
      - pass_conservative: hard gate with per-channel SNR threshold
        (percentile of FG spots' SNR) + shared ellipticity gate

    Requires 'channel' column in df.
    """
    if "channel" not in df.columns:
        return _score_single_group(df, cfg, snr_thresh_override=None)

    parts = []
    for ch, grp in df.groupby("channel"):
        # Per-channel conservative SNR threshold from FG spots
        fg_snr = grp.loc[grp["in_fg"], "snr"].values
        fg_snr = fg_snr[np.isfinite(fg_snr)]
        if len(fg_snr) > 0:
            snr_thresh = float(np.percentile(fg_snr, cfg.conservative_snr_percentile))
        else:
            snr_thresh = 0.0

        scored = _score_single_group(grp, cfg, snr_thresh_override=snr_thresh)
        parts.append(scored)

    out = pd.concat(parts, ignore_index=True)
    return out


def _score_single_group(
    df: pd.DataFrame,
    cfg: QualityGateConfig,
    snr_thresh_override: float | None,
) -> pd.DataFrame:
    """Score a single channel (or all spots if no channel column)."""
    out = df.copy()

    # --- SNR ---
    snr = df["snr"].values.astype(np.float32)
    snr = np.nan_to_num(snr, nan=0.0, neginf=0.0, posinf=cfg.snr_cap)
    snr = np.clip(snr, 0.0, None)
    snr_norm = np.log1p(snr) / np.log1p(cfg.snr_cap)
    snr_norm = np.clip(snr_norm, 0.0, 1.0).astype(np.float32)

    # --- Score (per-channel rank-based normalization) ---
    score_raw = df["score"].values.astype(np.float32)
    score_norm = np.tanh(score_raw / 10.0).astype(np.float32)

    # --- Symmetry (shared) ---
    ellip = df["ellipticity"].values.astype(np.float32)
    ellip = np.where(np.isfinite(ellip), ellip, 1.0)
    symmetry = np.exp(-np.abs(np.log(ellip))).astype(np.float32)

    # --- q_score ---
    raw = (cfg.snr_weight * snr_norm
           + cfg.score_weight * score_norm
           + cfg.symmetry_weight * symmetry)
    q = np.clip(raw, 0, 1).astype(np.float32)

    out["q_score"] = q
    out["pass_permissive"] = q >= cfg.permissive_thresh

    # --- Conservative: hard physical gate ---
    snr_min = snr_thresh_override if snr_thresh_override is not None else 0.0
    out["pass_conservative"] = (
        (df["snr"] >= snr_min)
        & (df["ellipticity"] <= cfg.conservative_ellip_max)
    )

    return out
