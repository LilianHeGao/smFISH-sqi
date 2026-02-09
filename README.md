<p align="center">
  <img src="assets/sqi_logo.png" width="220"/>
</p>

<h1 align="center">SQI</h1>
<p align="center">
  <strong>Sample quality index</strong><br/>
  A modular biological sample-level QC stack for smFISH / MERFISH imaging data
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey"/>
</p>

---

## Overview

**SQI (Sample Quality Index)** is a modular, sample-level quality control framework designed for  
**smFISH / MERFISH imaging-based spatial transcriptomics**.

Rather than focusing on molecule-level chemistry or probe design, SQI quantifies **sample integrity,
spatial signal structure, and image-derived RNA quality** directly from imaging data.

SQI is designed to:
- assess whether a biological sample is *fit-for-use* in MERFISH-style experiments
- support **dataset-level and cell-level QC**
- integrate seamlessly with existing spot-calling and segmentation pipelines

---

## Key design principles

- **Sample-level first**  
  QC metrics are defined at the *biological sample* level, not just FOVs or individual spots.

- **Imaging-native**  
  All metrics are derived from image-space features (spots, masks, spatial structure).

- **Modular & extensible**  
  Each QC component can be enabled, disabled, or replaced independently.

- **Model-agnostic**  
  Compatible with multiple spot callers, segmenters, and imaging platforms.

---

## QC modules (high level)

| Module | Description |
|------|------------|
| Foreground / background separation | Tissue-aware valid mask construction |
| Spot statistics | Density, intensity, spatial consistency |
| Cell-level aggregation | Per-cell RNA signal quality metrics |
| Spatial structure | Local clustering, neighborhood coherence |
| Dataset summary | Sample-level SQI score and diagnostics |

---

## Quickstart

```bash
# create environment (example)
conda create -n sqi python=3.11 -y
conda activate sqi

pip install -r requirements.txt

python run_sqi.py
