<p align="center">
  <img src="assets/sqi_logo.png" width="320"/>
</p>

---

# SQI – imaging-based sample quality control for smFISH 

**SQI (Sample Quality Index)** is a modular, image-derived quality control (QC) framework for
**smFISH and MERFISH spatial transcriptomics data**.

Unlike molecule-level chemistry or probe-centric pipelines, SQI focuses on **biological sample-level
integrity**, quantifying RNA quality, spatial signal structure, and tissue organization directly
from imaging features.

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey"/>
</p>

---

## Overview

Imaging-based spatial transcriptomics experiments are highly sensitive to **sample preservation,
RNA integrity, and tissue handling**. While existing pipelines emphasize spot calling, decoding,
and cell segmentation, there is limited support for **quantitative, sample-level QC** prior to
downstream biological interpretation.

SQI addresses this gap by providing a **modular QC stack** that operates directly in image space,
enabling:

- Early identification of low-quality or compromised samples
- Comparison of RNA quality across samples and experiments
- Reproducible, interpretable QC metrics at both **dataset** and **cell** levels

---

## How it works

For each cell in a FOV, SQI computes a per-cell **signal-to-noise ratio** from RNA spot densities:

```
SQI(cell) = spot_density(cell-proximal) / spot_density(cell-distal)
```

- **Cell-proximal (FG)**: a dilated ring around each nucleus — where real RNA signal concentrates.
- **Cell-distal (BG)**: valid tissue far from any nucleus — representing background noise / degraded RNA.

High SQI means RNA is concentrated near cells (good sample). SQI near 1 means RNA is diffusely spread (degraded sample).

### Pipeline

1. **DAPI loading** — read the nuclear channel from a per-FOV `.zarr`
2. **Nuclei segmentation** — Cellpose (`segmentation/cellpose_backend.py`)
3. **Tissue mask** — mosaic-level valid-tissue mask, cropped per FOV
4. **FG / BG masks** — binary dilation of nuclei → cell-proximal ring; remainder → cell-distal. Per-cell assignment via Voronoi (EDT)
5. **Spot detection** — Spotiflow, per channel (`sqi/spot_calling/`)
6. **Spot features & quality scoring** — SNR, symmetry, intensity → composite `q_score` with conservative / permissive gates (`sqi/spot_features/`)
7. **SQI computation** — quality-weighted FG/BG density ratio per cell (`sqi/qc/metrics.py`)
8. **Sanity check** — null-model comparison: uniformly sampled pseudo-spots vs real spots. Clear separation validates the metric (`sqi/qc/metrics.py::sqi_sanity_check`)

### Outputs

| File | Description |
|------|-------------|
| `sqi_summary.json` | FOV-level summary (median SQI, mean log10 SQI, per-channel breakdown) |
| `sqi_per_cell.csv` | Per-cell SQI, FG/BG spot counts |
| `sqi_per_cell_per_channel.csv` | Per-cell per-channel SQI |
| `sqi_distribution.png` | log10(SQI) histograms (per-channel + total) |
| `sqi_sanity_check.png` | Real vs null SQI distribution overlay |

<!-- TODO: add example output images -->

---

## Project structure

```
sqi/
  io/              Image & spots I/O (tiff, zarr, parquet)
  qc/              QC core: FG/BG masks, SQI metrics, sanity check, plotting
  spot_calling/    Spotiflow backend
  spot_features/   Per-spot feature extraction & quality scoring
  spatial/         Spatial structure analysis
segmentation/      Cellpose nuclei segmentation backend
scripts/           End-to-end pipelines & visualization helpers
configs/           Configuration files
```

---

## Quickstart

```bash
conda create -n sqi python=3.11 -y
conda activate sqi

pip install -e .
```

### Run on a single FOV

```bash
python scripts/run_sqi_from_fov_zarr.py \
  --fov_zarr  /path/to/Conv_zscan1_XXX.zarr \
  --data_fld  /path/to/parent_folder_with_all_zarrs \
  --cache_root /path/to/cache \
  --out_root   /path/to/output
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--resc` | 4 | Mosaic rescale factor |
| `--cell_proximal_px` | 24 | Dilation radius (px) for cell-proximal region |
| `--spot_model` | `general` | Spotiflow pretrained model |
| `--prob_thresh` | 0.5 | Spotiflow probability threshold |
| `--force` | off | Recompute all cached intermediates |

---

## Dependencies

numpy, scipy, scikit-image, matplotlib, tifffile, zarr, cellpose, spotiflow, pandas, pyarrow
