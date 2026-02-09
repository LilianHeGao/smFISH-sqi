<p align="center">
  <img src="assets/sqi_logo.png" width="320"/>
</p>

# SQI — Sample Quality Index for smFISH

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey"/>
</p>

A lightweight QC toolkit that scores **sample-level RNA integrity** in smFISH spatial transcriptomics, directly from images.

## The problem

Most spatial transcriptomics pipelines focus on spot calling, decoding, and segmentation — but whether the **sample itself** is any good is still judged by eye. SQI replaces that with a single, reproducible number.

## What it measures

For each cell, SQI computes a signal-to-noise ratio:

```
SQI(cell) = spot_density(cell-proximal) / spot_density(cell-distal)
```

- **SQI >> 1** → RNA concentrates near cells → good sample
- **SQI ≈ 1** → RNA diffusely spread → likely degraded

Per-cell scores are aggregated to a FOV-level summary. A built-in null model (uniform pseudo-spots) validates that the metric is capturing real signal, not noise.

**Scope:** designed for sparse-to-moderate density tissues. In very dense tissues (e.g. mouse brain), foreground/background separation breaks down.

## Pipeline

DAPI image → Cellpose segmentation → FG/BG mask construction → Spotiflow spot detection → per-spot quality scoring → SQI computation → sanity check

Everything runs from a single script. Intermediate results are cached.

## Quickstart

```bash
conda create -n sqi python=3.11 -y && conda activate sqi
pip install -e .

# GPU PyTorch (CUDA 12.1) — after pip install to avoid conflicts
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --solver=libmamba
```

```bash
python scripts/run_sqi_from_fov_zarr.py \
  --fov_zarr  /path/to/fov.zarr \
  --data_fld  /path/to/parent_folder \
  --cache_root /path/to/cache \
  --out_root   /path/to/output
```

## Outputs

| File | What it is |
|------|------------|
| `sqi_summary.json` | FOV-level median SQI, per-channel breakdown |
| `sqi_per_cell.csv` | Per-cell SQI and FG/BG spot counts |
| `sqi_distribution.png` | log₁₀(SQI) histograms per channel |
| `sqi_sanity_check.png` | Real vs null SQI overlay |

## Project structure

```
sqi/
  io/              — image & spots I/O (tiff, zarr, parquet)
  qc/              — FG/BG masks, SQI metrics, sanity check, plots
  spot_calling/    — Spotiflow backend
  spot_features/   — per-spot feature extraction & quality scoring
segmentation/      — Cellpose backend
scripts/           — run pipelines
configs/           — configuration files
```

## Dependencies

numpy · scipy · scikit-image · matplotlib · tifffile · zarr · cellpose · spotiflow · pandas · pyarrow
