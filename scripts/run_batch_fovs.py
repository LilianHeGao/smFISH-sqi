"""
Batch runner: auto-discover FOV zarrs, randomly select N,
run the SQI pipeline on each, and generate tissue overview(s)
with all selected FOVs highlighted.

Called by run_batch_fovs.bat.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_multichannel_from_conv_zarr
from sqi.qc.mosaic_coords import (
    build_mosaic_and_coords,
    build_fov_anchor_index,
    lookup_fov_anchor,
    fov_id_from_zarr_path,
    MosaicBuildConfig,
    mosaic_cache_paths,
)
from sqi.qc.qc_plots import plot_tissue_overview


def discover_fov_zarrs(data_fld: str) -> list[str]:
    pattern = os.path.join(data_fld, "Conv_zscan*.zarr")
    return sorted(glob.glob(pattern))


def normalize_set_name(token: str) -> str:
    """Accept '1' or 'set1' and normalize to folder-style 'set1'."""
    t = str(token).strip()
    if not t:
        raise ValueError("Empty set token")

    if t.isdigit():
        return f"set{int(t)}"

    tl = t.lower()
    if tl.startswith("set") and len(t) > 3 and t[3:].isdigit():
        return f"set{int(t[3:])}"

    return t


def build_fov_entries(data_fld: str, sets: list[str] | None) -> list[dict]:
    """
    Build a flat list of FOV entries.

    Each entry has keys:
      set_name, data_fld, fov_zarr, fov_id
    """
    entries = []

    if not sets:
        zarrs = discover_fov_zarrs(data_fld)
        set_name = os.path.basename(os.path.normpath(data_fld)) or "data"
        print(f"[DISCOVER] {set_name}: {len(zarrs)} FOV(s)")
        for zarr in zarrs:
            entries.append(
                {
                    "set_name": set_name,
                    "data_fld": data_fld,
                    "fov_zarr": zarr,
                    "fov_id": fov_id_from_zarr_path(zarr),
                }
            )
        return entries

    # Multi-set mode: --data_fld is parent, each set resolves to <data_fld>/<set_name>
    normalized_sets = []
    seen = set()
    for token in sets:
        s = normalize_set_name(token)
        if s not in seen:
            normalized_sets.append(s)
            seen.add(s)

    for set_name in normalized_sets:
        set_dir = os.path.join(data_fld, set_name)
        if not os.path.isdir(set_dir):
            print(f"[WARN] Set folder not found, skipping: {set_dir}")
            continue

        zarrs = discover_fov_zarrs(set_dir)
        print(f"[DISCOVER] {set_name}: {len(zarrs)} FOV(s)")
        for zarr in zarrs:
            entries.append(
                {
                    "set_name": set_name,
                    "data_fld": set_dir,
                    "fov_zarr": zarr,
                    "fov_id": fov_id_from_zarr_path(zarr),
                }
            )

    return entries


def get_set_roots(
    cache_root: str,
    out_root: str,
    set_name: str,
    use_set_subdirs: bool,
) -> tuple[str, str]:
    if use_set_subdirs:
        return os.path.join(cache_root, set_name), os.path.join(out_root, set_name)
    return cache_root, out_root


def run_label(set_name: str, fov_id: str, use_set_subdirs: bool) -> str:
    if use_set_subdirs:
        return f"{set_name}/{fov_id}"
    return fov_id


def get_fov_shape_from_zarr(zarr_path: str) -> tuple[int, int]:
    """Read one zarr to get the FOV shape at full resolution."""
    im = read_multichannel_from_conv_zarr(zarr_path)
    # im shape: (C, Z, Y, X) or (C, Y, X)
    h, w = im.shape[-2], im.shape[-1]
    return (h, w)


def compute_fov_bbox_mosaic(
    anchor_xy: tuple[float, float],
    fov_shape_hw: tuple[int, int],
    resc: int,
) -> tuple[int, int, int, int]:
    """Compute (r0, c0, r1, c1) in mosaic coords from center anchor."""
    a0, a1 = anchor_xy
    hf = fov_shape_hw[0] // (2 * resc)
    wf = fov_shape_hw[1] // (2 * resc)
    r0 = int(round(a0 - hf))
    c0 = int(round(a1 - wf))
    r1 = int(round(a0 + hf))
    c1 = int(round(a1 + wf))
    return (r0, c0, r1, c1)


def _load_log10_sqi_from_csv(csv_path: Path) -> np.ndarray:
    """Load log10(SQI) values from sqi_per_cell.csv."""
    if not csv_path.exists():
        return np.array([], dtype=np.float32)

    vals = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sqi = float(row.get("sqi", "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(sqi) and sqi > 0:
                vals.append(np.log10(sqi))

    return np.array(vals, dtype=np.float32)


def _load_log10_sanity_from_json(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load log10(real/null SQI) values saved by run_sqi_from_fov_zarr.py."""
    if not json_path.exists():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    real = np.array(payload.get("real_sqi", []), dtype=float)
    null = np.array(payload.get("null_sqi", []), dtype=float)

    real = real[np.isfinite(real) & (real > 0)]
    null = null[np.isfinite(null) & (null > 0)]

    return np.log10(real).astype(np.float32), np.log10(null).astype(np.float32)


def _shared_bins(arrays: list[np.ndarray], n_bins: int = 50) -> np.ndarray | None:
    arrays = [a for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        return None

    vals = np.concatenate(arrays)
    if len(vals) == 0:
        return None

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        vmin -= 0.5
        vmax += 0.5

    pad = 0.05 * (vmax - vmin)
    return np.linspace(vmin - pad, vmax + pad, n_bins + 1)


def _density_stack(arrays: list[np.ndarray], bins: np.ndarray) -> np.ndarray | None:
    rows = []
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        hist, _ = np.histogram(arr, bins=bins, density=True)
        rows.append(hist)
    if not rows:
        return None
    return np.vstack(rows)


def _mann_whitney_auc(real_log: np.ndarray, null_log: np.ndarray) -> float:
    if len(real_log) < 5 or len(null_log) < 5:
        return float("nan")
    u = 0.0
    for r in real_log:
        u += np.sum(r > null_log) + 0.5 * np.sum(r == null_log)
    return float(u / (len(real_log) * len(null_log)))


def _save_set_distribution_avg(
    out_path: Path,
    set_name: str,
    fov_logs: list[np.ndarray],
) -> bool:
    bins = _shared_bins(fov_logs, n_bins=50)
    if bins is None:
        return False

    stack = _density_stack(fov_logs, bins)
    if stack is None:
        return False

    import matplotlib.pyplot as plt

    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_d = stack.mean(axis=0)
    std_d = stack.std(axis=0)
    pooled = np.concatenate(fov_logs) if fov_logs else np.array([], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(pooled) > 0:
        ax.hist(
            pooled,
            bins=bins,
            density=True,
            color="0.85",
            alpha=0.5,
            label="pooled",
        )
    ax.plot(centers, mean_d, color="tab:blue", lw=2, label="mean density")
    ax.fill_between(
        centers,
        np.clip(mean_d - std_d, 0, None),
        mean_d + std_d,
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )
    ax.axvline(0, linestyle="--", color="k", linewidth=1)
    ax.set_xlabel("log10(SQI)")
    ax.set_ylabel("Density")
    ax.set_title(f"SQI distribution avg — {set_name} ({len(fov_logs)} FOVs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return True


def _save_set_sanity_avg(
    out_path: Path,
    set_name: str,
    real_logs: list[np.ndarray],
    null_logs: list[np.ndarray],
) -> bool:
    bins = _shared_bins(real_logs + null_logs, n_bins=50)
    if bins is None:
        return False

    stack_real = _density_stack(real_logs, bins)
    stack_null = _density_stack(null_logs, bins)
    if stack_real is None or stack_null is None:
        return False

    import matplotlib.pyplot as plt

    centers = 0.5 * (bins[:-1] + bins[1:])
    real_mean, real_std = stack_real.mean(axis=0), stack_real.std(axis=0)
    null_mean, null_std = stack_null.mean(axis=0), stack_null.std(axis=0)

    pooled_real = np.concatenate(real_logs)
    pooled_null = np.concatenate(null_logs)
    auc = _mann_whitney_auc(pooled_real, pooled_null)
    auc_txt = f"{auc:.3f}" if np.isfinite(auc) else "N/A"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, null_mean, color="tab:gray", lw=2, label="null mean density")
    ax.fill_between(
        centers,
        np.clip(null_mean - null_std, 0, None),
        null_mean + null_std,
        color="tab:gray",
        alpha=0.2,
        label="null ±1 std",
    )
    ax.plot(centers, real_mean, color="tab:blue", lw=2, label="real mean density")
    ax.fill_between(
        centers,
        np.clip(real_mean - real_std, 0, None),
        real_mean + real_std,
        color="tab:blue",
        alpha=0.2,
        label="real ±1 std",
    )
    ax.axvline(0, linestyle="--", color="k", linewidth=1)
    ax.set_xlabel("log10(SQI)")
    ax.set_ylabel("Density")
    ax.set_title(f"SQI sanity avg — {set_name} ({len(real_logs)} FOVs), AUC={auc_txt}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch SQI: auto-discover FOVs, run pipeline, generate tissue overview",
    )
    parser.add_argument(
        "--data_fld",
        required=True,
        help="Single-set mode: folder containing Conv_zscan*.zarr. "
             "Multi-set mode (with --set): parent folder containing set1/set2/...",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        nargs="+",
        default=None,
        help="Optional set selector, e.g. '--set 1 2 3' or '--set set1 set2'. "
             "If provided, FOVs are pooled from selected sets.",
    )
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument(
        "--n_fovs",
        type=int,
        default=10,
        help="Number of FOVs to randomly select from pooled candidates (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resc",
        type=int,
        default=4,
        help="Mosaic rescale factor (default: 4)",
    )
    parser.add_argument(
        "--rot_k",
        type=int,
        default=2,
        help="Tile rotation before stitching: np.rot90 k (0-3, default: 2). "
             "Use test_mosaic_orientation.py to pick the right value.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    use_set_subdirs = bool(args.sets)

    # --------------------------------------------------
    # 1. Discover and select FOVs
    # --------------------------------------------------
    all_entries = build_fov_entries(args.data_fld, args.sets)
    if not all_entries:
        if args.sets:
            print(f"[ERROR] No Conv_zscan*.zarr found under selected sets in {args.data_fld}")
        else:
            print(f"[ERROR] No Conv_zscan*.zarr found in {args.data_fld}")
        sys.exit(1)

    n_pick = min(args.n_fovs, len(all_entries))
    selected = random.sample(all_entries, n_pick)

    print("=" * 60)
    print(f"Selected {n_pick} / {len(all_entries)} FOVs:")
    for entry in selected:
        label = run_label(entry["set_name"], entry["fov_id"], use_set_subdirs)
        print(f"  {label}  ({os.path.basename(entry['fov_zarr'])})")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Run pipeline on each FOV
    # --------------------------------------------------
    failed = []
    successful_entries = []
    for i, entry in enumerate(selected):
        set_name = entry["set_name"]
        data_fld = entry["data_fld"]
        fov_zarr = entry["fov_zarr"]
        fov_id = entry["fov_id"]
        label = run_label(set_name, fov_id, use_set_subdirs)

        print(f"\n[{i + 1}/{n_pick}] Processing FOV {label} ...")

        set_cache_root, set_out_root = get_set_roots(
            args.cache_root, args.out_root, set_name, use_set_subdirs
        )
        Path(set_out_root).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/run_sqi_from_fov_zarr.py",
            "--fov_zarr",
            fov_zarr,
            "--data_fld",
            data_fld,
            "--cache_root",
            set_cache_root,
            "--out_root",
            set_out_root,
            "--resc",
            str(args.resc),
            "--rot_k",
            str(args.rot_k),
        ]
        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] FOV {label} failed!")
            failed.append(label)
        else:
            print(f"[OK] FOV {label} done.")
            successful_entries.append(entry)

    # --------------------------------------------------
    # 3. Generate tissue overview(s)
    # --------------------------------------------------
    print("\n[POST] Generating tissue overview(s) ...")

    import matplotlib
    matplotlib.use("Agg")

    selected_by_set = defaultdict(list)
    for entry in selected:
        selected_by_set[entry["set_name"]].append(entry)

    overview_paths = {}
    for set_name, set_entries in selected_by_set.items():
        data_fld = set_entries[0]["data_fld"]
        set_cache_root, set_out_root = get_set_roots(
            args.cache_root, args.out_root, set_name, use_set_subdirs
        )
        Path(set_out_root).mkdir(parents=True, exist_ok=True)

        mosaic_cfg = MosaicBuildConfig(resc=args.resc, rot_k=args.rot_k)
        _, fls_, xs, ys = build_mosaic_and_coords(
            data_fld,
            mosaic_cfg,
            cache_root=set_cache_root,
            cache=True,
        )
        fov_index = build_fov_anchor_index(fls_, xs, ys)

        mosaic_tif, _ = mosaic_cache_paths(data_fld, mosaic_cfg, set_cache_root)
        mosaic_img = tiff.imread(mosaic_tif).astype(np.float32)

        # Get FOV shape from one selected zarr in this set
        fov_shape = get_fov_shape_from_zarr(set_entries[0]["fov_zarr"])

        bboxes = []
        labels = []
        for entry in set_entries:
            fov_id = entry["fov_id"]
            label = run_label(set_name, fov_id, use_set_subdirs)
            try:
                anchor = lookup_fov_anchor(fov_index, fov_id)
                bbox = compute_fov_bbox_mosaic(anchor, fov_shape, args.resc)
                bboxes.append(bbox)
                labels.append(fov_id)
            except KeyError as e:
                print(f"  [WARN] Skipping FOV {label} in overview: {e}")

        dataset_name = set_name if use_set_subdirs else os.path.basename(data_fld)
        out_overview = Path(set_out_root) / "tissue_overview.png"
        plot_tissue_overview(
            mosaic_img,
            bboxes,
            labels,
            title=f"Tissue overview — {dataset_name} ({len(set_entries)} FOVs)",
            out_path=str(out_overview),
        )
        overview_paths[set_name] = out_overview

    # --------------------------------------------------
    # 4. Generate set-level avg SQI plots
    # --------------------------------------------------
    print("\n[POST] Generating set-level avg SQI plots (requires >=3 successful FOVs/set) ...")

    avg_distribution_paths = {}
    avg_sanity_paths = {}
    success_by_set = defaultdict(list)
    for entry in successful_entries:
        success_by_set[entry["set_name"]].append(entry)

    for set_name, set_entries in success_by_set.items():
        if len(set_entries) < 3:
            print(f"  [SKIP] {set_name}: only {len(set_entries)} successful FOV(s) (<3)")
            continue

        _, set_out_root = get_set_roots(
            args.cache_root, args.out_root, set_name, use_set_subdirs
        )
        set_out_root = Path(set_out_root)

        fov_logs = []
        real_logs = []
        null_logs = []

        for entry in set_entries:
            fov_out = set_out_root / entry["fov_id"]

            sqi_csv = fov_out / "sqi_per_cell.csv"
            sqi_log = _load_log10_sqi_from_csv(sqi_csv)
            if len(sqi_log) > 0:
                fov_logs.append(sqi_log)

            sanity_json = fov_out / "sqi_sanity_values.json"
            real_log, null_log = _load_log10_sanity_from_json(sanity_json)
            if len(real_log) > 0 and len(null_log) > 0:
                real_logs.append(real_log)
                null_logs.append(null_log)

        if len(fov_logs) >= 3:
            dist_avg_out = set_out_root / "sqi_distribution_avg.png"
            if _save_set_distribution_avg(dist_avg_out, set_name, fov_logs):
                avg_distribution_paths[set_name] = dist_avg_out
        else:
            print(f"  [SKIP] {set_name}: insufficient SQI distributions for avg plot")

        if len(real_logs) >= 3 and len(null_logs) >= 3:
            sanity_avg_out = set_out_root / "sqi_sanity_check_avg.png"
            if _save_set_sanity_avg(sanity_avg_out, set_name, real_logs, null_logs):
                avg_sanity_paths[set_name] = sanity_avg_out
        else:
            print(f"  [SKIP] {set_name}: insufficient sanity values for avg plot")

    # --------------------------------------------------
    # 5. Summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Batch complete: {n_pick - len(failed)}/{n_pick} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Output root: {out_root}")
    if use_set_subdirs:
        print("  Tissue overviews:")
        for set_name in sorted(overview_paths):
            print(f"    {set_name}: {overview_paths[set_name]}")
    else:
        only_set = next(iter(overview_paths))
        print(f"  Tissue overview: {overview_paths[only_set]}")
    if avg_distribution_paths:
        print("  Set SQI distribution avg plots:")
        for set_name in sorted(avg_distribution_paths):
            print(f"    {set_name}: {avg_distribution_paths[set_name]}")
    if avg_sanity_paths:
        print("  Set SQI sanity avg plots:")
        for set_name in sorted(avg_sanity_paths):
            print(f"    {set_name}: {avg_sanity_paths[set_name]}")
    print("=" * 60)

    # Collect AUCs from summaries
    aucs = {}
    for entry in successful_entries:
        set_name = entry["set_name"]
        fov_id = entry["fov_id"]
        label = run_label(set_name, fov_id, use_set_subdirs)

        _, set_out_root = get_set_roots(
            args.cache_root, args.out_root, set_name, use_set_subdirs
        )
        summary_path = Path(set_out_root) / fov_id / "sqi_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            aucs[label] = data.get("sanity_auc")

    if aucs:
        print("\nSanity-check AUCs:")
        for label in sorted(aucs):
            auc = aucs[label]
            status = "OK" if auc is not None and auc >= 0.6 else "WARN"
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            print(f"  FOV {label}: AUC={auc_str}  [{status}]")


if __name__ == "__main__":
    main()
