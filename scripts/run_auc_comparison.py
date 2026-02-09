"""
Randomly select FOVs from multiple conditions, run SQI pipeline,
and produce a comparison plot of sanity-check AUCs.

Called by run_auc_comparison.bat — designed to run on the Windows remote desktop.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def discover_fov_zarrs(data_fld: str) -> list[str]:
    """Find all Conv_zscan*.zarr folders in data_fld."""
    pattern = os.path.join(data_fld, "Conv_zscan*.zarr")
    return sorted(glob.glob(pattern))


def run_sqi_pipeline(fov_zarr, data_fld, cache_root, out_root, force=False):
    """Run run_sqi_from_fov_zarr.py as a subprocess. Returns True on success."""
    cmd = [
        sys.executable, "scripts/run_sqi_from_fov_zarr.py",
        "--fov_zarr", fov_zarr,
        "--data_fld", data_fld,
        "--cache_root", cache_root,
        "--out_root", out_root,
    ]
    if force:
        cmd.append("--force")

    print(f"  >> {os.path.basename(fov_zarr)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def collect_auc(out_root, fov_zarr):
    """Read sanity_auc from sqi_summary.json for a given FOV."""
    base = os.path.basename(fov_zarr)
    fov_id = base.split("_")[-1].split(".")[0]
    summary_path = os.path.join(out_root, fov_id, "sqi_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("sanity_auc")


def plot_auc_comparison(results: dict[str, list[float]], out_path: str):
    """
    Box + strip plot of AUCs per condition.

    results: {condition_label: [auc1, auc2, ...]}
    """
    labels = list(results.keys())
    data = [results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(labels)), 4))

    # Box plot
    bp = ax.boxplot(data, positions=range(len(labels)), widths=0.5,
                    patch_artist=True, showfliers=False)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.4)

    # Strip (jittered individual points)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=colors[i % len(colors)], s=40, edgecolors="k",
            linewidths=0.5, zorder=3,
        )

    # Threshold line
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1, label="threshold = 0.6")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sanity-check AUC")
    ax.set_title("SQI sanity-check AUC across conditions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SQI on random FOVs from multiple conditions and compare AUCs",
    )
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help="Condition specs as LABEL:DATA_FLD (e.g. 'mouse_6OHDA:\\\\server\\path')",
    )
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_root", required=True,
                        help="Base output dir (sub-dirs per condition)")
    parser.add_argument("--n_fovs", type=int, default=10,
                        help="Number of FOVs to randomly select per condition (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    out_base = Path(args.out_root)
    out_base.mkdir(parents=True, exist_ok=True)

    auc_results = {}  # {label: [auc, ...]}

    for cond_spec in args.conditions:
        label, data_fld = cond_spec.split(":", 1)
        print("=" * 60)
        print(f"Condition: {label}")
        print(f"  data_fld: {data_fld}")

        all_zarrs = discover_fov_zarrs(data_fld)
        if len(all_zarrs) == 0:
            print(f"  [WARN] No Conv_zscan*.zarr found in {data_fld}")
            auc_results[label] = []
            continue

        n_pick = min(args.n_fovs, len(all_zarrs))
        selected = random.sample(all_zarrs, n_pick)
        print(f"  Found {len(all_zarrs)} FOVs, selected {n_pick}")

        cond_out = str(out_base / label)
        aucs = []
        for fov_zarr in selected:
            ok = run_sqi_pipeline(
                fov_zarr, data_fld, args.cache_root, cond_out, force=args.force,
            )
            if ok:
                auc = collect_auc(cond_out, fov_zarr)
                if auc is not None:
                    aucs.append(auc)
                    print(f"     AUC = {auc:.3f}")
                else:
                    print(f"     AUC = N/A")
            else:
                print(f"     [ERROR] pipeline failed")

        auc_results[label] = aucs
        print(f"  Collected {len(aucs)} AUCs: "
              f"median={np.median(aucs):.3f}" if aucs else "  No AUCs collected")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("AUC Summary")
    print("=" * 60)
    for label, aucs in auc_results.items():
        if aucs:
            arr = np.array(aucs)
            print(f"  {label:30s}  n={len(aucs):2d}  "
                  f"median={np.median(arr):.3f}  "
                  f"mean={np.mean(arr):.3f}  "
                  f"min={np.min(arr):.3f}  max={np.max(arr):.3f}")
        else:
            print(f"  {label:30s}  n= 0  (no data)")

    # --- Plot ---
    plot_path = str(out_base / "auc_comparison.png")
    non_empty = {k: v for k, v in auc_results.items() if v}
    if non_empty:
        plot_auc_comparison(non_empty, plot_path)

    # --- Save raw numbers ---
    json_path = str(out_base / "auc_comparison.json")
    with open(json_path, "w") as f:
        json.dump(auc_results, f, indent=2)
    print(f"[JSON] Saved: {json_path}")


if __name__ == "__main__":
    main()
