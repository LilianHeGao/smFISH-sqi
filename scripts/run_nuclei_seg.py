import argparse
import json
from pathlib import Path

from sqi.io.image_io import read_dapi_from_conv_zarr, read_tif_2d, write_labels_tif
from segmentation.cellpose_backend import CellposeBackend, CellposeNucleiConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapi", required=True, help="Path to DAPI (.tif or Conv .zarr)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--gpu", type=int, default=-1, help="-1 auto, 0 cpu, 1 gpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = None if args.gpu < 0 else bool(args.gpu)

    cfg = CellposeNucleiConfig(
        model_type="nuclei",
        use_gpu=use_gpu,
        diameter=args.diameter,
        channels=(0, 0),
    )

    # --- Load DAPI ---
    if args.dapi.endswith(".zarr"):
        img = read_dapi_from_conv_zarr(args.dapi, channel=-1)
    else:
        img = read_tif_2d(args.dapi)

    # --- Run Cellpose ---
    backend = CellposeBackend(cfg)
    labels, meta = backend.segment_nuclei(img)

    # --- Save outputs ---
    write_labels_tif(str(out_dir / "nuclei_labels.tif"), labels)
    (out_dir / "nuclei_meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved:", out_dir / "nuclei_labels.tif")
    print("Meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
