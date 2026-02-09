# sqi/qc/mosaic_coords.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import tifffile

from sqi.io.image_io import read_multichannel_from_conv_zarr

@dataclass
class MosaicBuildConfig:
    resc: int = 4
    icol: int = 1
    frame: int = 20          # or 'all'
    rescz: int = 2
    um_per_pix_native: float = 0.1083333
    force: bool = False

def compose_mosaic(ims,xs_um,ys_um,ims_c=None,um_per_pix=0.108333,rot = 0,return_coords=False):
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    sx,sy = szs[-2],szs[-1]
    ### Apply rotation:
    theta=-np.deg2rad(rot)
    xs_um_ = np.array(xs_um)*np.cos(theta)-np.array(ys_um)*np.sin(theta)
    ys_um_ = np.array(ys_um)*np.cos(theta)+np.array(xs_um)*np.sin(theta)
    ### Calculate per pixel
    xs_pix = np.array(xs_um_)/um_per_pix
    xs_pix = np.array(xs_pix-np.min(xs_pix),dtype=int)
    ys_pix = np.array(ys_um_)/um_per_pix
    ys_pix = np.array(ys_pix-np.min(ys_pix),dtype=int)
    sx_big = np.max(xs_pix)+sx+1
    sy_big = np.max(ys_pix)+sy+1
    dim = [sx_big,sy_big]
    if len(szs)==3:
        dim = [szs[0],sx_big,sy_big]

    if ims_c is None:
        if len(ims)>25:
            try:
                ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5,0.65,0.75,0.9])
            except:
                imc_c = np.median(ims,axis=0)
        else:
            ims_c = np.median(ims,axis=0)

    im_big = np.zeros(dim,dtype = dtype)
    sh_ = np.nan
    for i,(im_,x_,y_) in enumerate(zip(ims,xs_pix,ys_pix)):
        if ims_c is not None:
            if len(ims_c)==2:
                im_coef,im_inters = np.array(ims_c,dtype = 'float32')
                im__=(np.array(im_,dtype = 'float32')-im_inters)/im_coef
            else:
                ims_c_ = np.array(ims_c,dtype = 'float32')
                im__=np.array(im_,dtype = 'float32')/ims_c_*np.median(ims_c_)
        else:
            im__=np.array(im_,dtype = 'float32')
        im__ = np.array(im__,dtype = dtype)
        im_big[...,x_:x_+sx,y_:y_+sy]=im__
        sh_ = im__.shape
    if return_coords:
        return im_big,xs_pix+sh_[-2]/2,ys_pix+sh_[-1]/2
    return im_big

def mosaic_cache_paths(
    data_fld: str,
    cfg: MosaicBuildConfig,
    cache_root: str,
) -> Tuple[str, str]:
    """
    Returns:
      mosaic_tif_path, coords_npz_path
    Saved under cache_root/_mosaic/, NOT alongside data.
    """
    mosaic_dir = os.path.join(cache_root, "_mosaic")
    os.makedirs(mosaic_dir, exist_ok=True)

    base = f"{os.path.basename(data_fld)}_frame{cfg.frame}_col{cfg.icol}_resc{cfg.resc}_rescz{cfg.rescz}"
    mosaic_tif = os.path.join(mosaic_dir, base + ".tiff")
    coords_npz = os.path.join(mosaic_dir, base + "_coords.npz")
    return mosaic_tif, coords_npz


def build_mosaic_and_coords(
    data_fld: str,
    cfg: MosaicBuildConfig,
    *,
    cache_root: str,
    cache: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

    """
    Build (or load cached) mosaic and FOV placement coordinates.

    Returns:
      mosaic_img: 2D float32
      fls_: list of .zarr paths (aligned with xs, ys)
      xs, ys: arrays of placement coordinates returned by compose_mosaic(return_coords=True)
    """
    mosaic_tif, coords_npz = mosaic_cache_paths(data_fld, cfg, cache_root)

    # If cached and not forcing rebuild, load coords + mosaic
    if cache and (not cfg.force) and os.path.exists(mosaic_tif) and os.path.exists(coords_npz):
        coords = np.load(coords_npz, allow_pickle=True)
        fls_ = coords["fls_"].tolist()
        xs = coords["xs"].astype(np.float32, copy=False)
        ys = coords["ys"].astype(np.float32, copy=False)
        return None, fls_, xs, ys

    # Otherwise, compute
    fls_ = np.sort(glob.glob(os.path.join(data_fld, "*.zarr"))).tolist()
    ims: List[np.ndarray] = []
    xs_um: List[float] = []
    ys_um: List[float] = []

    for fl in tqdm(fls_, desc="Reading FOVs for mosaic"):
        try:
            im, x, y = read_multichannel_from_conv_zarr(fl, return_pos=True)

            if str(cfg.frame).lower() == "all":
                # (Z,Y,X) after max over z
                tile = np.array(np.max(im[cfg.icol][::cfg.rescz, ::cfg.resc, ::cfg.resc][:, ::-1], axis=0),
                                dtype=np.float32)
            else:
                tile = np.array(im[cfg.icol][cfg.frame, ::cfg.resc, ::cfg.resc][:, ::-1],
                                dtype=np.float32)

            ims.append(tile)
            xs_um.append(x)
            ys_um.append(y)
        except Exception as e:
            print(f"[mosaic] Error processing {fl}: {e}")
            continue

    if not ims:
        raise RuntimeError("No images were successfully processed for mosaic.")

    # Keep your exact transforms (as in your function)
    rotated_ims = [np.rot90(im_, k=2) for im_ in ims]
    tiles = [im_.T[::-1, ::-1] for im_ in rotated_ims]

    um_per_pix = cfg.um_per_pix_native * cfg.resc

    mosaic_img, xs_center_pix, ys_center_pix = compose_mosaic(
		tiles,
		xs_um,
		ys_um,
		ims_c=None,
		um_per_pix=um_per_pix,
		rot=0,
		return_coords=True,
	)

    mosaic_img = mosaic_img.astype(np.float32, copy=False)
    xs = np.array(xs_center_pix, dtype=np.float32)
    ys = np.array(ys_center_pix, dtype=np.float32)

    # Cache
    if cache:
        tifffile.imwrite(mosaic_tif, mosaic_img)
        np.savez(coords_npz, fls_=np.array(fls_, dtype=object), xs=xs, ys=ys)
        print("[mosaic] Saved:", mosaic_tif)
        print("[mosaic] Saved:", coords_npz)

    return mosaic_img, fls_, xs, ys
	
def fov_id_from_zarr_path(zarr_path: str) -> str:
    base = os.path.basename(zarr_path)
    return base.split("_")[-1].split(".")[0]


def build_fov_anchor_index(fls_: List[str], xs: np.ndarray, ys: np.ndarray) -> Dict[str, Tuple[float, float]]:
    if len(fls_) != len(xs) or len(fls_) != len(ys):
        raise ValueError("fls_, xs, ys length mismatch")
    out: Dict[str, Tuple[float, float]] = {}
    for p, x, y in zip(fls_, xs, ys):
        out[fov_id_from_zarr_path(p)] = (float(x), float(y))
    return out
