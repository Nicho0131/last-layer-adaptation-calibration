"""
build the model inputs, the target and the normalisation statistics.

  - read the aligned rasters (Copernicus DSM, EGM2008 LiDAR DTM, 8-band feature stack)
  - target delta_h = DTM - DSM, valid where both exist and inside [target_min, target_max]
  - spatial split of Site A into train | gap | val | gap | test strips
  - 9 input channels (aspect -> sin / cos), per-channel mean / std on the TRAIN strip only
  - stats.json (saved inside every checkpoint too, so the model always uses the same numbers)
  - prepare_site(): the same inputs for a test site, standardised with Site A's statistics
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import rasterio
from affine import Affine


# =============================================================================
# raster I/O
# =============================================================================
def read_raster(path: str, band: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(array float32, valid bool, profile). band=None reads all bands -> [C, H, W].
    valid excludes NaN and the nodata value."""
    with rasterio.open(path) as src:
        arr = (src.read(band) if band is not None else src.read()).astype(np.float32)
        nodata, profile = src.nodata, src.profile
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    return arr, valid, profile


def save_tif(path: str, arr: np.ndarray, profile: dict) -> str:
    """Single-band float32 GeoTIFF on the grid of `profile`, NaN = no data."""
    p = profile.copy()
    p.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def site_band(feat_path: str, band: int, box=None) -> np.ndarray:
    """One band (1-based) of an aligned stack, NaN where invalid, optionally cut to box (y0, x0, y1, x1)."""
    a, ok, _ = read_raster(feat_path, band)
    a = np.where(ok, a, np.nan).astype(np.float32)
    if box is not None:
        y0, x0, y1, x1 = box
        a = a[y0:y1, x0:x1]
    return a


# =============================================================================
# inputs
# =============================================================================
# band order of the 8-band GEE stack: Fused_DSM, Slope, Aspect, TRI, Canopy_Height, LAI, EVI, Building_Height
CHANNELS = ["DSM", "Slope", "Aspect_E", "Aspect_N", "TRI", "Canopy", "LAI", "EVI", "Bldg"]
CANOPY_BAND, BUILDING_BAND = 5, 8          # 1-based bands of the GEE stack


def build_inputs(dsm: np.ndarray, dsm_ok: np.ndarray, feat: np.ndarray, feat_ok: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """8 GEE bands + DSM -> 9 model channels [9, H, W] and their validity masks.
    Aspect (degrees) becomes sin / cos so 359 and 1 degree are close."""
    assert feat.shape[0] == 8, f"Expected length 8 got length {feat.shape[0]}"
    aspect = np.radians(feat[2])
    X = np.stack([dsm, feat[1], np.sin(aspect), np.cos(aspect),
                  feat[3], feat[4], feat[5], feat[6], feat[7]]).astype(np.float32)
    X_ok = np.stack([dsm_ok, feat_ok[1], feat_ok[2], feat_ok[2],
                     feat_ok[3], feat_ok[4], feat_ok[5], feat_ok[6], feat_ok[7]])
    return X, X_ok


def standardize(X: np.ndarray, X_ok: np.ndarray, means, stds) -> np.ndarray:
    """(X - mean) / std per channel; invalid pixels set to 0 (= the training mean)."""
    Xn = (X - np.asarray(means, np.float32)[:, None, None]) / np.asarray(stds, np.float32)[:, None, None]
    Xn[~X_ok] = 0.0
    return Xn.astype(np.float32)


def make_target(dsm, dsm_ok, dtm, dtm_ok, target_min: float, target_max: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """delta_h = DTM - DSM (NaN where either is missing) and valid = both exist and inside the range."""
    both = dsm_ok & dtm_ok
    target = np.where(both, dtm - dsm, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        valid = both & (target >= target_min) & (target <= target_max)
    pct = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    print("delta_h = DTM - DSM percentiles (m):")
    print({p: round(float(v), 2) for p, v in zip(pct, np.nanpercentile(target, pct))})
    print(f"valid target pixels: {valid.mean() * 100:.1f}%  (removed as outliers: {(both & ~valid).sum():,})")
    return target, valid


# =============================================================================
# spatial split (Site A)
# =============================================================================
def split_strips(n: int, val_frac: float = 0.15, test_frac: float = 0.15, gap_px: int = 128):
    """(start, end) of train, val, test along one axis, with gap_px unused between them."""
    v0 = int(n * (1 - val_frac - test_frac))
    t0 = int(n * (1 - test_frac))
    return (0, v0 - gap_px), (v0, t0 - gap_px), (t0, n)


def split_boxes(H: int, W: int, val_frac: float = 0.15, test_frac: float = 0.15, gap_px: int = 128):
    """Train / val / test boxes (y0, x0, y1, x1): strips along the longer axis."""
    if W >= H:                                   # strips side by side (split along x)
        (a0, a1), (b0, b1), (c0, c1) = split_strips(W, val_frac, test_frac, gap_px)
        return (0, a0, H, a1), (0, b0, H, b1), (0, c0, H, c1)
    (a0, a1), (b0, b1), (c0, c1) = split_strips(H, val_frac, test_frac, gap_px)
    return (a0, 0, a1, W), (b0, 0, b1, W), (c0, 0, c1, W)


# =============================================================================
# statistics
# =============================================================================
def compute_stats(X: np.ndarray, X_ok: np.ndarray, train_box) -> tuple[list[float], list[float]]:
    """Per-channel mean / std over valid pixels inside the training box only."""
    y0, x0, y1, x1 = train_box
    means, stds = [], []
    for c in range(X.shape[0]):
        v = X[c, y0:y1, x0:x1][X_ok[c, y0:y1, x0:x1]].astype(np.float64)
        if not v.size:
            raise ValueError("train box is empty i.e. no data")
        m, s = v.mean(), v.std()
        means.append(float(m))
        stds.append(float(s) if s > 1e-6 else 1.0)          # constant channel guard
    return means, stds


def save_stats(stats: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path


def load_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def means_stds(stats: dict) -> tuple[list[float], list[float]]:
    """Means / stds from a stats dict (older files used the key 'global_std')."""
    return stats["global_means"], stats.get("global_stds", stats.get("global_std"))


# =============================================================================
# Site A: everything training needs (cells 19-21)
# =============================================================================
def prepare_site_a(cfg: dict, write_stats: bool = True) -> dict:
    """Read Site A, build target / split / inputs / statistics.

    Returns a dict with what the notebook kept as globals:
      Xn [9, H, W] (X_norm), target [H, W] (NaN -> 0, TARGET_t), valid [H, W] (VALID_t),
      dsm (RAW_DSM), dsm_ok, features [8, H, W], features_valid, profile (DSM_PROFILE),
      H, W, pixel_m, train_box, val_box, test_box, stats.
    """
    p, d = cfg["paths"], cfg["data"]
    dsm, dsm_ok, dsm_prof = read_raster(p["copernicus_dsm"], 1)
    dtm, dtm_ok, _ = read_raster(p["dtm_30m_egm2008"], 1)
    feat, feat_ok, _ = read_raster(p["gee_features_aligned"])
    print(dsm.shape, dtm.shape, feat.shape)
    assert dsm.shape == dtm.shape == feat.shape[1:], "All arrays must have the same bounding areas"

    H, W = dsm.shape
    pixel_m = abs(dsm_prof["transform"].a)
    print(f"DSM Pixel Size: {pixel_m} meters, ({H * pixel_m / 1000:.1f} x {W * pixel_m / 1000:.1f} km)")

    target, valid = make_target(dsm, dsm_ok, dtm, dtm_ok, d["target_min"], d["target_max"])

    s = d["split"]
    train_box, val_box, test_box = split_boxes(H, W, s["val_frac"], s["test_frac"], s["gap_px"])
    for name, (y0, x0, y1, x1) in (("train", train_box), ("val", val_box), ("test", test_box)):
        print(f"{name:5s} box {(y0, x0, y1, x1)} | {(y1 - y0) * pixel_m / 1000:.1f} x "
              f"{(x1 - x0) * pixel_m / 1000:.1f} km | valid target {valid[y0:y1, x0:x1].mean() * 100:.1f}%")

    X, X_ok = build_inputs(dsm, dsm_ok, feat, feat_ok)
    means, stds = compute_stats(X, X_ok, train_box)
    Xn = standardize(X, X_ok, means, stds)
    del X
    for name, m, sd in zip(CHANNELS, means, stds):
        print(f"{name:9s}: mean {m:10.3f}, std {sd:9.3f}")

    stats = dict(channels=CHANNELS, global_means=means, global_stds=stds,
                 target_min=d["target_min"], target_max=d["target_max"], pixel_m=pixel_m,
                 val_box=val_box, train_box=train_box, test_box=test_box, dsm_hw=(H, W))
    if write_stats:
        save_stats(stats, p["stats"])
        print("saved", p["stats"])

    return dict(Xn=Xn, target=np.nan_to_num(target, nan=0.0), valid=valid,
                dsm=dsm, dsm_ok=dsm_ok, features=feat, features_valid=feat_ok, profile=dsm_prof,
                H=H, W=W, pixel_m=pixel_m, train_box=train_box, val_box=val_box, test_box=test_box,
                stats=stats)


# =============================================================================
# test sites (cell 106, prepare_site)
# =============================================================================
def prepare_site(site: dict, stats: dict, model_pixel_m: float, box=None,
                 canopy_band: int = CANOPY_BAND, building_band: int = BUILDING_BAND) -> dict:
    """Inputs and target for a test site, standardised with Site A's statistics (never the site's own).

    site: a sites: entry of config.yaml (needs feat, dtm, dtm_odn).
    box:  optional (y0, x0, y1, x1) in pixels; everything is then cut to that box, with its own transform.
      1a lidar in EGM2008 (converted with shift_to_EGM2008 if the file does not exist yet)
      1b stack + lidar on one grid; DSM = band 1 of the stack
      1c optional crop to `box`
      1d inputs: build_inputs, standardize with Site A's statistics
      1e target delta_h = lidar - DSM; valid = both exist and inside the training range; canopy, buildings
    """
    if not os.path.exists(site["dtm"]):                                          # 1a
        from .prepare_rasters import shift_to_EGM2008
        print("lidar not in EGM2008 yet -> shift_to_EGM2008 (run setup_pyproj first)")
        shift_to_EGM2008(site["dtm_odn"], site["dtm"])

    feat, feat_ok, prof = read_raster(site["feat"])                              # 1b
    assert feat.shape[0] == 8, f"expected 8 feature bands, got {feat.shape[0]}"
    dtm, dtm_ok, prof_t = read_raster(site["dtm"], 1)
    assert dtm.shape == feat.shape[1:] and prof_t["transform"].almost_equals(prof["transform"]), \
        "lidar is not on the stack's grid (run prepare_rasters for this site first)"
    T = prof["transform"]

    if box is not None:                                                          # 1c
        y0, x0, y1, x1 = box
        feat, feat_ok = feat[:, y0:y1, x0:x1], feat_ok[:, y0:y1, x0:x1]
        dtm, dtm_ok = dtm[y0:y1, x0:x1], dtm_ok[y0:y1, x0:x1]
        T = T * Affine.translation(x0, y0)

    dsm, dsm_ok = feat[0], feat_ok[0]
    pixel_m = abs(T.a)
    assert abs(pixel_m / model_pixel_m - 1) < 0.05, \
        f"pixel {pixel_m:.2f} m, model trained at {model_pixel_m:.2f} m"

    X, X_ok = build_inputs(dsm, dsm_ok, feat, feat_ok)                           # 1d
    Xn = standardize(X, X_ok, *means_stds(stats))
    del X

    target = (dtm - dsm).astype(np.float32)                                      # 1e
    with np.errstate(invalid="ignore"):
        valid = dsm_ok & dtm_ok & (target >= stats["target_min"]) & (target <= stats["target_max"])
    kc, kb = canopy_band - 1, building_band - 1
    canopy = np.where(feat_ok[kc], feat[kc], 0.0).clip(min=0).astype(np.float32)
    bldg = np.where(feat_ok[kb], feat[kb], 0.0).clip(min=0).astype(np.float32)

    H, W = dsm.shape
    print(f"site: {W} x {H} px at {pixel_m:g} m ({W * pixel_m / 1000:.1f} x {H * pixel_m / 1000:.1f} km)"
          f"{' | box ' + str(box) if box is not None else ''} | valid lidar {valid.mean():.1%}")
    return dict(Xn=Xn, dsm=dsm, dsm_ok=dsm_ok, target=np.where(valid, target, 0.0).astype(np.float32),
                valid=valid, canopy=canopy, bldg=bldg, transform=T, crs=prof["crs"], prof=prof,
                H=H, W=W, pixel_m=pixel_m, stats=stats)


# =============================================================================
# run: Site A -> stats.json
# =============================================================================
def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description="Build Site A inputs and write stats.json.")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    A = prepare_site_a(cfg, write_stats=True)
    print(f"Site A ready: inputs {A['Xn'].shape} | valid target {A['valid'].mean():.1%}")


if __name__ == "__main__":
    main()