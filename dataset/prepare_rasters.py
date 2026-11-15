"""
put the LiDAR DTM and the GEE features on the Copernicus 30 m grid, in EGM2008 heights.

  1. merge     Digimap 1 m LiDAR tiles -> one virtual mosaic (.vrt)
  2. align     mosaic -> Copernicus 30 m grid (average of the 1 m pixels)
               GEE feature stack -> Copernicus 30 m grid (bilinear)
  3. datum     LiDAR heights ODN -> EGM2008 (the Copernicus / ATL08 geoid), via OSGM15

"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import ssl
import urllib.request

import numpy as np
import pyproj
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, reproject


# =============================================================================
# 1. merge Digimap tiles
# =============================================================================
def digimap_merge(input_folder: str, output_filepath: str) -> str | None:
    """All raw .tif tiles in input_folder -> one GDAL virtual mosaic (.vrt) in British National Grid.
    A .vrt only references the tiles, so it is instant and takes no disk space."""
    from osgeo import gdal

    print("=" * 50)
    print("STARTING TILE MERGE (VRT)")
    print("=" * 50)

    raw_tiles = [f for f in glob.glob(f"{input_folder}/*.tif")
                 if not f.endswith("_merge.tif") and not f.endswith("_average.tif")]
    if not raw_tiles:
        print(f"ERROR: No valid raw tiles found in {input_folder}")
        return None
    print(f"Building virtual mosaic from {len(raw_tiles)} tiles...")

    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    opts = gdal.BuildVRTOptions(outputSRS="EPSG:27700", VRTNodata=-9999.0, resampleAlg="nearest")
    vrt = gdal.BuildVRT(output_filepath, raw_tiles, options=opts)
    vrt.FlushCache()
    vrt = None                                   # closes and writes the .vrt

    # quick check on a decimated read (does not load the full 1 m grid)
    with rasterio.open(output_filepath) as src:
        print("\nMERGE COMPLETE. VERIFICATION:")
        print(f"  Shape:       {src.width} x {src.height} pixels")
        k = 50
        preview = src.read(1, out_shape=(src.height // k, src.width // k), masked=True)
        valid = preview.compressed()
        valid = valid[~np.isnan(valid)]
        if valid.size:
            print(f"  Min Elev:    {valid.min():.2f}m (approx)")
            print(f"  Max Elev:    {valid.max():.2f}m (approx)")
            print(f"  Valid frac:  {valid.size / preview.size * 100:.1f}% (approx)")
        else:
            print("  CRITICAL ERROR: Mosaic appears blank!")
    print("=" * 50)
    return output_filepath


# =============================================================================
# 2. align to the Copernicus grid
# =============================================================================
def align_digi_to_copernicus(digimap_filepath: str, copernicus_target_path: str,
                             output_resampled_path: str) -> str:
    """1 m LiDAR DTM -> exactly the Copernicus 30 m grid, each 30 m pixel = average of its 1 m pixels."""
    print("=" * 60)
    print("STARTING DIGIMAP TO COPERNICUS ALIGNMENT & AGGREGATION")
    print("=" * 60)

    with rasterio.open(copernicus_target_path) as cop:
        target_profile = cop.profile.copy()
        cop_crs, cop_transform = cop.crs, cop.transform
        cop_w, cop_h, cop_bounds = cop.width, cop.height, cop.bounds
        cop_nodata = cop.nodata if cop.nodata is not None else np.nan

    print("[TARGET COPERNICUS]")
    print(f"  Shape:       {cop_w} x {cop_h} pixels")
    print(f"  CRS:         {cop_crs}")
    print(f"  Bounds:      {cop_bounds}")
    print(f"  NoData Val:  {cop_nodata}")

    with rasterio.open(digimap_filepath) as digi:
        digi_nodata = digi.nodata if digi.nodata is not None else np.nan
        print("\n[SOURCE DIGIMAP]")
        print(f"  Shape:       {digi.width} x {digi.height} pixels")
        print(f"  CRS:         {digi.crs}")
        print(f"  Bounds:      {digi.bounds}")
        print(f"  NoData Val:  {digi_nodata}")

        resampled = np.full((cop_h, cop_w), cop_nodata, dtype=np.float32)
        print(f"\nResampling {digimap_filepath} (1m) -> Copernicus (30m)...")
        reproject(source=rasterio.band(digi, 1), destination=resampled,
                  src_transform=digi.transform, src_crs=digi.crs, src_nodata=digi_nodata,
                  dst_transform=cop_transform, dst_crs=cop_crs, dst_nodata=cop_nodata,
                  resampling=Resampling.average)

    target_profile.update(dtype=rasterio.float32, count=1, nodata=cop_nodata, compress="lzw")
    os.makedirs(os.path.dirname(output_resampled_path) or ".", exist_ok=True)
    with rasterio.open(output_resampled_path, "w", **target_profile) as dst:
        dst.write(resampled, 1)

    # diagnostics
    is_nodata = np.isnan(resampled) if np.isnan(cop_nodata) else (resampled == cop_nodata)
    valid_data = resampled[~is_nodata]
    total = resampled.size
    print("\n" + "-" * 40)
    print("RESAMPLING DIAGNOSTIC RESULTS:")
    print("-" * 40)
    print(f"Output File:       {output_resampled_path}")
    print(f"Total Grid Cells:  {total} ({resampled.shape[0]}x{resampled.shape[1]})")
    print(f"NoData/NaN Cells:  {is_nodata.sum()} ({is_nodata.sum() / total * 100:.2f}%)")
    print(f"Valid Data Cells:  {valid_data.size} ({valid_data.size / total * 100:.2f}%)")
    if valid_data.size:
        print(f"Elevation Min:     {valid_data.min():.2f} meters")
        print(f"Elevation Max:     {valid_data.max():.2f} meters")
        print(f"Elevation Mean:    {valid_data.mean():.2f} meters")
    else:
        print("CRITICAL WARNING: Output raster contains NO valid elevation data!")
    print("=" * 60 + "\n")
    return output_resampled_path


def align_gee_features_to_master(gee_filepath: str, master_filepath: str, output_filepath: str) -> str:
    """Every band of the GEE feature stack -> the master (Copernicus) grid, bilinear, NaN = no data."""
    print("=" * 60)
    print("SNAPPING GEE FEATURES TO MASTER GRID")
    print("=" * 60)

    with rasterio.open(master_filepath) as master:
        master_meta = master.profile.copy()
        m_transform, m_crs = master.transform, master.crs
        m_w, m_h = master.width, master.height

    with rasterio.open(gee_filepath) as gee:
        gee_nodata = gee.nodata if gee.nodata is not None else np.nan
        num_bands = gee.count
        stack = np.empty((num_bands, m_h, m_w), dtype=np.float32)
        print(f"Resampling {num_bands} GEE feature bands to match master grid...")
        for i in range(1, num_bands + 1):
            # bilinear for continuous variables (slope, EVI, heights, ...)
            reproject(source=rasterio.band(gee, i), destination=stack[i - 1],
                      src_transform=gee.transform, src_crs=gee.crs, src_nodata=gee_nodata,
                      dst_transform=m_transform, dst_crs=m_crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            print(f" - Band {i} aligned.")

    master_meta.update(dtype=rasterio.float32, count=num_bands, nodata=np.nan, compress="lzw")
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    with rasterio.open(output_filepath, "w", **master_meta) as dst:
        dst.write(stack)

    print(f"\nSUCCESS: GEE Features aligned. New shape: {stack.shape}")
    print("=" * 60)
    return output_filepath


# =============================================================================
# 3. vertical datum: ODN -> EGM2008
#    per pixel: H_EGM2008 = H_ODN + N_OSGM15 - N_EGM2008
# =============================================================================
GEOID_GRIDS = ("uk_os_OSGM15_GB.tif", "us_nga_egm08_25.tif")


def download_geoid_grids(grid_dir: str, grids=GEOID_GRIDS, base_url: str = "https://cdn.proj.org") -> None:
    """Download the PROJ geoid grids once into grid_dir (skipped if already there)."""
    import certifi

    os.makedirs(grid_dir, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    for f in grids:
        path = os.path.join(grid_dir, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"found {f} ({os.path.getsize(path) / 1e6:.1f} MB)")
            continue
        print(f"downloading {f} ...")
        tmp = path + ".part"
        try:
            with urllib.request.urlopen(f"{base_url}/{f}", context=ctx) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out)
        except Exception as err:                       # fall back to the system downloader
            print(f"  python download failed ({err}); trying wget")
            os.system(f"wget -q -O '{tmp}' {base_url}/{f}")
        assert os.path.exists(tmp) and os.path.getsize(tmp) > 0, \
            f"could not download {f}: get it from {base_url}/{f} in a browser and put it in {grid_dir}"
        os.replace(tmp, path)
        print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def setup_pyproj(grid_dir: str, grids=GEOID_GRIDS, base_url: str = "https://cdn.proj.org") -> None:
    """Make sure the grids exist, tell PROJ where they are, and use local grids only."""
    download_geoid_grids(grid_dir, grids, base_url)
    pyproj.datadir.append_data_dir(grid_dir)
    pyproj.network.set_network_enabled(False)


def geoid_transformers() -> tuple[Transformer, Transformer]:
    """(to_odn, to_egm): ellipsoidal height 0 -> -N_OSGM15 and -N_EGM2008 at a lon/lat."""
    to_odn = Transformer.from_crs("EPSG:4937", "EPSG:4258+5701", always_xy=True, allow_ballpark=False)
    to_egm = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True, allow_ballpark=False)
    return to_odn, to_egm


def check_geoids(lon: float = -1.47, lat: float = 53.38) -> float:
    """Sanity check at one point (cell 13). Returns the EGM2008 - ODN offset in metres."""
    to_odn, to_egm = geoid_transformers()
    print("ODN step:    ", to_odn.description)
    print("EGM2008 step:", to_egm.description)
    _, _, t_odn = to_odn.transform(lon, lat, 0.0, errcheck=True)     # = -N_OSGM15
    _, _, t_egm = to_egm.transform(lon, lat, 0.0, errcheck=True)     # = -N_EGM2008
    print(f"test point: N_OSGM15 {-t_odn:.3f} m | N_EGM2008 {-t_egm:.3f} m | offset {t_egm - t_odn:+.3f} m")
    assert -60 < t_odn < -40 and -60 < t_egm < -40, "geoid heights implausible for GB: check the grid files"
    return t_egm - t_odn


def shift_to_EGM2008(src_dtm: str, dst_dtm: str) -> str:
    """LiDAR DTM in ODN heights -> EGM2008 heights, pixel by pixel. Run setup_pyproj() first."""
    to_odn, to_egm = geoid_transformers()

    with rasterio.open(src_dtm) as src:
        dtm = src.read(1).astype(np.float64)
        prof, T, crs, nodata = src.profile, src.transform, CRS.from_user_input(src.crs), src.nodata
    ok = np.isfinite(dtm) if nodata is None or np.isnan(nodata) else (np.isfinite(dtm) & (dtm != nodata))

    # pixel centres -> lon / lat
    r, c = np.nonzero(ok)
    x = T.c + (c + 0.5) * T.a + (r + 0.5) * T.b
    y = T.f + (c + 0.5) * T.d + (r + 0.5) * T.e
    if crs.is_geographic:
        lon, lat = x, y
    else:
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    print(f"{len(lon):,} valid pixels | lon {lon.min():.3f} to {lon.max():.3f} | "
          f"lat {lat.min():.3f} to {lat.max():.3f}")

    _, _, h_odn0 = to_odn.transform(lon, lat, np.zeros_like(lon), errcheck=True)
    _, _, h_egm0 = to_egm.transform(lon, lat, np.zeros_like(lon), errcheck=True)
    assert np.all(np.isfinite(h_odn0)) and np.all(np.isfinite(h_egm0)), "non-finite geoid heights"
    offset = h_egm0 - h_odn0

    out = np.full(dtm.shape, np.nan if nodata is None else nodata, np.float32)
    out[r, c] = (dtm[r, c] + offset).astype(np.float32)
    prof.update(dtype="float32", compress="deflate")
    os.makedirs(os.path.dirname(dst_dtm) or ".", exist_ok=True)
    with rasterio.open(dst_dtm, "w", **prof) as dst:
        dst.write(out, 1)
        dst.update_tags(VERTICAL_DATUM="EGM2008 height (EPSG:3855), converted from ODN (EPSG:5701) via OSGM15")
    print(f"offset (EGM2008 - ODN): min {offset.min():+.3f} | mean {offset.mean():+.3f} | max {offset.max():+.3f} m")
    print("saved", dst_dtm)
    return dst_dtm


# =============================================================================
# all steps for one area (the notebook's _align + shift_to_EGM2008, cells 81 and 89)
# =============================================================================
def prepare_area(digimap_raw: str, merge_vrt: str, copernicus: str, dtm_odn: str, dtm_egm: str,
                 gee_features: str, gee_aligned: str, grid_dir: str,
                 steps=("merge", "align", "datum"), grids=GEOID_GRIDS,
                 grid_url: str = "https://cdn.proj.org") -> None:
    if "merge" in steps:
        digimap_merge(digimap_raw, merge_vrt)
    if "align" in steps:
        align_digi_to_copernicus(merge_vrt, copernicus, dtm_odn)
        align_gee_features_to_master(gee_features, copernicus, gee_aligned)
    if "datum" in steps:
        setup_pyproj(grid_dir, grids, grid_url)
        check_geoids()
        shift_to_EGM2008(dtm_odn, dtm_egm)


def area_paths(cfg: dict, site: str) -> dict:
    """The file paths prepare_area needs, for 'site_a' or a key under sites: in config.yaml."""
    if site == "site_a":
        p = cfg["paths"]
        return dict(digimap_raw=p["digimap_dtm_raw"], merge_vrt=p["digimap_merge_vrt"],
                    copernicus=p["copernicus_dsm"], dtm_odn=p["dtm_30m_odn"], dtm_egm=p["dtm_30m_egm2008"],
                    gee_features=p["gee_features"], gee_aligned=p["gee_features_aligned"])
    s = cfg["sites"][site]
    return dict(digimap_raw=s["digimap_raw"], merge_vrt=s["merge_vrt"], copernicus=s["copernicus"],
                dtm_odn=s["dtm_odn"], dtm_egm=s["dtm"], gee_features=s["gee_features"],
                gee_aligned=s["feat"])


def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description="Merge, align and datum-shift one area.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--site", default="site_a", help="'site_a' or a key under sites: (e.g. tile_1)")
    p.add_argument("--steps", nargs="+", default=["merge", "align", "datum"],
                   choices=["merge", "align", "datum"])
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["datum"]
    prepare_area(**area_paths(cfg, args.site), grid_dir=cfg["paths"]["geoid_grids"], steps=args.steps,
                 grids=tuple(d["geoid_grids"]), grid_url=d["grid_url"])


if __name__ == "__main__":
    main()