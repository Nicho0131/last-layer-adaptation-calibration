"""
reference DEMs for comparison (FABDEM, FathomDEM) on exactly a site's 30 m grid.

Both products come as 1 x 1 degree GeoTIFF tiles inside large zip files. This module
  - works out which 1-degree tiles a site needs
  - finds them inside the zips without unpacking (GDAL /vsizip/)
  - mosaics them on the tiles' own pixel grid (no half-pixel shift)
  - resamples the mosaic bilinearly onto the site's grid and writes one GeoTIFF per product
  - read_baseline() reads such a file back, checked against the site's grid
"""

from __future__ import annotations

import argparse
import math
import os
import zipfile

import numpy as np
import rasterio
from affine import Affine
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds


# =============================================================================
# tile names
# =============================================================================
def fabdem_name(lat: int, lon: int) -> str:
    """FABDEM V1-2 tile whose south-west corner is (lat, lon), e.g. N53W002_FABDEM_V1-2.tif."""
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}_FABDEM_V1-2.tif"


def fathom_name(lat: int, lon: int) -> str:
    """FathomDEM tile whose south-west corner is (lat, lon), e.g. n53w002.tif."""
    return f"{'n' if lat >= 0 else 's'}{abs(lat):02d}{'e' if lon >= 0 else 'w'}{abs(lon):03d}.tif"


# =============================================================================
# the tiles a site needs
# =============================================================================
def site_lonlat_bounds(template_path: str, buffer_deg: float = 0.01) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of the template, plus a small buffer for the bilinear edge."""
    with rasterio.open(template_path) as t:
        w, s, e, n = transform_bounds(t.crs, "EPSG:4326", *t.bounds, densify_pts=21)
    return w - buffer_deg, s - buffer_deg, e + buffer_deg, n + buffer_deg


def tiles_for_site(template_path: str, name_fn, buffer_deg: float = 0.01) -> list[str]:
    w, s, e, n = site_lonlat_bounds(template_path, buffer_deg)
    return [name_fn(lat, lon)
            for lat in range(math.floor(s), math.floor(n) + 1)
            for lon in range(math.floor(w), math.floor(e) + 1)]


# =============================================================================
# find tiles inside the zips
# =============================================================================
def index_zips(zip_paths) -> dict[str, str]:
    """{tile file name (lower case): GDAL path inside its zip} for every .tif in the given zips."""
    zip_paths = [zip_paths] if isinstance(zip_paths, str) else list(zip_paths)
    idx = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.lower().endswith(".tif"):
                    idx[os.path.basename(m).lower()] = f"/vsizip/{zp}/{m}"
    return idx


# =============================================================================
# one product on one site's grid
# =============================================================================
def same_grid(path: str, template_path: str) -> bool:
    with rasterio.open(path) as a, rasterio.open(template_path) as t:
        return ((a.height, a.width) == (t.height, t.width)
                and a.transform.almost_equals(t.transform) and a.crs == t.crs)


def dem_on_grid(template_path: str, zip_paths, name_fn, out_path: str, scale: float = 1.0,
                label: str = "DEM", overwrite: bool = False, buffer_deg: float = 0.01) -> str:
    """A global 1-degree-tiled DEM (inside zips) -> a GeoTIFF on exactly the template's grid (metres)."""
    if os.path.exists(out_path) and not overwrite:                    # already made
        assert same_grid(out_path, template_path), f"{out_path} exists but is not on the template grid"
        print(f"{label}: already on the grid -> {out_path}")
        return out_path

    names = tiles_for_site(template_path, name_fn, buffer_deg)
    idx = index_zips(zip_paths)
    have = [idx[nm.lower()] for nm in names if nm.lower() in idx]
    missing = [nm for nm in names if nm.lower() not in idx]
    if missing:
        print(f"{label}: not in the zip(s) (open sea, or wrong zip): {missing}")
    assert have, f"{label}: none of {names} found; zip(s) hold e.g. {sorted(idx)[:3]}"

    # Snap the mosaic bounds OUTWARD onto the tiles' own 1" pixel grid. With arbitrary bounds,
    # rasterio.merge rounds its read/write windows to whole pixels, which shifts the mosaic by up to
    # a pixel (~20-30 m): the striped slope errors the co-registration check found.
    w, s, e, n = site_lonlat_bounds(template_path, buffer_deg)
    with rasterio.open(have[0]) as s0:
        T0, (rx, ry) = s0.transform, s0.res
        reg = s0.tags().get("AREA_OR_POINT", "not set (Area)")
    w = T0.c + math.floor((w - T0.c) / rx) * rx
    e = T0.c + math.ceil((e - T0.c) / rx) * rx
    n = T0.f - math.floor((T0.f - n) / ry) * ry
    s = T0.f - math.ceil((T0.f - s) / ry) * ry
    print(f"{label}: tile pixel {rx * 3600:.3f}\" x {ry * 3600:.3f}\" | registration tag: {reg}")

    srcs = [rasterio.open(p) for p in have]
    try:
        mosaic, m_tr = merge(srcs, bounds=(w, s, e, n), res=(rx, ry), nodata=np.nan, dtype="float32")
        src_crs = srcs[0].crs
    finally:
        for src in srcs:
            src.close()

    with rasterio.open(template_path) as t:                           # onto the site grid
        prof = t.profile.copy()
        out = np.full((t.height, t.width), np.nan, np.float32)
        reproject(mosaic[0], out, src_transform=m_tr, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=t.transform, dst_crs=t.crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    out *= scale                                                       # units -> metres

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.update_tags(SOURCE=label, VERTICAL_DATUM="EGM2008 (EPSG:3855)",
                        TILES=",".join(os.path.basename(h) for h in have))
    print(f"{label}: {len(have)} tile(s) | {np.isfinite(out).mean():.1%} valid | "
          f"{np.nanmin(out):.1f} to {np.nanmax(out):.1f} m -> {out_path}")
    return out_path


def fabdem_on_grid(template_path: str, fabdem_zips, out_path: str, **kw) -> str:
    """FABDEM V1-2: metres. fabdem_zips: e.g. both UK zips (N50W010-N60E000, N50E000-N60E010)."""
    return dem_on_grid(template_path, fabdem_zips, fabdem_name, out_path, scale=1.0, label="FABDEM", **kw)


def fathomdem_on_grid(template_path: str, fathom_zips, out_path: str, **kw) -> str:
    """FathomDEM: stored in centimetres -> metres."""
    return dem_on_grid(template_path, fathom_zips, fathom_name, out_path, scale=0.01, label="FathomDEM", **kw)


# =============================================================================
# the one call per site
# =============================================================================
def baselines_for_site(template_path: str, out_dir: str, name: str, fabdem_zips=None,
                       fathom_zips=None, overwrite: bool = False) -> dict[str, str]:
    """template_path: the site's ALIGNED feature stack (its 30 m BNG grid).
    Returns {product: path} for the products you give zips for."""
    out = {}
    if fabdem_zips:
        out["FABDEM"] = fabdem_on_grid(template_path, fabdem_zips, f"{out_dir}/FABDEM_{name}.tif",
                                       overwrite=overwrite)
    if fathom_zips:
        out["FathomDEM"] = fathomdem_on_grid(template_path, fathom_zips, f"{out_dir}/FathomDEM_{name}.tif",
                                             overwrite=overwrite)
    return out


def read_baseline(path: str, site: dict, box=None) -> np.ndarray:
    """A baseline DEM as float32 [H, W], NaN = no data, checked to be on the site's grid.
    site: the dict from build_inputs.prepare_site (needs H, W, transform).
    box:  the same (y0, x0, y1, x1) the site was cut to, if any."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        T, nod = src.transform, src.nodata
    if nod is not None and np.isfinite(nod):
        a[a == nod] = np.nan
    if a.shape != (site["H"], site["W"]):                  # the site was cut to a box: cut the baseline too
        assert box is not None, f"{path}: shape {a.shape} differs from the site and no box was given"
        y0, x0, y1, x1 = box
        a, T = a[y0:y1, x0:x1], T * Affine.translation(x0, y0)
    assert T.almost_equals(site["transform"]), f"{path} is not on the site's grid"
    return a


# =============================================================================
# run from config.yaml
# =============================================================================
def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description="Put FABDEM / FathomDEM on a test site's grid.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--site", default="tile_1", help="a key under sites: in config.yaml")
    p.add_argument("--name", default=None, help="name in the output file (default: the site key)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    b, site = cfg["baselines"], cfg["sites"][args.site]
    paths = baselines_for_site(site["feat"], b["out_dir"], args.name or args.site,
                               fabdem_zips=b.get("fabdem_zips"), fathom_zips=b.get("fathom_zips"),
                               overwrite=args.overwrite)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()