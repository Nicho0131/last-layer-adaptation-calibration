"""
download the raw inputs.

  - Copernicus GLO-30 DSM            (Google Earth Engine -> Google Drive)
  - 8-band feature stack             (Google Earth Engine -> Google Drive)
  - ICESat-2 ATL08 granules          (NASA earthaccess -> a local / Drive folder)

"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import time

import rasterio
from rasterio.warp import transform_bounds


# =============================================================================
# shared helpers
# =============================================================================
def tile_bounds(folder: str) -> tuple[float, float, float, float]:
    """Union bounds (left, bottom, right, top) of all raw .tif tiles in `folder`, in their own CRS
    (British National Grid for Digimap). Merged / averaged outputs are skipped."""
    files = [f for f in glob.glob(f"{folder}/*.tif")
             if not f.endswith(("_merge.tif", "_average.tif"))]
    assert files, f"no .tif files in {folder}"
    L, B, R, T = [], [], [], []
    for f in files:
        with rasterio.open(f) as src:
            b = src.bounds
            L.append(b.left); B.append(b.bottom); R.append(b.right); T.append(b.top)
    print(f"{len(files)} tiles in {folder}")
    return min(L), min(B), max(R), max(T)


def to_lonlat_bounds(bounds, crs: str = "EPSG:27700") -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees. `bounds` is (left, bottom, right, top) in `crs`."""
    if crs == "EPSG:4326":
        w, s, e, n = bounds
    else:
        w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    # sorted so that south < north even if the input had them swapped
    return min(w, e), min(s, n), max(w, e), max(s, n)


# =============================================================================
# Google Earth Engine
# =============================================================================
_EE_READY = False


def init_ee(project: str) -> None:
    """Authenticate (once per session) and initialise Earth Engine with your Cloud project."""
    global _EE_READY
    import ee
    if not _EE_READY:
        ee.Authenticate()
        ee.Initialize(project=project)
        _EE_READY = True


def ee_name(s: str) -> str:
    """A string valid as an Earth Engine task description / file prefix."""
    return re.sub(r"[^A-Za-z0-9.,:;_-]", "_", s)[:100]


def ee_roi(lonlat_bounds):
    import ee
    w, s, e, n = lonlat_bounds
    return ee.Geometry.BBox(w, s, e, n)


def copernicus_image(roi, collection: str = "COPERNICUS/DEM/GLO30_2024_1"):
    """Copernicus GLO-30 DSM mosaic clipped to roi (cell 81, download_dsm)."""
    import ee
    return ee.ImageCollection(collection).select("DEM").mosaic().clip(roi)


def feature_stack_image(roi, crs: str = "EPSG:27700", scale: int = 30,
                        collection: str = "COPERNICUS/DEM/GLO30_2024_1",
                        year: int = 2021, max_cloud_pct: float = 70):
    """The 8-band feature stack (cell 81, download_feat):
    Fused_DSM, Slope, Aspect, TRI, Canopy_Height, LAI, EVI, Building_Height.
    Water (80) and wetland (70) in ESA WorldCover are masked. All bands float32."""
    import ee

    fused_dsm = (ee.ImageCollection(collection).filterBounds(roi).select("DEM").mosaic()
                 .setDefaultProjection(crs=crs).reproject(crs=crs, scale=scale)
                 .rename("Fused_DSM"))

    slope = ee.Terrain.slope(fused_dsm).rename("Slope")
    aspect = ee.Terrain.aspect(fused_dsm).rename("Aspect")

    # terrain ruggedness index over the 8 neighbours
    kernel = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    focal_sum = fused_dsm.reduceNeighborhood(reducer=ee.Reducer.sum(), kernel=kernel)
    focal_sum_sq = fused_dsm.pow(2).reduceNeighborhood(reducer=ee.Reducer.sum(), kernel=kernel)
    tri = (focal_sum_sq
           .subtract(fused_dsm.multiply(focal_sum).multiply(2))
           .add(fused_dsm.pow(2).multiply(8))
           .sqrt()
           .rename("TRI"))

    canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").rename("Canopy_Height")

    start, end = f"{year}-01-01", f"{year}-12-31"
    lai = (ee.ImageCollection("MODIS/061/MCD15A3H")
           .filterDate(start, end).select("Lai").median().multiply(0.1).rename("LAI"))

    def add_evi(image):
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))",
            {"NIR": image.select("B8"), "RED": image.select("B4"), "BLUE": image.select("B2")},
        ).rename("EVI")
        return image.addBands(evi)

    evi = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(roi)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
           .map(add_evi)
           .select("EVI")
           .median())

    building_height = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
                       .select("built_height").rename("Building_Height"))

    world_cover = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    valid_land = world_cover.neq(80).multiply(world_cover.neq(70))

    return (ee.Image.cat([fused_dsm, slope, aspect, tri, canopy_height, lai, evi, building_height])
            .updateMask(valid_land).clip(roi).toFloat())       # float32 for GeoTIFF


def export_to_drive(image, description: str, folder: str, roi, scale: int = 30,
                    crs: str = "EPSG:27700", max_pixels: float = 1e13, wait: bool = True,
                    poll_s: int = 30):
    """Start an Earth Engine export to Google Drive; optionally wait until it finishes."""
    import ee
    task = ee.batch.Export.image.toDrive(
        image=image, description=ee_name(description), folder=ee_name(folder),
        region=roi, scale=scale, crs=crs, maxPixels=max_pixels, fileFormat="GeoTIFF")
    task.start()
    print(f"export started: {description} -> Drive folder '{ee_name(folder)}'")
    if not wait:
        return task
    while task.active():
        print(f"  task status: {task.status()['state']}")
        time.sleep(poll_s)
    state = task.status()["state"]
    print(f"export finished: {state}")
    if state == "FAILED":
        raise RuntimeError(f"Earth Engine export failed: {task.status().get('error_message')}")
    return task


def download_dsm(lonlat_bounds, tile: str, crs: str = "EPSG:27700", scale: int = 30,
                 collection: str = "COPERNICUS/DEM/GLO30_2024_1", wait: bool = True):
    """Copernicus DSM for one area -> Drive folder Copernicus_tile_<tile>."""
    roi = ee_roi(lonlat_bounds)
    return export_to_drive(copernicus_image(roi, collection),
                           description=f"Copernicus_GLO30_2024_{tile}",
                           folder=f"Copernicus_tile_{tile}",
                           roi=roi, scale=scale, crs=crs, max_pixels=1e10, wait=wait)


def download_feat(lonlat_bounds, tile: str, crs: str = "EPSG:27700", scale: int = 30,
                  collection: str = "COPERNICUS/DEM/GLO30_2024_1", year: int = 2021,
                  max_cloud_pct: float = 70, wait: bool = True):
    """8-band feature stack for one area -> Drive folder GEE_FEATURE_TILE_<tile>."""
    roi = ee_roi(lonlat_bounds)
    image = feature_stack_image(roi, crs, scale, collection, year, max_cloud_pct)
    return export_to_drive(image,
                           description=f"GEE_FEATURE_STACK_{tile}",
                           folder=f"GEE_FEATURE_TILE_{tile}",
                           roi=roi, scale=scale, crs=crs, max_pixels=1e13, wait=wait)


# =============================================================================
# ICESat-2 ATL08 (NASA earthaccess)
# =============================================================================
_EARTHDATA_LOGGED_IN = False


def _size_mb(granule) -> float:
    """Granule size in MB, whether this earthaccess version has size as a method or a value."""
    s = granule.size
    return float(s() if callable(s) else s)


def download_atl08(bounds, out_dir: str, start: str = "2022-01-01", end: str = "2024-12-31",
                   crs: str = "EPSG:27700", version: str = "007", dry_run: bool = False,
                   max_gb: float | None = None, stage_dir: str = "/content/atl08_stage") -> list[str]:
    """Download every ATL08 granule over `bounds` (left, bottom, right, top in `crs`) into out_dir.
    Granules already in out_dir are skipped. Files are downloaded to the fast local `stage_dir`
    first, then copied to out_dir (Drive is slow for many small writes).
    Returns the sorted list of .h5 files in out_dir."""
    import earthaccess

    global _EARTHDATA_LOGGED_IN
    if not _EARTHDATA_LOGGED_IN:
        earthaccess.login(strategy="interactive", persist=True)
        _EARTHDATA_LOGGED_IN = True

    bbox = to_lonlat_bounds(bounds, crs)
    results = earthaccess.search_data(short_name="ATL08", version=version,
                                      bounding_box=tuple(bbox), temporal=(start, end))

    os.makedirs(out_dir, exist_ok=True)
    have = set(os.listdir(out_dir))
    todo = [r for r in results if not all(os.path.basename(u) in have for u in r.data_links())]
    gb_all = sum(_size_mb(r) for r in results) / 1024
    gb_todo = sum(_size_mb(r) for r in todo) / 1024
    print(f"bbox {tuple(round(v, 4) for v in bbox)} | {start} to {end}")
    print(f"{len(results)} granules ({gb_all:.2f} GB) | already here: {len(results) - len(todo)} | "
          f"to download: {len(todo)} ({gb_todo:.2f} GB)")
    if dry_run or not todo:
        return sorted(glob.glob(f"{out_dir}/*.h5"))
    if max_gb is not None and gb_todo > max_gb:
        raise RuntimeError(f"{gb_todo:.2f} GB exceeds max_gb={max_gb}: narrow the dates or raise the limit")

    os.makedirs(stage_dir, exist_ok=True)
    local = earthaccess.download(todo, local_path=stage_dir)
    for f in local:
        shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))
        os.remove(f)
    files = sorted(glob.glob(f"{out_dir}/*.h5"))
    print(f"done: {len(files)} ATL08 files in {out_dir}")
    return files


# =============================================================================
# run from config.yaml
# =============================================================================
def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description="Download Copernicus DSM, GEE features and ATL08.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--site", default="site_a",
                   help="'site_a' (pre-training area, gee.roi_bbox) or a key under sites: (e.g. tile_1)")
    p.add_argument("--what", nargs="+", default=["gee"], choices=["gee", "atl08"])
    p.add_argument("--no-wait", action="store_true", help="start GEE exports without waiting")
    p.add_argument("--dry-run", action="store_true", help="ATL08: list granules, download nothing")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["gee"]

    if args.site == "site_a":
        bounds, bounds_crs, tile = g["roi_bbox"], "EPSG:4326", "site_a"
        atl08_dir = None                                   # Site A trains on simulated ICESat tracks
        if "atl08" in args.what:
            p.error("site_a has no ATL08 folder in config.yaml (it uses simulated tracks); "
                    "use --site tile_1 or add an atl08_dir")
    else:
        site = cfg["sites"][args.site]
        bounds, bounds_crs = tile_bounds(site["digimap_raw"]), "EPSG:27700"
        tile = args.site.removeprefix("tile_")             # "tile_1" -> "1", as in the notebook names
        atl08_dir = site["atl08_dir"]

    if "gee" in args.what:
        init_ee(g["project"])
        lonlat = to_lonlat_bounds(bounds, bounds_crs)
        download_dsm(lonlat, tile, crs=g["export_crs"], scale=g["scale_m"],
                     collection=g["copernicus_collection"], wait=not args.no_wait)
        download_feat(lonlat, tile, crs=g["export_crs"], scale=g["scale_m"],
                      collection=g["copernicus_collection"], year=g["feature_year"],
                      max_cloud_pct=g["max_cloud_pct"], wait=not args.no_wait)

    if "atl08" in args.what:
        a = cfg["atl08"]
        download_atl08(bounds, atl08_dir, start=a["start"], end=a["end"], crs=bounds_crs,
                       version=str(a["version"]), dry_run=args.dry_run, max_gb=a["max_download_gb"])


if __name__ == "__main__":
    main()