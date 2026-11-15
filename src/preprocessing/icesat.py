
"""
Real ICESat-2 ATL08 ground heights -> observations on a site's 30 m grid.
 
  read_atl08_segments     one row per 100 m land segment inside a lon/lat box (pandas)
  add_segment_classes     day / night and canopy-height class columns, for summaries
  along_track_deviation   |ground height - running median along the same beam pass|
  apply_filter            the quality filters (uncertainty, ground photons, beam, night, deviation)
  segments_to_points      ellipsoid -> EGM2008 heights, lon/lat -> the DSM's CRS
  rasterize_obs           points -> obs_dh / obs_mask on the DSM grid (mean per pixel)
  site_icesat             all of the above for one prepared site (what the Site B results use)
  load_atl08              the older, simpler reader used by calibrate(): uncertainty cut only
 
 
The EGM2008 conversion needs the PROJ geoid grid us_nga_egm08_25.tif: run
dataset.prepare_rasters.setup_pyproj(<geoid_grids folder>) once per session first.

"""
 
from __future__ import annotations
 
import glob
import os
import re
 
import h5py
import numpy as np
import pandas as pd
import pyproj
import rasterio
 
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
 
# the filter chosen in the notebook (cell 106); config.yaml -> atl08.filter
DEFAULT_FILTER = dict(min_ph=20, max_dev=3.0)
 
 
# =============================================================================
# reading
# =============================================================================
def beam_type(h5, beam: str) -> str:
    """'strong' / 'weak' from the beam's own attribute; fallback: spacecraft orientation
    (orbit_info/sc_orient: 0 backward -> left beams strong, 1 forward -> right beams strong)."""
    t = h5[beam].attrs.get("atlas_beam_type", None)
    if t is not None:
        return t.decode() if isinstance(t, bytes) else str(t)
    if "orbit_info/sc_orient" in h5:
        o = int(h5["orbit_info/sc_orient"][0])
        if o in (0, 1):
            strong_side = "l" if o == 0 else "r"
            return "strong" if beam.endswith(strong_side) else "weak"
    return "unknown"
 
 
def read_atl08_segments(files, bbox_wgs84) -> pd.DataFrame:
    """One row per ATL08 100 m land segment inside bbox (lon_min, lat_min, lon_max, lat_max).
    Columns: file, date, beam, beam_type, lon, lat, h_te (ground, WGS84 ellipsoid, m), h_te_unc,
    n_te_ph (ground photons), n_seg_ph, h_canopy, night, msw. Fill values become NaN."""
    lon0, lat0, lon1, lat1 = bbox_wgs84
    parts = []
    for f in files:
        m = re.search(r"ATL08_(\d{8})", os.path.basename(f))
        date = pd.to_datetime(m.group(1), format="%Y%m%d") if m else pd.NaT
        with h5py.File(f, "r") as h5:
            for beam in BEAMS:
                g = f"{beam}/land_segments"
                if g not in h5:
                    continue
                la, lo = h5[f"{g}/latitude"][:], h5[f"{g}/longitude"][:]
                inb = (lo >= lon0) & (lo <= lon1) & (la >= lat0) & (la <= lat1)
                if not inb.any():
                    continue
 
                def get(path, g=g, inb=inb):
                    p = f"{g}/{path}"
                    return h5[p][:][inb].astype(np.float64) if p in h5 else np.full(int(inb.sum()), np.nan)
 
                parts.append(pd.DataFrame(dict(
                    file=os.path.basename(f), date=date, beam=beam, beam_type=beam_type(h5, beam),
                    lon=lo[inb], lat=la[inb],
                    h_te=get("terrain/h_te_best_fit"),          # ground height, WGS84 ellipsoid (m)
                    h_te_unc=get("terrain/h_te_uncertainty"),
                    n_te_ph=get("terrain/n_te_photons"),        # ground photons in the segment
                    n_seg_ph=get("n_seg_ph"),                   # all signal photons in the segment
                    h_canopy=get("canopy/h_canopy"),            # canopy height above ground (m)
                    night=get("night_flag"), msw=get("msw_flag"))))
    if not parts:
        print("no ATL08 segments inside the box")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    for c in ("h_te", "h_te_unc", "h_canopy", "n_te_ph", "n_seg_ph", "night", "msw"):
        df.loc[df[c].abs() > 1e9, c] = np.nan                   # fill values (~3.4e38, 2^31-1) -> NaN
    return df
 
 
def add_segment_classes(df: pd.DataFrame, canopy_edges=(-np.inf, 2, 5, 10, 20, np.inf)) -> pd.DataFrame:
    """Add 'light' (day / night / ?) and 'canopy' (canopy-height class) columns (from summarize_atl08)."""
    light = np.where(df.night == 1, "night", np.where(df.night == 0, "day", "?"))
    canopy = pd.cut(df.h_canopy, list(canopy_edges),
                    labels=["< 2 m", "2-5 m", "5-10 m", "10-20 m", "> 20 m"])
    return df.assign(light=light, canopy=canopy)
 
 
# =============================================================================
# filtering
# =============================================================================
def along_track_deviation(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """|ground height - running median of its neighbours on the same beam pass| (m).
    Segments are ordered by latitude within each file + beam."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["file", "beam"]):
        g = g.sort_values("lat")
        med = g.h_te.rolling(window, center=True, min_periods=3).median()
        out.loc[g.index] = (g.h_te - med).abs()
    return out
 
 
def apply_filter(df: pd.DataFrame, max_unc: float | None = None, min_ph: float | None = None,
                 strong_only: bool = False, night_only: bool = False,
                 max_dev: float | None = None) -> pd.DataFrame:
    """Segments that pass every filter given. max_dev needs a 'dev' column (along_track_deviation)."""
    keep = df.h_te.notna()
    if max_unc is not None:
        keep &= df.h_te_unc < max_unc
    if min_ph is not None:
        keep &= df.n_te_ph >= min_ph
    if strong_only:
        keep &= df.beam_type == "strong"
    if night_only:
        keep &= df.night == 1
    if max_dev is not None:
        keep &= df.dev < max_dev
    return df[keep]
 
 
# =============================================================================
# heights and coordinates
# =============================================================================
def segments_to_points(df: pd.DataFrame, dsm_crs, geoid_epsg: int = 3855
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y in the DSM's CRS and ground height in the DSM's geoid, from a (filtered) segment table."""
    to_geoid = pyproj.Transformer.from_crs("EPSG:4979", f"EPSG:4326+{geoid_epsg}", always_xy=True,
                                           allow_ballpark=False)
    _, _, h = to_geoid.transform(df.lon.values, df.lat.values, df.h_te.values, errcheck=True)
    x, y = pyproj.Transformer.from_crs("EPSG:4326", dsm_crs, always_xy=True).transform(
        df.lon.values, df.lat.values)
    print(f"{len(df):,} filtered ATL08 points")
    return np.asarray(x), np.asarray(y), np.asarray(h)
 
 
def rasterize_obs(x, y, h_ground, dsm, dsm_ok, transform, dh_range
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Points -> obs_dh [H, W] (mean ground - DSM per pixel) and obs_mask [H, W].
    Points off the grid, on invalid DSM pixels, or with delta_h outside dh_range are dropped."""
    H, W = dsm.shape
    rows, cols = rasterio.transform.rowcol(transform, x, y)
    rows, cols, h = np.asarray(rows), np.asarray(cols), np.asarray(h_ground, np.float64)
    keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & np.isfinite(h)
    rows, cols, h = rows[keep], cols[keep], h[keep]
    keep = dsm_ok[rows, cols]
    rows, cols, h = rows[keep], cols[keep], h[keep]
    dh = h - dsm[rows, cols]
    keep = (dh >= dh_range[0]) & (dh <= dh_range[1])
    print(f"{len(dh):,} points on valid DSM pixels | {int((~keep).sum()):,} outside {dh_range} m dropped")
    rows, cols, dh = rows[keep], cols[keep], dh[keep]
    s, n = np.zeros((H, W)), np.zeros((H, W), np.int64)
    np.add.at(s, (rows, cols), dh)
    np.add.at(n, (rows, cols), 1)
    obs_mask = n > 0
    return np.where(obs_mask, s / np.maximum(n, 1), 0.0).astype(np.float32), obs_mask
 
 
# =============================================================================
# one prepared site (cell 106)
# =============================================================================
def site_icesat(S: dict, atl08_dir: str, flt: dict | None = None, window: int = 5,
                geoid_epsg: int = 3855, years=None) -> dict:
    """Real ATL08 on a site prepared by dataset.build_inputs.prepare_site.
      2a the .h5 files in atl08_dir (download them with dataset/data_download.py)
      2b read the segments inside the site + the chosen filter
      2c convert to EGM2008 / the site's CRS
      2d rasterise with the training delta_h range
    years: e.g. (2023,) keeps only segments acquired in those years (the "one year of ATL08" ablation);
           the filtered segments per year are printed first, so you can pick a typical year.
    Returns dict(obs_dh, obs_mask, n_segments)."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
 
    flt = DEFAULT_FILTER if flt is None else flt
    files = sorted(glob.glob(f"{atl08_dir}/*.h5"))                                   # 2a
    assert files, (f"no .h5 files in {atl08_dir}: download them with "
                   f"python -m dataset.data_download --site <site> --what atl08")
 
    bounds = array_bounds(S["H"], S["W"], S["transform"])                            # 2b
    bbox = transform_bounds(S["crs"], "EPSG:4326", *bounds, densify_pts=21)
    seg = read_atl08_segments(files, bbox)
    seg = seg[seg.h_te.notna()]
    seg = seg.assign(dev=along_track_deviation(seg, window))
    seg_f = apply_filter(seg, **flt)
    print(f"ATL08: {len(files)} files | {len(seg):,} segments -> {len(seg_f):,} after the filter")
    if years is not None:
        print("filtered segments per year:", seg_f.groupby(seg_f.date.dt.year).size().to_dict())
        seg_f = seg_f[seg_f.date.dt.year.isin(list(years))]
        print(f"years {list(years)}: {len(seg_f):,} segments")
 
    x, y, h = segments_to_points(seg_f, S["crs"].to_string(), geoid_epsg)            # 2c
    obs_dh, obs_mask = rasterize_obs(x, y, h, S["dsm"], S["dsm_ok"], S["transform"],   # 2d
                                     (S["stats"]["target_min"], S["stats"]["target_max"]))
    d = (obs_dh - S["target"])[obs_mask & S["valid"]]
    if d.size:
        print(f"ATL08 pixels: {obs_mask.sum():,} | of which with lidar: {(obs_mask & S['valid']).sum():,} | "
              f"ATL08 - lidar there: median {np.median(d):+.2f} m, RMSE {np.sqrt(np.mean(d ** 2)):.2f} m")
    return dict(obs_dh=obs_dh, obs_mask=obs_mask, n_segments=len(seg_f))
 
 
# =============================================================================
# older reader used by calibrate() (cell 82)
# =============================================================================
def load_atl08(files, dsm_crs, dsm_geoid_epsg: int = 3855, max_unc_m: float = 5.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ATL08 100 m land segments -> x, y in the DSM's CRS and ground height in the DSM's geoid.
    Only filter: h_te_uncertainty < max_unc_m. No bounding box (rasterize_obs drops points off the grid)."""
    lon, lat, h = [], [], []
    for f in files:
        with h5py.File(f, "r") as h5:
            for beam in BEAMS:
                g = f"{beam}/land_segments"
                if g not in h5:
                    continue
                hb = h5[f"{g}/terrain/h_te_best_fit"][:].astype(np.float64)
                unc = h5[f"{g}/terrain/h_te_uncertainty"][:].astype(np.float64)
                keep = np.isfinite(hb) & (hb < 1e30) & (unc < max_unc_m)   # fill value ~3.4e38
                lat.append(h5[f"{g}/latitude"][:][keep])
                lon.append(h5[f"{g}/longitude"][:][keep])
                h.append(hb[keep])
    if not h:
        print("no ATL08 segments found")
        return np.zeros(0), np.zeros(0), np.zeros(0)
    lon, lat, h_ell = np.concatenate(lon), np.concatenate(lat), np.concatenate(h)
    pyproj.network.set_network_enabled(True)        # PROJ may download the geoid grid if it is missing
    to_geoid = pyproj.Transformer.from_crs("EPSG:4979", f"EPSG:4326+{dsm_geoid_epsg}", always_xy=True)
    _, _, h_geo = to_geoid.transform(lon, lat, h_ell)
    shift = np.nanmean(h_ell - h_geo)
    assert np.isfinite(shift) and abs(shift) > 0.5, "ellipsoid -> geoid conversion did nothing (geoid grid missing)"
    x, y = pyproj.Transformer.from_crs("EPSG:4326", dsm_crs, always_xy=True).transform(lon, lat)
    print(f"{len(h_geo):,} ATL08 segments | ellipsoid -> geoid {shift:+.1f} m on average")
    return np.asarray(x), np.asarray(y), np.asarray(h_geo)
 