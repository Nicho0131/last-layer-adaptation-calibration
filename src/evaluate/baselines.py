"""
Methods the model is compared against, and the co-registration check of reference DEMs.

    network_offset          network only + ONE scalar: mean(ATL08 delta_h - mu) over the ATL08 pixels
    atl08_mean_only         no network: mu = mean ATL08 delta_h, sigma = SD of the ATL08 values
    add_dsm_offset          B3: DSM + mean ATL08 delta_h (one scalar, no network)
    ridge_baseline          B4: per 16 km cell, ridge regression of delta_h on canopy + building height
    regression_kriging      B5: B4 + ordinary kriging of the B4 residuals (sigma = kriging SD)
    add_classical_baselines B3 + B4 + B5 into a P dict
    coregistration_check    horizontal shift of a DEM against the lidar ground (open-ground slopes)

Reference DEMs themselves (FABDEM / FathomDEM on the site grid, read_baseline) are in
dataset/reference_dems.py.

All functions take the S / OBS / P dicts described in src/evaluation/inference.py.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift as nd_shift
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from src.evaluation.metrics import error_metrics


# =============================================================================
# global baselines
# =============================================================================
def network_offset(net: dict, OBS: dict) -> tuple[dict, float]:
    """The network-only prediction plus one scalar b = mean(ATL08 delta_h - mu) over the ATL08 pixels;
    sigma = the network's sigma. Returns (dict(mu, sigma), b)."""
    o = OBS["obs_mask"]
    b = float(np.mean(OBS["obs_dh"][o] - net["mu"][o])) if o.any() else 0.0
    return dict(mu=net["mu"] + b, sigma=net["sigma"]), b


def atl08_mean_only(S: dict, OBS: dict) -> tuple[dict, float, float]:
    """No network: mu = mean ATL08 delta_h everywhere, sigma = SD of the ATL08 delta_h (cell 129).
    Returns (dict(mu, sigma), mean, SD)."""
    o = OBS["obs_mask"]
    m, s = float(np.mean(OBS["obs_dh"][o])), float(np.std(OBS["obs_dh"][o]))
    return (dict(mu=np.full((S["H"], S["W"]), m, np.float32), sigma=np.full((S["H"], S["W"]), s, np.float32)),
            m, s)


def add_dsm_offset(P: dict, S: dict, OBS: dict, name: str = "DSM + ATL08 offset") -> float:
    """B3: the Copernicus DSM plus ONE scalar, the mean ATL08 delta_h over the track pixels (no network)."""
    o = OBS["obs_mask"]
    b = float(np.mean(OBS["obs_dh"][o])) if o.any() else 0.0
    P[name] = dict(mu=np.full((S["H"], S["W"]), b, np.float32), sigma=None)
    return b


# =============================================================================
# B4: regional ridge regression
# =============================================================================
def _cell_index(S: dict, cell_km: float) -> tuple[np.ndarray, int]:
    """Cell of every pixel for square cells of cell_km, laid out like the model's grid level
    (number of cells = ceil(size / cell), spread evenly over the site)."""
    res_px = cell_km * 1000 / S["pixel_m"]
    Hg, Wg = max(1, int(np.ceil(S["H"] / res_px))), max(1, int(np.ceil(S["W"] / res_px)))
    cy = np.minimum((np.arange(S["H"]) * Hg // S["H"]), Hg - 1)
    cx = np.minimum((np.arange(S["W"]) * Wg // S["W"]), Wg - 1)
    return cy[:, None] * Wg + cx[None, :], Hg * Wg


def ridge_baseline(S: dict, OBS: dict, cell_km: float = 16.0, lam: float = 1.0, min_pts: int = 30
                   ) -> tuple[np.ndarray, np.ndarray]:
    """B4: per cell, ridge regression of the ATL08 delta_h on standardised canopy and building height
    (the intercept is not penalised); cells with < min_pts ATL08 pixels use the site-wide fit.
    Returns mu [H, W] and the residuals at the ATL08 pixels."""
    bldg = S.get("bldg")
    if bldg is None:
        print("no building band in S: B4 / B5 use canopy only")
        bldg = np.zeros((S["H"], S["W"]), np.float32)
    o = OBS["obs_mask"]
    y = OBS["obs_dh"][o].astype(np.float64)
    feats = [S["canopy"].astype(np.float64), np.nan_to_num(bldg).astype(np.float64)]
    mz = [f[o].mean() for f in feats]
    sz = [max(f[o].std(), 1e-6) for f in feats]
    Z = [(f - m) / s for f, m, s in zip(feats, mz, sz)]                   # standardised by the ATL08 pixels
    X_all = np.stack([np.ones((S["H"], S["W"]))] + Z, -1)               # [H, W, 3]
    cell, n_cells = _cell_index(S, cell_km)
    pen = lam * np.diag([0.0, 1.0, 1.0])

    def solve(Xo, yo):
        return np.linalg.solve(Xo.T @ Xo + pen + 1e-9 * np.eye(3), Xo.T @ yo)

    Xo, co = X_all[o], cell[o]
    beta_site = solve(Xo, y)
    mu = np.empty((S["H"], S["W"]), np.float64)
    n_local = 0
    for k in range(n_cells):
        m = co == k
        beta = solve(Xo[m], y[m]) if m.sum() >= min_pts else beta_site
        n_local += m.sum() >= min_pts
        pix = cell == k
        mu[pix] = X_all[pix] @ beta
    print(f"B4 ridge: {n_cells} cells of {cell_km:g} km, {n_local} fitted locally (>= {min_pts} ATL08 px), "
          f"the rest with the site-wide fit")
    return mu.astype(np.float32), (y - mu[o])


# =============================================================================
# B5: regression kriging
# =============================================================================
def _exp_vgm(h, c0, c, a):
    return c0 + c * (1.0 - np.exp(-h / a))


def fit_residual_variogram(xy, r, max_lag_m: float = 10000.0, short_m: float = 2000.0, n_bins: int = 20,
                           n_long: int = 2_000_000, seed: int = 0):
    """Exponential variogram (nugget c0, partial sill c, range parameter a in m) of the residuals r at xy.
    Short lags from all pairs within short_m (KD-tree), long lags from random pairs.
    Returns ((c0, c, a), (lag centres, semivariance))."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(xy)
    pr = tree.query_pairs(short_m, output_type="ndarray")
    if len(pr) > 3_000_000:
        pr = pr[rng.choice(len(pr), 3_000_000, replace=False)]
    i, j = rng.integers(0, len(r), n_long), rng.integers(0, len(r), n_long)
    d_long = np.hypot(*(xy[i] - xy[j]).T)
    keep = (d_long > short_m) & (d_long <= max_lag_m)
    I = np.concatenate([pr[:, 0], i[keep]]); J = np.concatenate([pr[:, 1], j[keep]])
    d = np.hypot(*(xy[I] - xy[J]).T)
    g = 0.5 * (r[I] - r[J]) ** 2
    edges = np.geomspace(max(30.0, d[d > 0].min()), max_lag_m, n_bins + 1)
    b = np.digitize(d, edges) - 1
    ok = [k for k in range(n_bins) if (b == k).sum() >= 30]
    h = np.array([d[b == k].mean() for k in ok]); gm = np.array([g[b == k].mean() for k in ok])
    n = np.array([(b == k).sum() for k in ok])
    v = float(np.var(r))
    try:
        (c0, c, a), _ = curve_fit(_exp_vgm, h, gm, p0=(0.3 * v, 0.7 * v, 1000.0),
                                  bounds=([0, 0, 30.0], [10 * v, 10 * v, 1e5]), sigma=1 / np.sqrt(n), maxfev=20000)
    except RuntimeError:
        c0, c, a = v, 1e-6, 1000.0                                      # no spatial structure: pure nugget
    print(f"B5 residual variogram: nugget {c0:.3f} m², partial sill {c:.3f} m², range parameter {a / 1000:.2f} km "
          f"(practical range ~{3 * a / 1000:.1f} km)")
    return (c0, c, a), (h, gm)


def ordinary_kriging(xy_obs, r_obs, xy_pred, params, k: int = 32, batch: int = 20000):
    """Ordinary kriging with the k nearest observations. Returns prediction and kriging variance."""
    c0, c, a = params
    k = min(k, len(r_obs))
    tree = cKDTree(xy_obs)
    pred = np.empty(len(xy_pred)); var = np.empty(len(xy_pred))
    for s in range(0, len(xy_pred), batch):
        q = xy_pred[s:s + batch]
        d0, idx = tree.query(q, k)
        d0, idx = d0.reshape(len(q), k), idx.reshape(len(q), k)
        P_ = xy_obs[idx]                                                # [b, k, 2]
        Dij = np.linalg.norm(P_[:, :, None, :] - P_[:, None, :, :], axis=-1)
        C = c * np.exp(-Dij / a) + c0 * np.eye(k)[None]                 # nugget on the diagonal only
        A = np.ones((len(q), k + 1, k + 1)); A[:, :k, :k] = C; A[:, k, k] = 0.0
        rhs = np.ones((len(q), k + 1)); rhs[:, :k] = c * np.exp(-d0 / a)
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        w, mu_l = sol[:, :k], sol[:, k]
        pred[s:s + batch] = (w * r_obs[idx]).sum(1)
        var[s:s + batch] = (c0 + c) - (w * rhs[:, :k]).sum(1) - mu_l
    return pred, np.maximum(var, 1e-6)


def regression_kriging(S: dict, OBS: dict, mu4, resid, step_px: int = 4, k: int = 32,
                       atl08_noise_var: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """B5: B4 + ordinary kriging of the B4 residuals. Kriging runs on a grid every step_px pixels and is
    interpolated bilinearly to every pixel (the residual field is smooth at that scale).
    The kriging variance is that of a NEW ATL08-like value (the nugget includes ATL08 noise); pass the
    ATL08 noise variance as atl08_noise_var to get the uncertainty of the lidar ground instead."""
    o = OBS["obs_mask"]
    rr, cc = np.nonzero(o)
    xy_obs = np.stack([cc, -rr], 1).astype(np.float64) * S["pixel_m"]    # metres, north up
    params, _ = fit_residual_variogram(xy_obs, resid)
    gr = np.arange(step_px // 2, S["H"], step_px); gc = np.arange(step_px // 2, S["W"], step_px)
    GR, GC = np.meshgrid(gr, gc, indexing="ij")
    xy_pred = np.stack([GC.ravel(), -GR.ravel()], 1).astype(np.float64) * S["pixel_m"]
    p, v = ordinary_kriging(xy_obs, resid, xy_pred, params, k=k)
    rows = (np.arange(S["H"]) - step_px // 2) / step_px
    cols = (np.arange(S["W"]) - step_px // 2) / step_px
    R, Cc = np.meshgrid(rows, cols, indexing="ij")
    up = lambda arr: map_coordinates(arr.reshape(GR.shape), [R, Cc], order=1, mode="nearest")
    v = np.maximum(up(v) - atl08_noise_var, 1e-6)
    return (mu4 + up(p)).astype(np.float32), np.sqrt(v).astype(np.float32)


def add_classical_baselines(P: dict, S: dict, OBS: dict, cell_km: float = 16.0, lam: float = 1.0,
                            kriging: bool = True, atl08_noise_var: float = 0.0) -> dict:
    """Adds B3 ("DSM + ATL08 offset"), B4 ("ridge regression") and, if kriging, B5 ("regression kriging")."""
    b = add_dsm_offset(P, S, OBS)
    mu4, resid = ridge_baseline(S, OBS, cell_km=cell_km, lam=lam)
    P["ridge regression"] = dict(mu=mu4, sigma=None)
    if kriging:
        mu5, s5 = regression_kriging(S, OBS, mu4, resid, atl08_noise_var=atl08_noise_var)
        P["regression kriging"] = dict(mu=mu5, sigma=s5)
    print(f"B3 global ATL08 offset {b:+.3f} m")
    return P


# =============================================================================
# co-registration of a DEM against the lidar ground (cell 112)
# =============================================================================
def gradients(z, pixel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """dz/dEast and dz/dNorth (m/m). Rows run southwards, so dz/dNorth = -dz/drow / pixel."""
    g_row, g_col = np.gradient(z, pixel_m)
    return g_col, -g_row


def shift_dem(dem, dx_m: float, dy_m: float, pixel_m: float) -> np.ndarray:
    """Move the DEM's content by (dx east, dy north) metres, bilinear, NaN kept as NaN."""
    ok = np.isfinite(dem)
    filled = np.where(ok, dem, np.nanmean(dem))
    s = (-dy_m / pixel_m, dx_m / pixel_m)                               # (rows, cols): north = up = fewer rows
    out = nd_shift(filled, s, order=1, mode="nearest")
    ok_s = nd_shift(ok.astype(np.float32), s, order=1, mode="constant", cval=0.0) > 0.999
    return np.where(ok_s, out, np.nan).astype(np.float32)


def _shift_stats(e) -> dict:
    m = error_metrics(e, crps=False)
    return dict(rmse=m["RMSE"], nmad=m["NMAD"], bias=m["ME"], n=int(np.size(e)))


def coregistration_check(S: dict, dem, name: str = "DEM", ref=None, canopy_max: float = 5.0,
                         slope_min_deg: float = 3.0, slope_max_deg: float = 60.0, n_iter: int = 4,
                         trim_k: float = 3.0) -> dict:
    """Horizontal shift of `dem` (on the site grid, e.g. FABDEM, or S["dsm"] for Copernicus) against `ref`
    (default: the lidar DTM = DSM + target), fitted on open-ground slopes as e = -dx*gE - dy*gN + c with
    outlier trimming, iterated. Returns dx, dy (m), shift in m and px, before / after stats, the corrected
    DEM, and the arrays the co-registration figure needs (key "plot")."""
    z_ref = (S["dsm"] + S["target"]) if ref is None else ref
    ref_ok = S["valid"] if ref is None else np.isfinite(ref)
    z_ref = np.where(ref_ok, z_ref, np.nan).astype(np.float64)
    gE, gN = gradients(z_ref, S["pixel_m"])                             # NaN next to gaps -> excluded below
    tan_s = np.hypot(gE, gN)
    slope = np.degrees(np.arctan(tan_s))

    # 1) pixels where the DEM should equal the ground: open (low canopy), on a slope (so a shift shows)
    base = (np.isfinite(gE) & np.isfinite(gN) & (S["canopy"] < canopy_max)
            & (slope >= slope_min_deg) & (slope <= slope_max_deg))
    print(f"{name}: {base.sum():,} open-ground slope pixels (canopy < {canopy_max:g} m, "
          f"slope {slope_min_deg:g}-{slope_max_deg:g} deg)")

    def fit(cur):
        """one least-squares fit of e = -dx*gE - dy*gN + c, with outliers trimmed at trim_k * NMAD."""
        e = cur - z_ref
        m = base & np.isfinite(e)
        for _ in range(3):                                              # trim, refit
            A = np.stack([-gE[m], -gN[m], np.ones(m.sum())], 1)
            coef, *_ = np.linalg.lstsq(A, e[m], rcond=None)
            r = e[m] - A @ coef
            nm = 1.4826 * np.median(np.abs(r - np.median(r)))
            keep = np.abs(r - np.median(r)) <= trim_k * max(nm, 1e-3)
            idx = np.flatnonzero(m)
            m = np.zeros_like(m); m.flat[idx[keep]] = True
        return coef[0], coef[1], m

    # 2) iterate: estimate, shift the DEM back by the estimate, estimate what is left
    dem = np.asarray(dem, np.float64)
    dx_tot, dy_tot, cur = 0.0, 0.0, dem
    for it in range(n_iter):
        dx, dy, _ = fit(cur)
        dx_tot += dx; dy_tot += dy
        cur = shift_dem(dem, -dx_tot, -dy_tot, S["pixel_m"])            # undo the total shift found so far
        if np.hypot(dx, dy) < 0.05:                                     # < 5 cm left: converged
            break

    # 3) before / after on the SAME pixels (open-ground slopes, finite in both)
    e0, e1 = dem - z_ref, cur - z_ref
    same = base & np.isfinite(e0) & np.isfinite(e1) & (np.abs(e0) < 50)
    before, after = _shift_stats(e0[same]), _shift_stats(e1[same])
    d = float(np.hypot(dx_tot, dy_tot))
    print(f"  shift of {name} relative to the reference: dx {dx_tot:+.2f} m east, dy {dy_tot:+.2f} m north "
          f"| |d| {d:.2f} m = {d / S['pixel_m']:.2f} px | {it + 1} iteration(s)")
    print(f"  open-ground slopes: RMSE {before['rmse']:.3f} -> {after['rmse']:.3f} m | "
          f"NMAD {before['nmad']:.3f} -> {after['nmad']:.3f} m | bias {before['bias']:+.3f} -> {after['bias']:+.3f} m")
    verdict = ("aligned (< 0.1 px)" if d < 0.1 * S["pixel_m"] else
               "small shift (0.1-0.3 px)" if d < 0.3 * S["pixel_m"] else
               "clear misregistration (>= 0.3 px): correct before comparing")
    print(f"  -> {verdict}")
    return dict(dx=dx_tot, dy=dy_tot, shift_m=d, shift_px=d / S["pixel_m"], before=before, after=after,
                corrected=cur.astype(np.float32),
                plot=dict(e0=e0, e1=e1, gE=gE, gN=gN, tan_s=tan_s, slope=slope, mask=same, name=name))


def registration_tag(path: str) -> str:
    """'Area' or 'Point' pixel registration of a GeoTIFF (also inside a zip: a /vsizip/... path)."""
    import rasterio
    with rasterio.open(path) as s:
        return s.tags().get("AREA_OR_POINT", "not set (GDAL assumes Area)")
