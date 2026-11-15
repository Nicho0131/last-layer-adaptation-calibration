"""
All figures. Numbers come from src/evaluation (evaluate, metrics, ablations); this module only draws.

Every figure function takes save=".../name.pdf" (300 dpi when saved) and calls plt.show().
Call paper_style() once before plotting for print-ready fonts and lines.

    style and helpers   paper_style, save_figure, site_style, models_in, extent_km, window_mask,
                        hillshade, map_aspect, log_ticks, show_map, draw_split
    training            plot_loss_curve
    Site A pixels       plot_coverage, plot_z_tails, plot_breakdown, plot_sigma_vs, show_outliers, plot_lai_evi
    Site A blocks       block_maps, block_sets_plot, block_error_plot, canopy_hist_by_set, variogram_plot
    one site (P, S, OBS) plot_distance, plot_canopy, plot_error_maps, plot_rasters, plot_canopy_error_2x2,
                        plot_block_maps, plot_block_scatter, plot_canopy_hist, plot_edge_profiles,
                        plot_anomaly_map, plot_coregistration
    pooled / studies    plot_by_slope, plot_calibration, plot_uncertainty_panel, plot_distance_variants,
                        plot_lambda_sweeps, plot_noise_distribution, plot_noise_conditions, plot_atl08

Site A edge profiles (cell 61): build P = {"M0 full": dict(mu=mu_full, sigma=sig_full)},
S = site_arrays(site)[0] with S["score"] = set_map(site) >= 1, OBS = dict(obs_mask=obs_full, obs_dh=...),
then plot_edge_profiles(P, S, OBS, ...).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.ndimage import binary_dilation
from scipy.stats import norm

from src.evaluation.ablations import NOISE_CLASS_EDGES, PIX, SEG, condition_label
from src.evaluation.evaluate import (CANOPY_EDGES, DIST_EDGES_KM, LEVELS, SLOPE_EDGES, breakdown_table, class_table,
                                     distance_to_obs_km, edge_profile_table, method_label, method_role, pooled_errors,
                                     score_mask, semivariogram, set_map, sharpness_table, sigma_distance_table,
                                     spread_skill_table, variogram_reach)
from src.evaluation.inference import predict_episode
from src.evaluation.metrics import coverage, error_tail_stats, qedges, residual_stats

PANEL = "abcdefghijklmnopqrstuvwxyz"
BOLD = dict(fontweight="bold")
C_MODEL, C_SIGMA, C_REF = "#2a78d6", "#eb6834", "#8a8a85"
C_HIGH, C_LOW, C_TRACK = "#d95926", "#2a78d6", "#0b0b0b"          # too high / too low / ATL08 pixels
SET_NAMES = ("train", "val", "test")
SET_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")                    # train blue, val orange, test green
ROLE_COLORS = {                                                    # colour follows the method role (cell 106)
    "uncorrected DSM": "#8a8a85", "FABDEM": "#6250d6", "FathomDEM": "#e87ba4",
    "network alone": "#eb6834", "+ global offset": "#1baf7a", "full": "#2a78d6",
    "DSM + ATL08 offset": "#008300", "ridge regression": "#eda100", "regression kriging": "#e34948",
}
OKABE_ITO = {                                                      # colour-blind-safe method colours (cells 132-134)
    "uncorrected DSM": "#999999", "FABDEM": "#E69F00", "M0 network alone": "#CC79A7", "M0 full": "#0072B2",
}
SITE_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
METHOD_LINES = {"M0 full": "-", "M0 network alone": "--"}


# =============================================================================
# style and helpers
# =============================================================================
def paper_style(font_size: int = 9, family: str = "serif") -> None:
    """Matplotlib settings for print: small serif text, thin spines, light grid, 300 dpi when saved.
    Call once before plotting (family="sans-serif" for a sans font)."""
    plt.rcParams.update({
        "font.family": family, "font.size": font_size, "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size, "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.5, "lines.linewidth": 1.6,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlelocation": "left",
        "mathtext.fontset": "dejavuserif",
    })


def save_figure(fig, save: str | None) -> None:
    if save:
        fig.savefig(save)
        print("saved", save)


def models_in(names) -> list[str]:
    """Model prefixes in method names, in order: ["M0", "M1"] from "M0 full", "M1 network alone", ..."""
    out = []
    for n in names:
        _, m = method_role(n)
        if m and m not in out:
            out.append(m)
    return out


def site_style(name: str, model_order) -> dict:
    """Colour from the role; a second model (e.g. M1) gets the same colour, dashed, another marker."""
    r, m = method_role(name)
    k = model_order.index(m) if m in model_order else 0
    ls = "--" if (name == "uncorrected DSM" or k == 1) else ("-." if k > 1 else "-")
    return dict(color=ROLE_COLORS.get(r, "#52514e"), ls=ls, marker="osD^v"[k % 5])


def extent_km(S: dict) -> tuple[float, float, float, float]:
    T = S["transform"]
    return T.c / 1000, (T.c + S["W"] * T.a) / 1000, (T.f + S["H"] * T.e) / 1000, T.f / 1000


def window_mask(S: dict, window_km) -> np.ndarray:
    """Pixels inside window_km = (east_min, east_max, north_min, north_max) in km (None = whole site)."""
    if window_km is None:
        return np.ones((S["H"], S["W"]), bool)
    T = S["transform"]
    e = (T.c + (np.arange(S["W"]) + 0.5) * T.a) / 1000
    n = (T.f + (np.arange(S["H"]) + 0.5) * T.e) / 1000
    e0, e1, n0, n1 = window_km
    return ((n >= n0) & (n <= n1))[:, None] & ((e >= e0) & (e <= e1))[None, :]


def hillshade(z, pixel_m: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Standard hillshade (0..1), used as a quiet background under maps."""
    z = np.where(np.isfinite(z), z, np.nanmean(z))
    gy, gx = np.gradient(z, pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def map_aspect(S: dict, window_km=None) -> float:
    """height / width of the mapped area, so map panels get the right shape."""
    if window_km is not None:
        return (window_km[3] - window_km[2]) / (window_km[1] - window_km[0])
    return S["H"] / S["W"]


def _row(n: int, panel_w: float = 3.1, panel_h: float = 3.1, extra_h: float = 0.0):
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h + extra_h), squeeze=False, constrained_layout=True)
    return fig, axes[0]


def _map_axes(ax, i: int, window_km=None, xlabel: bool = True):
    if xlabel:
        ax.set_xlabel("Easting (km)")
    if i == 0:
        ax.set_ylabel("Northing (km)")
    else:
        ax.tick_params(labelleft=False)
    if window_km is not None:
        ax.set_xlim(window_km[0], window_km[1]); ax.set_ylim(window_km[2], window_km[3])
    ax.set_aspect("equal")


def log_ticks(ax) -> None:
    """Readable labels (0.2, 0.5, 1, 2, 5 ...) on a logarithmic y axis."""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---------- show_map (cell 55) ----------
def _nice_step(span: float, n: int = 5) -> float:
    """A round step (1, 2, 2.5, 5 x 10^k) giving about n gridlines over `span`."""
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= 0.7 * raw:
            return m * mag


def _deg_label(v: float, step: float, pos: str, neg: str) -> str:
    dec = max(0, -math.floor(math.log10(step)) + (1 if step / 10 ** math.floor(math.log10(step)) == 2.5 else 0))
    return f"{abs(v):.{dec}f}°{pos if v >= 0 else neg}"


def _to_lonlat(crs):
    import pyproj
    tr = pyproj.Transformer.from_crs(pyproj.CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
    return lambda x, y: tr.transform(x, y)


def show_map(arr, transform, crs, title: str = "", cmap: str = "viridis", vmin=None, vmax=None, log: bool = False,
             cbar_label: str = "", coords: str = "en", n_lines: int = 5, max_px: int = 2000, figsize=(7, 9),
             background=None, bg_cmap: str = "gray", extra=None, save: str | None = None):
    """A raster in map coordinates with gridlines, coordinates on the edges only.
    coords "en": easting / northing in km; "lonlat": curved longitude / latitude graticule.
    log: logarithmic colour scale. background: array drawn underneath in grey (NaN pixels of arr show it).
    extra: function extra(ax) called before showing (e.g. to draw lines). Returns (fig, ax)."""
    h, w = arr.shape
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * w, top + transform.e * h
    k = max(1, math.ceil(max(h, w) / max_px))                 # thin large rasters for display only
    show = arr[::k, ::k]
    fig, ax = plt.subplots(figsize=figsize)
    finite = show[np.isfinite(show)]
    if log:
        lo = vmin if vmin is not None else np.percentile(finite[finite > 0], 1)
        hi = vmax if vmax is not None else np.percentile(finite, 99)
        nrm, vmin_, vmax_ = LogNorm(vmin=lo, vmax=hi), None, None
    else:
        nrm, vmin_, vmax_ = None, vmin, vmax
    if background is not None:
        bg = background[::k, ::k]
        ax.imshow(bg, extent=(left, right, bottom, top), origin="upper", cmap=bg_cmap, interpolation="nearest",
                  vmin=np.nanpercentile(bg, 2), vmax=np.nanpercentile(bg, 98))
    im = ax.imshow(show, extent=(left, right, bottom, top), origin="upper", cmap=cmap, norm=nrm, vmin=vmin_,
                   vmax=vmax_, interpolation="nearest")
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(title)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)
    geographic = getattr(crs, "is_geographic", False)
    if coords == "en" or geographic:
        sx, sy = _nice_step(right - left, n_lines), _nice_step(top - bottom, n_lines)
        xt = np.arange(math.ceil(left / sx) * sx, right, sx)
        yt = np.arange(math.ceil(bottom / sy) * sy, top, sy)
        ax.set_xticks(xt); ax.set_yticks(yt)
        if geographic:
            ax.set_xticklabels([_deg_label(v, sx, "E", "W") for v in xt])
            ax.set_yticklabels([_deg_label(v, sy, "N", "S") for v in yt])
        else:
            ax.set_xticklabels([f"{v / 1000:g}" for v in xt]); ax.set_yticklabels([f"{v / 1000:g}" for v in yt])
            ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
        ax.grid(True, color="black", lw=0.5, alpha=0.4)
    else:
        to_ll = _to_lonlat(crs)
        gx, gy = np.meshgrid(np.linspace(left, right, 200), np.linspace(bottom, top, 200))
        lon, lat = (np.asarray(v).reshape(gx.shape) for v in to_ll(gx.ravel(), gy.ravel()))
        s_lon, s_lat = _nice_step(lon.max() - lon.min(), n_lines), _nice_step(lat.max() - lat.min(), n_lines)
        lon_lv = np.arange(math.ceil(lon.min() / s_lon) * s_lon, lon.max(), s_lon)
        lat_lv = np.arange(math.ceil(lat.min() / s_lat) * s_lat, lat.max(), s_lat)
        ax.contour(gx, gy, lon, levels=lon_lv, colors="black", linewidths=0.5, alpha=0.4, linestyles="solid")
        ax.contour(gx, gy, lat, levels=lat_lv, colors="black", linewidths=0.5, alpha=0.4, linestyles="solid")
        o = np.argsort(lon[0])
        xt = np.interp(lon_lv, lon[0][o], gx[0][o], left=np.nan, right=np.nan)
        o = np.argsort(lat[:, 0])
        yt = np.interp(lat_lv, lat[:, 0][o], gy[:, 0][o], left=np.nan, right=np.nan)
        kx, ky = np.isfinite(xt), np.isfinite(yt)
        ax.set_xticks(xt[kx]); ax.set_xticklabels([_deg_label(v, s_lon, "E", "W") for v in lon_lv[kx]])
        ax.set_yticks(yt[ky]); ax.set_yticklabels([_deg_label(v, s_lat, "N", "S") for v in lat_lv[ky]])
    if extra is not None:
        extra(ax)
    plt.tight_layout()
    save_figure(fig, save); plt.show()
    return fig, ax


def draw_split(ax, site) -> None:
    """Draw the train / val / test boxes of Site A on a map in map coordinates (metres)."""
    T = site.transform
    for k, c in zip(SET_NAMES, SET_COLOURS):
        y0, x0, y1, x1 = site.boxes[k]
        xs = [T.c + x0 * T.a, T.c + x1 * T.a]
        ys = [T.f + y0 * T.e, T.f + y1 * T.e]
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], color=c, lw=2)


# =============================================================================
# training
# =============================================================================
def plot_loss_curve(history: dict, skip_steps: int = 150, save: str | None = None) -> None:
    """Training and validation loss after the warmup (cell 40). history: from train_meta or ck["history"]."""
    tr_s, tr_l = np.asarray(history["train_step"]), np.asarray(history["train_loss"])
    va_s, va_l = np.asarray(history["val_step"]), np.asarray(history["val_loss"])
    fig = plt.figure(figsize=(9, 4))
    plt.plot(tr_s[tr_s > skip_steps], tr_l[tr_s > skip_steps], label="training loss")
    plt.plot(va_s[va_s > skip_steps], va_l[va_s > skip_steps], "o-", label="validation loss")
    plt.xlabel("optimiser step"); plt.ylabel("loss"); plt.legend(); plt.title(f"Loss after step {skip_steps}")
    save_figure(fig, save); plt.show()


# =============================================================================
# Site A pixels (cells 47-50, 55-56)
# =============================================================================
def plot_coverage(err, sigma, levels=LEVELS, save: str | None = None) -> None:
    """Nominal vs observed coverage of mu ± k σ (cell 47; numbers: evaluate.coverage_table)."""
    obs = coverage(err, sigma, levels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.4, 1], [0.4, 1], color=C_REF, ls="--", lw=1, label="perfect calibration")
    ax.plot(levels, obs, color=C_MODEL, marker="o", ms=8, lw=2, label="model")
    ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1); ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage (interval μ ± k·σ)"); ax.set_ylabel("observed coverage")
    ax.grid(alpha=0.3); ax.legend(loc="upper left"); ax.set_title("Interval coverage")
    plt.tight_layout(); save_figure(fig, save); plt.show()


def plot_z_tails(err, sigma, save: str | None = None) -> None:
    """Histogram of z = error / σ against N(0, 1) on a log scale (cell 48, tail_plot)."""
    t = error_tail_stats(err, sigma)
    print(f"|z| > 3: {t['|z|>3']:.2%} of pixels (normal: {t['normal |z|>3']:.2%}), carrying "
          f"{t['sq. error in |z|>3']:.1%} of the squared error | largest 1% of |error| "
          f"(>= {t['top 1% |e| threshold (m)']:.1f} m) carry {t['sq. error in top 1%']:.1%}")
    z = np.asarray(err) / np.asarray(sigma)
    xs = np.linspace(-10, 10, 400)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(z, -10, 10), bins=np.linspace(-10, 10, 161), density=True, histtype="step", color=C_MODEL,
            lw=2, label="z = error / σ (clipped at ±10)")
    ax.plot(xs, norm.pdf(xs), color=C_REF, ls="--", lw=1.5, label="normal N(0, 1)")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_xlabel("standardised error z"); ax.set_ylabel("density (log)")
    ax.grid(alpha=0.3); ax.legend(); ax.set_title("Standardised errors vs normal")
    plt.tight_layout(); save_figure(fig, save); plt.show()


def plot_breakdown(Px: dict, key: str, label: str, edges=None, n_bins: int = 8, save: str | None = None):
    """Error by classes of Px[key] (collect_pixels output): RMSE, RMS σ and DSM RMSE (log scale), and the
    share of pixels vs share of squared error per class (cell 49). Returns the table."""
    df = breakdown_table(Px, key, edges, n_bins)
    pos = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    a1.plot(pos, df["RMSE"], color=C_MODEL, marker="o", ms=8, lw=2, label="model RMSE")
    a1.plot(pos, df["RMS_sigma"], color=C_SIGMA, marker="s", ms=8, lw=2, label="RMS predicted σ (matches RMSE if calibrated)")
    a1.plot(pos, df["DSM_RMSE"], color=C_REF, ls="--", lw=1.5, label="uncorrected DSM RMSE")
    a1.set_yscale("log"); log_ticks(a1)
    a1.set_ylabel("m (log scale)"); a1.grid(alpha=0.3); a1.legend(); a1.set_title(f"Error by {label}")
    w = 0.38
    a2.bar(pos - w / 2, df["share_px"] * 100, width=w - 0.02, color=C_REF, label="share of pixels")
    a2.bar(pos + w / 2, df["share_sq_err"] * 100, width=w - 0.02, color=C_MODEL, label="share of squared error")
    a2.set_ylabel("%"); a2.grid(alpha=0.3, axis="y")
    a2.set_ylim(0, 1.4 * max(df["share_px"].max(), df["share_sq_err"].max()) * 100)
    a2.legend(loc="upper left", ncol=2)
    a2.set_xticks(pos); a2.set_xticklabels(df.index, rotation=30, ha="right"); a2.set_xlabel(label)
    plt.tight_layout(); save_figure(fig, save); plt.show()
    return df


def plot_sigma_vs(Px: dict, key: str, label: str, n_bins: int = 12, save: str | None = None) -> None:
    """Predicted σ against Px[key] (hexbin, log), with RMS σ and RMSE per equal-count bin (cell 50)."""
    x, sig, err = Px[key], Px["sigma"], Px["err"]
    ok = np.isfinite(x)
    x, sig, err = x[ok], sig[ok], err[ok]
    edges = qedges(x, n_bins)
    b = np.digitize(x, edges) - 1
    c, rs, rm = [], [], []
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() > 50:
            c.append(np.median(x[m])); rs.append(np.sqrt(np.mean(sig[m] ** 2))); rm.append(np.sqrt(np.mean(err[m] ** 2)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hb = ax.hexbin(x, sig, gridsize=60, yscale="log", bins="log", cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="pixels (log)")
    ax.plot(c, rs, color=C_SIGMA, marker="s", ms=7, lw=2, label="RMS σ per bin")
    ax.plot(c, rm, color="black", marker="o", ms=6, lw=1.5, ls="--", label="model RMSE per bin")
    log_ticks(ax)
    ax.set_xlabel(label); ax.set_ylabel("m (log scale)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left"); ax.set_title(f"Predicted σ vs {label}")
    plt.tight_layout(); save_figure(fig, save); plt.show()


def show_outliers(model, ep: dict, site, z_thresh: float = 5.0, grow_px: int = 3, background: str = "canopy",
                  amp=None, save: str | None = None) -> np.ndarray:
    """Pixels with |error / σ| > z_thresh in one Site A episode, over a grey context layer (cell 56).
    orange: mu too high (ground left too high); blue: mu too low. grow_px only enlarges the marks.
    background: "dsm", "canopy", "lai", "evi" or "bldg". Returns z [h, w]."""
    from affine import Affine

    mu, sig = predict_episode(model, ep, amp=amp)
    tgt, val, obs = (ep[k][0, 0].cpu().numpy() for k in ("target", "valid", "obs_mask"))
    z = np.where(val.astype(bool) & ~obs.astype(bool), (mu - tgt) / sig, np.nan)
    hi, lo = z > z_thresh, z < -z_thresh
    print(f"|z| > {z_thresh:g}: {hi.sum():,} px too high, {lo.sum():,} px too low "
          f"(of {np.isfinite(z).sum():,} evaluated px)")
    cls = np.full(z.shape, np.nan)
    cls[binary_dilation(lo, iterations=grow_px)] = 0
    cls[binary_dilation(hi, iterations=grow_px)] = 1
    oy, ox = ep["origin"]
    h, w = z.shape
    T = site.transform * Affine.translation(ox, oy)
    bands = {"canopy": 4, "lai": 5, "evi": 6, "bldg": 7}             # 0-based bands of the GEE stack
    bg = site.dsm[oy:oy + h, ox:ox + w] if background == "dsm" else \
        site.extra["features"][bands[background], oy:oy + h, ox:ox + w]
    show_map(cls, T, site.crs, title=f"|z| > {z_thresh:g} over {background}",
             cmap=ListedColormap(["#2a78d6", "#eb6834"]), vmin=-0.5, vmax=1.5,
             cbar_label="blue: ground too low   |   orange: ground too high", background=bg, save=save)
    return z


def plot_lai_evi(feat, feat_ok, transform, crs, coords: str = "en", lai_band: int = 5, evi_band: int = 6) -> None:
    """LAI and EVI maps from the 8-band stack (0-based bands 5 and 6). Colour limits: 1st-99th percentile."""
    for name, b in (("LAI", lai_band), ("EVI", evi_band)):
        a = np.where(feat_ok[b], feat[b], np.nan)
        v = a[np.isfinite(a)]
        print(f"{name}: valid {feat_ok[b].mean() * 100:.1f}% | min {v.min():.3f} | median {np.median(v):.3f} | "
              f"max {v.max():.3f}")
        show_map(a, transform, crs, title=name, cmap="YlGn", cbar_label="m² / m²" if name == "LAI" else "",
                 coords=coords, vmin=np.percentile(v, 1), vmax=np.percentile(v, 99))


# =============================================================================
# Site A blocks (cells 58-59): df = evaluate.block_table(...)
# =============================================================================
def _site_extent_km(site) -> tuple[float, float, float, float]:
    T, (H, W) = site.transform, site.hw
    return T.c / 1000, (T.c + W * T.a) / 1000, (T.f + H * T.e) / 1000, T.f / 1000


def _set_outlines(ax, site, ext, color: str = "#0b0b0b") -> None:
    """Outline of each set (train / val / test), labelled just inside the top edge."""
    T, sets = site.transform, set_map(site)
    for i, name in enumerate(SET_NAMES):
        m = (sets == i).astype(float)
        if not m.any():
            continue
        ax.contour(m, levels=[0.5], extent=ext, origin="upper", colors=color, linewidths=0.8)
        cols = np.nonzero(m.any(0))[0]
        x_mid = (T.c + (cols.min() + cols.max() + 1) / 2 * T.a) / 1000
        ax.annotate(name.capitalize(), xy=(x_mid, 1), xycoords=("data", "axes fraction"), xytext=(0, -3),
                    textcoords="offset points", ha="center", va="top", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def block_maps(df, site, keys=(("canopy_frac", "Share of canopy > 5 m", "Greens", (0, 1), "share"),
                               ("rmse", "RMSE", "magma_r", None, "m"),
                               ("rms_sigma", "RMS predicted σ", "magma_r", None, "m")), save: str | None = None) -> None:
    """One row of block maps (one cell per block) with the train / val / test outlines.
    RMSE and σ share one colour scale so they can be compared directly."""
    B, (H, W), T = df.attrs["B"], site.hw, site.transform
    nby, nbx = math.ceil(H / B), math.ceil(W / B)
    ext = _site_extent_km(site)
    ext_blk = (T.c / 1000, (T.c + nbx * B * T.a) / 1000, (T.f + nby * B * T.e) / 1000, T.f / 1000)
    grids = {}
    for key, *_ in keys:
        g = np.full((nby, nbx), np.nan)
        for _, r in df.iterrows():
            g[int(r["by"]) // B, int(r["bx"]) // B] = r[key]
        grids[key] = g
    shared = [k for k, _, _, lim, _ in keys if lim is None]
    vmax = float(np.nanpercentile(np.concatenate([grids[k].ravel() for k in shared]), 98)) if shared else None
    fig, axes = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 2.9 * H / W + 1.0), constrained_layout=True)
    for i, (ax, (key, title, cmap, lim, unit)) in enumerate(zip(np.atleast_1d(axes), keys)):
        v0, v1 = lim if lim is not None else (0, vmax)
        im = ax.imshow(grids[key], cmap=cmap, vmin=v0, vmax=v1, extent=ext_blk, interpolation="nearest")
        _set_outlines(ax, site, ext)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30)
        cb.set_label(unit); cb.outline.set_linewidth(0.4)
        ax.set_title(f"({PANEL[i]}) {title}", fontsize=9, pad=4)
        _map_axes(ax, i)
    save_figure(fig, save); plt.show()


def block_sets_plot(df, save: str | None = None, seed: int = 0) -> None:
    """What each set contains, one dot per block: canopy share, building share, elevation, SD of true Δh."""
    keys = (("canopy_frac", "Canopy > 5 m (share)"), ("bldg_frac", "Buildings > 2 m (share)"),
            ("elev", "Mean elevation (m)"), ("dh_std", "SD of true Δh (m)"))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5), constrained_layout=True)
    for j, (ax, (key, label)) in enumerate(zip(axes, keys)):
        for i, col in enumerate(SET_COLOURS):
            v = df.loc[df["set"] == i, key].to_numpy(float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            ax.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=14, color=col, edgecolor="white", linewidth=0.5)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#0b0b0b", lw=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels([n.capitalize() for n in SET_NAMES]); ax.set_xlim(-0.6, 2.6)
        ax.set_title(f"({PANEL[j]}) {label}", fontsize=8.5); ax.grid(True, axis="y")
    save_figure(fig, save); plt.show()


def block_error_plot(df, save: str | None = None) -> None:
    """(a) block RMSE vs share of tall canopy; (b) block RMSE vs RMS σ (on the dashed line = honest σ)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    for i, col in enumerate(SET_COLOURS):
        r = df[df["set"] == i]
        kw = dict(s=18, color=col, edgecolor="white", linewidth=0.5)
        a1.scatter(r["canopy_frac"], r["rmse"], **kw)
        a2.scatter(r["rms_sigma"], r["rmse"], **kw)
    a1.set_xlabel("Share of canopy > 5 m in block"); a1.set_ylabel("Block RMSE (m)")
    a1.set_title("(a) Block error vs canopy cover"); a1.grid(True)
    lim = 1.08 * max(df["rmse"].max(), df["rms_sigma"].max())
    a2.plot([0, lim], [0, lim], color=C_REF, ls="--", lw=1)
    a2.set_xlim(0, lim); a2.set_ylim(0, lim); a2.set_aspect("equal")
    a2.set_xlabel("Block RMS predicted σ (m)"); a2.set_ylabel("Block RMSE (m)")
    a2.set_title("(b) Block calibration"); a2.grid(True)
    handles = [Line2D([], [], marker="o", ls="none", color=c, markeredgecolor="white", ms=5, label=n.capitalize())
               for n, c in zip(SET_NAMES, SET_COLOURS)]
    handles.append(Line2D([], [], color=C_REF, ls="--", lw=1, label="RMSE = RMS σ"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    save_figure(fig, save); plt.show()


def canopy_hist_by_set(site, max_m: float = 40, save: str | None = None) -> None:
    """Pixel-level canopy-height distribution of each set (log density), median in the legend."""
    feat, feat_ok = site.extra["features"], site.extra["features_valid"]
    canopy = np.where(feat_ok[4], feat[4], np.nan)
    sets, valid = set_map(site), site.valid.cpu().numpy()
    bins = np.linspace(0, max_m, 81)
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    for i, (name, col) in enumerate(zip(SET_NAMES, SET_COLOURS)):
        v = canopy[(sets == i) & valid]
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(np.clip(v, 0, max_m), bins=bins, density=True, histtype="step", lw=1.5, color=col,
                    label=f"{name.capitalize()} (median {np.median(v):.1f} m)")
    ax.set_yscale("log"); ax.set_xlabel("Canopy height (m)"); ax.set_ylabel("Density (log)"); ax.grid(True)
    fig.legend(loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, save); plt.show()


def variogram_plot(site, err_full, block_km: float = 8.0, buffer_km: float | None = None, rf_px: int = 13,
                   save: str | None = None) -> None:
    """Semivariogram of (left) the true correction and (right) the model error on held-out pixels
    (val + test), with the receptive field, buffer and block size marked (cell 59).
    err_full: [H, W] off-track errors of the full-map prediction (NaN elsewhere)."""
    sets = set_map(site)
    tgt = np.where(site.valid.cpu().numpy(), site.target.cpu().numpy(), np.nan)
    held = np.where(sets >= 1, err_full, np.nan)
    marks = [(rf_px * site.pixel_m / 1000, f"receptive field {rf_px} px"), (block_km, f"block {block_km:g} km")]
    if buffer_km is not None:
        marks.insert(1, (buffer_km, f"buffer {buffer_km:g} km"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, a, name in ((axes[0], tgt, "true correction Δh"), (axes[1], held, "model error (val + test)")):
        h, g = semivariogram(a, site.pixel_m)
        sill = np.nanvar(a)
        reach = variogram_reach(h, g, sill)
        ax.plot(h, g, color=C_MODEL, marker="o", ms=6, lw=2, label="semivariance")
        ax.axhline(sill, color=C_REF, ls="--", lw=1, label="variance (sill)")
        for x, lab in marks:
            ax.axvline(x, color="black", lw=0.8, ls=":")
            ax.text(x, 0.97, f"{lab} ", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("lag distance (km)"); ax.set_ylabel("semivariance (m²)")
        ax.set_title(f"{name}\n95% of variance reached at ~{reach:.2g} km"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
        print(f"{name}: variance {sill:.2f} m², 95% of it reached at ~{reach:.2g} km")
    plt.tight_layout(); save_figure(fig, save); plt.show()


# =============================================================================
# one site: P, S, OBS (cells 106, 116, 119, 132)
# =============================================================================
def plot_distance(P: dict, S: dict, OBS: dict, edges=DIST_EDGES_KM, methods=None, min_px: int = 200,
                  title: str | None = None, save: str | None = None) -> None:
    """(a) RMSE and (b) RMS predicted σ vs distance to the nearest ATL08 pixel, one line per method."""
    methods = [m for m in (list(P) if methods is None else methods) if m in P]
    order = models_in(P)
    multi = len(order) > 1
    sel = score_mask(S, OBS, P)
    b = np.digitize(distance_to_obs_km(OBS["obs_mask"], S["pixel_m"])[sel], np.asarray(edges)[1:-1])
    keep = [j for j in range(len(edges) - 1) if (b == j).sum() >= min_px]
    pos = np.arange(len(keep))
    fig, (a1, a2) = _row(2, panel_w=3.6, panel_h=2.8)
    handles = []
    for name in methods:
        p, st = P[name], site_style(name, order)
        err = (p["mu"] - S["target"])[sel]
        h, = a1.plot(pos, [np.sqrt(np.mean(err[b == j] ** 2)) for j in keep], ms=4.5, **st, label=method_label(name, multi))
        handles.append(h)
        if p["sigma"] is not None:
            s = p["sigma"][sel]
            a2.plot(pos, [np.sqrt(np.mean(s[b == j] ** 2)) for j in keep], ms=4.5, **st)
    for ax, t in ((a1, "(a) RMSE"), (a2, "(b) RMS predicted σ")):
        ax.set_xticks(pos); ax.set_xticklabels([f"{edges[j]:g}–{edges[j + 1]:g}" for j in keep])
        ax.set_xlabel("Distance to nearest ATL08 pixel (km)"); ax.set_ylabel("m"); ax.set_title(t); ax.grid(True, axis="y")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 3), frameon=False)
    if title:
        fig.suptitle(title)
    save_figure(fig, save); plt.show()


def plot_canopy(P: dict, S: dict, OBS: dict, edges=CANOPY_EDGES, methods=None, min_px: int = 200,
                title: str | None = None, save: str | None = None) -> None:
    """Grouped bars: RMSE per canopy-height class, one bar per method; pixels under each class."""
    methods = [m for m in (list(P) if methods is None else methods) if m in P]
    order = models_in(P)
    multi = len(order) > 1
    sel = score_mask(S, OBS, P)
    cls = np.digitize(S["canopy"][sel], np.asarray(edges)[1:-1])
    keep = [j for j in range(len(edges) - 1) if (cls == j).sum() >= min_px]
    fig, (ax,) = _row(1, panel_w=6.6, panel_h=2.9)
    w = 0.8 / len(methods)
    for i, name in enumerate(methods):
        err = (P[name]["mu"] - S["target"])[sel]
        st = site_style(name, order)
        ax.bar(np.arange(len(keep)) + (i - (len(methods) - 1) / 2) * w, [np.sqrt(np.mean(err[cls == j] ** 2)) for j in keep],
               width=w * 0.92, color=st["color"], hatch="///" if st["ls"] != "-" and name != "uncorrected DSM" else None,
               edgecolor="white", linewidth=0.4, label=method_label(name, multi))
    ax.set_xticks(np.arange(len(keep)))
    ax.set_xticklabels([f"{edges[j]:g}–{edges[j + 1]:g}".replace("inf", "∞") + f"\n(n = {int((cls == j).sum()) / 1e3:.0f}k)"
                        for j in keep])
    ax.set_xlabel("Canopy height (m)"); ax.set_ylabel("RMSE (m)"); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    fig.legend(loc="outside lower center", ncol=min(len(methods), 3), frameon=False)
    if title:
        fig.suptitle(title)
    save_figure(fig, save); plt.show()


def plot_error_maps(P: dict, S: dict, OBS: dict, methods=None, thresh: float = 2.0, window_km=None,
                    show_tracks: bool = True, title: str | None = None, save: str | None = None) -> None:
    """One row: per method, the pixels where the ground is > thresh too high (orange) or too low (blue)
    vs lidar, over a hillshade of the DSM; ATL08 pixels in black. window_km zooms every panel."""
    methods = list(P) if methods is None else methods
    skipped = [m for m in methods if m not in P]
    methods = [m for m in methods if m in P]
    if skipped:
        print(f"not in P, skipped: {skipped} | available: {list(P)}")
    multi = len(models_in(P)) > 1
    win = window_mask(S, window_km)
    sel = score_mask(S, OBS, P) & win
    ext = extent_km(S)
    hs = hillshade(np.where(S["dsm_ok"], S["dsm"], np.nan), S["pixel_m"])
    fig, axes = _row(len(methods), panel_w=3.0, panel_h=3.0 * map_aspect(S, window_km), extra_h=0.9)
    for i, (ax, name) in enumerate(zip(axes, methods)):
        err = np.where(sel, P[name]["mu"] - S["target"], np.nan)
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest", alpha=0.9)
        over = np.zeros((S["H"], S["W"], 4), np.float32)
        over[err > thresh] = mcolors.to_rgba(C_HIGH, 0.9)
        over[err < -thresh] = mcolors.to_rgba(C_LOW, 0.9)
        if show_tracks:
            over[OBS["obs_mask"] & win] = mcolors.to_rgba(C_TRACK, 1.0)
        ax.imshow(over, extent=ext, interpolation="nearest")
        ax.set_title(f"({PANEL[i]}) {method_label(name, multi)}\n|error| > {thresh:g} m: "
                     f"{np.mean(np.abs(err[sel]) > thresh):.1%}", fontsize=9)
        _map_axes(ax, i, window_km)
    handles = [Patch(color=C_HIGH, label=f"ground too high (> +{thresh:g} m)"),
               Patch(color=C_LOW, label=f"ground too low (< −{thresh:g} m)")]
    if show_tracks:
        handles.append(Patch(color=C_TRACK, label="ATL08 pixels"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title)
    save_figure(fig, save); plt.show()


def plot_rasters(P: dict, S: dict, OBS: dict, method: str, err_lim: float = 5.0, window_km=None,
                 title: str | None = None, save: str | None = None) -> None:
    """One method: (a) reference Δh, (b) predicted Δh (same scale), (c) predicted σ, (d) error (±err_lim m)."""
    p = P[method]
    ext = extent_km(S)
    tgt = np.where(S["valid"], S["target"], np.nan)
    lim = float(np.nanpercentile(np.abs(tgt), 99))
    panels = [("Reference Δh (lidar DTM − DSM)", tgt, "RdBu", -lim, lim),
              ("Predicted Δh", np.where(S["dsm_ok"], p["mu"], np.nan), "RdBu", -lim, lim)]
    if p["sigma"] is not None:
        sg = np.where(S["dsm_ok"], p["sigma"], np.nan)
        panels.append(("Predicted σ", sg, "Blues", 0, float(np.nanpercentile(sg, 99))))
    panels.append(("Error (predicted − reference)", np.where(S["valid"], p["mu"] - S["target"], np.nan),
                   "RdBu_r", -err_lim, err_lim))
    fig, axes = _row(len(panels), panel_w=3.0, panel_h=3.0 * map_aspect(S, window_km), extra_h=0.9)
    for i, (ax, (t, a, cmap, v0, v1)) in enumerate(zip(axes, panels)):
        im = ax.imshow(a, cmap=cmap, vmin=v0, vmax=v1, extent=ext, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=30)
        cb.set_label("m"); cb.outline.set_linewidth(0.4)
        ax.set_title(f"({PANEL[i]}) {t}", fontsize=9)
        ax.tick_params(labelbottom=False)
        _map_axes(ax, i, window_km, xlabel=False)
    if title:
        fig.suptitle(title)
    save_figure(fig, save); plt.show()


def plot_canopy_error_2x2(P: dict, S: dict, OBS: dict, ours: str = "M0 full", other: str = "FABDEM",
                          window_km=None, err_lim: float = 5.0, diff_lim: float = 3.0, canopy_max: float = 25.0,
                          save: str | None = None) -> None:
    """(a) canopy height, (b) |error| ours − |error| other, (c) error ours, (d) error other (cell 116)."""
    sel = score_mask(S, OBS, P) & window_mask(S, window_km)
    e_ours = np.where(sel, P[ours]["mu"] - S["target"], np.nan)
    e_oth = np.where(sel, P[other]["mu"] - S["target"], np.nan)
    diff = np.abs(e_ours) - np.abs(e_oth)                              # < 0: ours closer to the lidar
    better = np.mean(diff[sel] < 0)
    lo = method_label(other)
    panels = [("(a) Canopy height (ETH 2020)", np.where(S["dsm_ok"], S["canopy"], np.nan), "Greens", 0, canopy_max,
               "canopy height (m)"),
              (f"(b) |error|_model − |error|_{lo}\nModel closer to lidar at {better:.0%} of pixels", diff, "RdBu_r",
               -diff_lim, diff_lim, f"blue: ours better · red: {lo} better (m)"),
              (f"(c) Error, {method_label(ours)}", e_ours, "RdBu_r", -err_lim, err_lim, "predicted − lidar (m)"),
              (f"(d) Error, {lo}", e_oth, "RdBu_r", -err_lim, err_lim, "predicted − lidar (m)")]
    ext = extent_km(S)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0 * map_aspect(S, window_km) + 1.6), constrained_layout=True)
    for i, (ax, (t, arr, cmap, v0, v1, cb_lab)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(arr, cmap=cmap, vmin=v0, vmax=v1, extent=ext, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=35, extend="both")
        cb.set_label(cb_lab); cb.outline.set_linewidth(0.4)
        ax.set_title(t, fontsize=9)
        ax.tick_params(labelbottom=(i >= 2), labelleft=(i % 2 == 0))
        if i % 2 == 0:
            ax.set_ylabel("Northing (km)")
        if i >= 2:
            ax.set_xlabel("Easting (km)")
        if window_km is not None:
            ax.set_xlim(window_km[0], window_km[1]); ax.set_ylim(window_km[2], window_km[3])
        ax.set_aspect("equal")
    save_figure(fig, save); plt.show()


def _block_grid(df, S: dict, key: str) -> np.ndarray:
    B = df.attrs["B"]
    g = np.full((math.ceil(S["H"] / B), math.ceil(S["W"] / B)), np.nan)
    for _, r in df.iterrows():
        g[int(r["by"]) // B, int(r["bx"]) // B] = r[key]
    return g


def _block_extent(df, S: dict) -> tuple[float, float, float, float]:
    """Extent of the block grid in km (the last blocks may reach past the site edge)."""
    T, B = S["transform"], df.attrs["B"]
    nby, nbx = math.ceil(S["H"] / B), math.ceil(S["W"] / B)
    return T.c / 1000, (T.c + nbx * B * T.a) / 1000, (T.f + nby * B * T.e) / 1000, T.f / 1000


def plot_block_maps(df, S: dict, ours: str = "M0 full", other: str = "FABDEM", rmse_max: float | None = None,
                    diff_lim: float = 2.0, save: str | None = None) -> None:
    """df = evaluate.site_block_table: (a) tall-canopy share, (b) RMSE ours, (c) RMSE other,
    (d) RMSE ours − other (blue: ours better)."""
    has_other = other in df.attrs["methods"]
    ext = _block_extent(df, S)
    vmax = rmse_max or float(np.nanpercentile(df[[c for c in (f"rmse | {ours}", f"rmse | {other}") if c in df]].to_numpy(), 98))
    panels = [(f"Share of canopy > {df.attrs['tall_m']:g} m", _block_grid(df, S, "canopy_frac"), "Greens", 0, 1, "share"),
              (f"RMSE, {method_label(ours)}", _block_grid(df, S, f"rmse | {ours}"), "magma_r", 0, vmax, "m")]
    if has_other:
        panels += [(f"RMSE, {method_label(other)}", _block_grid(df, S, f"rmse | {other}"), "magma_r", 0, vmax, "m"),
                   (f"RMSE difference (Model − {method_label(other)})",
                    _block_grid(df, S, f"rmse | {ours}") - _block_grid(df, S, f"rmse | {other}"),
                    "RdBu_r", -diff_lim, diff_lim, "m (blue: ours better)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.9 * S["H"] / S["W"] + 1.0), squeeze=False,
                             constrained_layout=True)
    for i, (ax, (t, g, cmap, v0, v1, unit)) in enumerate(zip(axes[0], panels)):
        im = ax.imshow(g, cmap=cmap, vmin=v0, vmax=v1, extent=ext, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30)
        cb.set_label(unit); cb.outline.set_linewidth(0.4)
        ax.set_title(f"({PANEL[i]}) {t}", fontsize=9)
        ax.set_xlim(ext[0], S["transform"].c / 1000 + S["W"] * S["pixel_m"] / 1000)
        ax.set_ylim(S["transform"].f / 1000 - S["H"] * S["pixel_m"] / 1000, ext[3])
        _map_axes(ax, i)
    save_figure(fig, save); plt.show()


def plot_block_scatter(df, save: str | None = None) -> None:
    """df = evaluate.site_block_table: (a) block RMSE vs tall-canopy share per method; (b) block RMSE vs
    RMS σ for the methods with σ (on the dashed line = honest σ, above = overconfident)."""
    methods = df.attrs["methods"]
    order = models_in(methods)
    sig_methods = [m for m in methods if f"rms_sigma | {m}" in df]
    fig, axes = plt.subplots(1, 2 if sig_methods else 1, figsize=(7.2 if sig_methods else 3.8, 3.1), squeeze=False,
                             constrained_layout=True)
    a1 = axes[0][0]
    for name in methods:
        st = site_style(name, order)
        a1.scatter(df["canopy_frac"], df[f"rmse | {name}"], s=18, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=0.5, label=method_label(name))
    a1.set_xlabel(f"Share of canopy > {df.attrs['tall_m']:g} m in block"); a1.set_ylabel("Block RMSE (m)")
    a1.set_title("(a) Block error vs canopy cover"); a1.grid(True)
    h, l = a1.get_legend_handles_labels()
    if sig_methods:
        a2 = axes[0][1]
        lim = 1.1 * float(np.nanmax(np.concatenate([df[f"rmse | {m}"].to_numpy() for m in sig_methods] +
                                                   [df[f"rms_sigma | {m}"].to_numpy() for m in sig_methods])))
        a2.plot([0, lim], [0, lim], color=C_REF, ls="--", lw=1, label="RMSE = RMS σ")
        for name in sig_methods:
            st = site_style(name, order)
            a2.scatter(df[f"rms_sigma | {name}"], df[f"rmse | {name}"], s=18, color=st["color"], marker=st["marker"],
                       edgecolor="white", linewidth=0.5)
        a2.set_xlim(0, lim); a2.set_ylim(0, lim); a2.set_aspect("equal")
        a2.set_xlabel("Block RMS predicted σ (m)"); a2.set_ylabel("Block RMSE (m)")
        a2.set_title("(b) Block calibration"); a2.grid(True)
        h2, l2 = a2.get_legend_handles_labels()
        h, l = h + h2, l + l2
    fig.legend(h, l, loc="outside lower center", ncol=min(len(h), 3), frameon=False)
    save_figure(fig, save); plt.show()


def plot_canopy_hist(S: dict, OBS: dict, others: dict | None = None, max_m: float = 40, label: str = "Site",
                     save: str | None = None) -> None:
    """Density of canopy height over the site's scored pixels; others = {"Site A (training)": array, ...}
    adds the distribution of other areas to show any shift."""
    sel = S["valid"] & ~OBS["obs_mask"]
    series = {label: S["canopy"][sel]}
    for k, v in (others or {}).items():
        series[k] = np.asarray(v)[np.isfinite(v)]
    bins = np.linspace(0, max_m, 81)
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    for (k, v), c in zip(series.items(), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
        ax.hist(np.clip(v, 0, max_m), bins=bins, density=True, histtype="step", lw=1.6, color=c,
                label=f"{k} (median {np.median(v):.1f} m)")
    ax.set_yscale("log"); ax.set_xlabel("Canopy height (m)"); ax.set_ylabel("Density (log)"); ax.grid(True)
    fig.legend(loc="outside lower center", ncol=min(len(series), 2), frameon=False)
    save_figure(fig, save); plt.show()


def plot_edge_profiles(P: dict, S: dict, OBS: dict, layers, methods=("M0 full", "FABDEM"), max_px: int = 30,
                       rf_px: int = 13, min_px: int = 200, save: str | None = None) -> None:
    """layers: [(array, thresh_m, name), ...], e.g. [(S["bldg"], 2.0, "building"), (S["canopy"], 5.0, "canopy")].
    Per edge type: RMSE (+ RMS σ, dotted) and mean error against the signed distance to the edge (px, > 0
    inside). Grey band: within ±rf_px//2 px of the edge the receptive field sees both sides."""
    methods = [m for m in methods if m in P]
    order = models_in(methods)
    half = rf_px // 2
    fig, axes = plt.subplots(len(layers), 2, figsize=(7.4, 2.6 * len(layers) + 0.7), constrained_layout=True, squeeze=False)
    handles = {}
    for r, (layer, thresh_m, name) in enumerate(layers):
        df = edge_profile_table(P, S, OBS, layer, thresh_m, methods, max_px, min_px)
        ks = df.index.to_numpy()
        a1, a2 = axes[r]
        for ax in (a1, a2):
            ax.axvspan(-half, half, color=C_REF, alpha=0.15, lw=0)
            ax.axvline(0, color="#1a1a19", lw=0.8)
            ax.set_xlabel(f"Distance to {name} edge (px, > 0 inside)"); ax.set_ylabel("m"); ax.grid(True, axis="y")
            ax.set_xlim(min(ks.min(), -half) - 2, max(ks.max(), half) + 2)
        a2.axhline(0, color="#1a1a19", lw=0.8)
        for m in methods:
            st, lab = site_style(m, order), method_label(m)
            h, = a1.plot(ks, df[f"RMSE | {m}"], color=st["color"], marker="o", ms=2.5, lw=1.4)
            a2.plot(ks, df[f"ME | {m}"], color=st["color"], marker="o", ms=2.5, lw=1.4)
            handles.setdefault(f"{lab} RMSE", h)
            if f"RMS σ | {m}" in df:
                h2, = a1.plot(ks, df[f"RMS σ | {m}"], color=st["color"], ls=":", lw=1.6, alpha=0.8)
                handles.setdefault(f"{lab} RMS σ", h2)
        cap = name.capitalize()
        a1.set_title(f"({PANEL[2 * r]}) {cap} edges (> {thresh_m:g} m): RMSE", fontsize=9)
        a2.set_title(f"({PANEL[2 * r + 1]}) {cap} edges: mean error (> 0: ground too high)", fontsize=9)
    keys = list(handles)
    h = [handles[k] for k in keys] + [Patch(color=C_REF, alpha=0.3)]
    fig.legend(h, keys + [f"receptive field ({rf_px}×{rf_px}) spans the edge"], loc="outside lower center", ncol=2,
               frameon=False)
    save_figure(fig, save); plt.show()


def plot_anomaly_map(S: dict, extra: dict, window_km=None, dh_lim: float = 10.0, save: str | None = None) -> None:
    """extra from evaluate.anomaly_tables: (a) the canopy 0-2 m pixels coloured by Δh over a hillshade;
    (b) the canopy map of the same area (cell 132)."""
    ext = extent_km(S)
    hs = hillshade(np.where(S["dsm_ok"], S["dsm"], np.nan), S["pixel_m"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6 * map_aspect(S, window_km) + 0.9), constrained_layout=True)
    a1.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest", alpha=0.8)
    im = a1.imshow(np.where(extra["cls"], S["target"], np.nan), cmap="RdBu", vmin=-dh_lim, vmax=dh_lim, extent=ext,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=a1, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30, extend="both")
    cb.set_label("Δh = lidar − DSM (m)", **BOLD)
    a1.set_title("(a) Canopy < 2 m pixels, coloured by Δh", fontsize=9, **BOLD)
    im2 = a2.imshow(np.where(S["dsm_ok"], S["canopy"], np.nan), cmap="Greens", vmin=0, vmax=25, extent=ext,
                    interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=a2, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30, extend="max")
    cb2.set_label("ETH canopy height (m)", **BOLD)
    a2.set_title("(b) Canopy height (ETH 2020)", fontsize=9, **BOLD)
    for i, ax in enumerate((a1, a2)):
        _map_axes(ax, i, window_km)
    save_figure(fig, save); plt.show()


def plot_coregistration(result: dict, save: str | None = None) -> None:
    """result = baselines.coregistration_check(...): (a) error / tan(slope) vs upslope direction before and
    after, with the fitted shift; (b) NMAD by slope class before and after (cell 112)."""
    q = result["plot"]
    e0, e1, gE, gN, tan_s, slope, m = (q[k] for k in ("e0", "e1", "gE", "gN", "tan_s", "slope", "mask"))
    dx, dy = result["dx"], result["dy"]
    up = (np.degrees(np.arctan2(gE[m], gN[m])) + 360) % 360           # upslope direction, deg clockwise from N
    t = np.where(tan_s[m] > 0.05, tan_s[m], np.nan)
    bins = np.arange(0, 361, 15)
    k = np.digitize(up, bins) - 1
    cen = bins[:-1] + 7.5
    med = lambda e: [np.nanmedian((e[m] / t)[k == j]) if (k == j).sum() > 50 else np.nan for j in range(len(cen))]
    th = np.radians(np.arange(0, 361))
    s_edges = (3, 5, 10, 15, 20, 30, 60)
    sk = np.digitize(slope[m], s_edges[1:-1])
    nm = lambda e: [residual_stats(e[m][sk == j])["robust_sd"] if (sk == j).sum() > 200 else np.nan
                    for j in range(len(s_edges) - 1)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    a1.axhline(0, color=C_REF, lw=0.6)
    a1.plot(cen, med(e0), "o", color="#eb6834", ms=4, label="before")
    a1.plot(np.degrees(th), -(dx * np.sin(th) + dy * np.cos(th)), "-", color="#eb6834", lw=1.2, label="fitted shift")
    a1.plot(cen, med(e1), "s", color="#2a78d6", ms=4, label="after correction")
    a1.set_xticks(range(0, 361, 90)); a1.set_xlim(0, 360)
    a1.set_xlabel("Upslope direction (° from north)"); a1.set_ylabel("error / tan(slope) (m)")
    a1.set_title("(a) Error vs terrain aspect"); a1.grid(True, axis="y")
    pos = np.arange(len(s_edges) - 1)
    a2.plot(pos, nm(e0), "o-", color="#eb6834", ms=4, label="before")
    a2.plot(pos, nm(e1), "s-", color="#2a78d6", ms=4, label="after correction")
    a2.set_xticks(pos); a2.set_xticklabels([f"{s_edges[j]}–{s_edges[j + 1]}" for j in pos])
    a2.set_xlabel("Slope (°)"); a2.set_ylabel("NMAD (m)"); a2.set_title("(b) Error spread vs slope"); a2.grid(True, axis="y")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, save); plt.show()


# =============================================================================
# pooled items and studies (cells 92, 127/128, 129, 131-134)
# =============================================================================
def plot_by_slope(items, methods, edges=SLOPE_EDGES, show_bias: bool = True, save: str | None = None):
    """Grouped bars per slope class: (a) RMSE, (b) mean error (if show_bias); pixel share under each class.
    items: [(P, S, OBS)] with S["slope"]. Returns the table (evaluate.class_table)."""
    df = class_table(items, "slope", edges, methods)
    n_pan = 2 if show_bias else 1
    fig, axes = plt.subplots(1, n_pan, figsize=(3.8 * n_pan, 2.9), constrained_layout=True, squeeze=False)
    x, w = np.arange(len(df)), 0.8 / len(methods)
    for stat, ax in zip(("RMSE", "Bias")[:n_pan], axes[0]):
        for i, m in enumerate(methods):
            lab = method_label(m)
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, df[f"{stat} | {lab}"], width=w * 0.92,
                   color=OKABE_ITO.get(m, "#52514e"), edgecolor="white", linewidth=0.4, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{s}\n({sh:.0%})" for s, sh in zip(df.index, df["Share"])])
        ax.set_xlabel("Slope (share of pixels)", **BOLD); ax.grid(True, axis="y"); ax.set_axisbelow(True)
        if stat == "Bias":
            ax.axhline(0, color="#52514e", lw=0.6)
            ax.set_ylabel("ME (m; > 0: undercorrection)", **BOLD); ax.set_title("(b) Bias", fontsize=9, **BOLD)
        else:
            ax.set_ylabel("RMSE (m)", **BOLD); ax.set_title("(a) RMSE" if show_bias else "RMSE by slope", fontsize=9, **BOLD)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(len(methods), 3), frameon=False, prop={"weight": "bold"})
    save_figure(fig, save); plt.show()
    return df


def plot_calibration(sites: dict, methods=("M0 full",), z_lim: float = 8.0, band_pp: float = 3.0, z_tail: float = 3.0,
                     save: str | None = None) -> None:
    """(a) reliability diagram per site (and method), ±band_pp shaded; (b) histogram of z on a log scale with
    N(0, 1). sites: {"Site A (simulated ATL08)": items, ...} (numbers: evaluate.calibration_table)."""
    p = np.r_[np.arange(0.05, 0.951, 0.05), 0.99]
    bins = np.linspace(-z_lim, z_lim, 81)
    bw, zc = bins[1] - bins[0], (bins[:-1] + bins[1:]) / 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4), constrained_layout=True)
    a1.fill_between([0, 1], [-band_pp / 100, 1 - band_pp / 100], [band_pp / 100, 1 + band_pp / 100], color=C_REF,
                    alpha=0.18, lw=0)
    a1.plot([0, 1], [0, 1], color="#52514e", lw=0.8)
    xx = np.linspace(-z_lim, z_lim, 400)
    a2.plot(xx, norm.pdf(xx), color="#1a1a19", ls=":", lw=1.3)
    handles = []
    for k, (site, items) in enumerate(sites.items()):
        col = SITE_COLOURS[k % len(SITE_COLOURS)]
        for m in methods:
            ls = METHOD_LINES.get(m, "-")
            e, s = pooled_errors(items, m)
            z = e / s
            a1.plot(p, coverage(e, s, p), color=col, ls=ls, marker="o", ms=3, lw=1.5)
            h, _ = np.histogram(z, bins=bins)
            a2.step(zc, np.where(h > 0, h / (len(z) * bw), np.nan), where="mid", color=col, ls=ls, lw=1.4)
            name = site if len(methods) == 1 else f"{site}, {method_label(m)}"
            handles.append(Line2D([], [], color=col, ls=ls, marker="o", ms=3, lw=1.5,
                                  label=f"{name}  (|z| > {z_tail:g}: {np.mean(np.abs(z) > z_tail):.1%})"))
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_aspect("equal"); a1.grid(True)
    a1.set_xlabel("Nominal coverage of central interval", **BOLD); a1.set_ylabel("Observed coverage", **BOLD)
    a1.set_title("(a) Reliability diagram", fontsize=9, **BOLD)
    a2.set_yscale("log"); a2.set_xlim(-z_lim, z_lim); a2.set_ylim(1e-5, 1.0); a2.grid(True, axis="y")
    a2.set_xlabel("Standardised error z = (μ − Y) / σ", **BOLD); a2.set_ylabel("Density (log)", **BOLD)
    a2.set_title("(b) Standardised errors", fontsize=9, **BOLD)
    handles += [Line2D([], [], color="#1a1a19", ls=":", lw=1.3, label=f"N(0, 1) (|z| > {z_tail:g}: {2 * norm.sf(z_tail):.1%})"),
                Patch(color=C_REF, alpha=0.35, label=f"±{band_pp:g} percentage points")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2 if len(handles) <= 4 else 3, frameon=False,
               prop={"weight": "bold"})
    save_figure(fig, save); plt.show()


def _draw_sigma_distance(a1, a2, df):
    methods = list(dict.fromkeys(df["key"]))
    for m in methods:
        r = df[df["key"] == m]
        x, c = np.arange(len(r)), OKABE_ITO.get(m, "#52514e")
        a1.plot(x, r["RMS σ (m)"], color=c, marker="o", ms=4, lw=1.6)
        a1.plot(x, r["RMSE (m)"], color=c, marker="s", ms=4, mfc="white", ls="--", lw=1.2)
        a2.plot(x, r["Cov. 95%"], color=c, marker="o", ms=4, lw=1.6)
    ticks = df[df["key"] == methods[-1]]
    for ax in (a1, a2):
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_xticklabels([f"{t}\n({sh:.0%})" for t, sh in zip(ticks["Distance"].str.replace(" km", ""), ticks["Share"])])
        ax.set_xlabel("Distance to nearest ATL08 px (km)", **BOLD); ax.grid(True, axis="y")
    a1.set_ylabel("m", **BOLD); a1.set_title("(a) Predicted σ and observed error", fontsize=9, **BOLD)
    a2.axhline(0.95, color="#52514e", lw=0.8, ls=":")
    a2.set_ylabel("Observed coverage of 95% int.", **BOLD); a2.set_title("(b) 95 % coverage", fontsize=9, **BOLD)
    handles = [Line2D([], [], color=OKABE_ITO.get(m, "#52514e"), lw=1.6, label=method_label(m)) for m in methods]
    return handles + [Line2D([], [], color="#52514e", marker="o", ms=4, lw=1.6, label="RMS σ"),
                      Line2D([], [], color="#52514e", marker="s", ms=4, mfc="white", ls="--", lw=1.2, label="RMSE"),
                      Line2D([], [], color="#52514e", ls=":", lw=0.8, label="Nominal 95 %")]


def _draw_spread_skill(ax, df):
    sites, methods = list(dict.fromkeys(df["Site"])), list(dict.fromkeys(df["key"]))
    handles = []
    for k, site in enumerate(sites):
        col = SITE_COLOURS[k % len(SITE_COLOURS)]
        for m in methods:
            r = df[(df["Site"] == site) & (df["key"] == m)]
            ax.plot(r["RMS σ (m)"], r["RMSE (m)"], color=col, ls=METHOD_LINES.get(m, "-"), marker="o", ms=4, lw=1.5)
        handles.append(Line2D([], [], color=col, marker="o", ms=4, lw=1.5, label=site))
    lim = 1.08 * max(df["RMS σ (m)"].max(), df["RMSE (m)"].max())
    ax.plot([0, lim], [0, lim], color="#52514e", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("RMS predicted σ in bin (m)", **BOLD); ax.set_ylabel("Observed RMSE in bin (m)", **BOLD)
    ax.set_title(f"(c) Spread–skill ({int(df['Bin'].max())} σ bins)", fontsize=9, **BOLD)
    for k, site in enumerate(sites):
        rho = df.loc[(df["Site"] == site) & (df["key"] == methods[-1]), "rho"].iloc[0]
        ax.text(0.04, 0.96 - 0.08 * k, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top",
                color=SITE_COLOURS[k % len(SITE_COLOURS)], fontsize=8, **BOLD)
    if len(methods) > 1:
        handles += [Line2D([], [], color="#52514e", ls=METHOD_LINES.get(m, "-"), lw=1.5, label=method_label(m))
                    for m in methods]
    handles.append(Line2D([], [], color="#52514e", lw=0.8, label="RMSE = RMS σ"))
    return handles


def _draw_sharpness(ax, df):
    markers = {"M0 network alone": "s", "M0 full": "o"}
    for k, site in enumerate(df["Site"].unique()):
        col = SITE_COLOURS[k % len(SITE_COLOURS)]
        r = df[df["Site"] == site].set_index("key")
        if {"M0 network alone", "M0 full"} <= set(r.index):
            a, b = r.loc["M0 network alone"], r.loc["M0 full"]
            ax.annotate("", xy=(b["RMS σ (m)"], b["Cov. 95%"]), xytext=(a["RMS σ (m)"], a["Cov. 95%"]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=5, shrinkB=5))
        for key, row in r.iterrows():
            ax.plot(row["RMS σ (m)"], row["Cov. 95%"], marker=markers.get(key, "o"), ms=7, color=col,
                    mfc=col if key == "M0 full" else "white", mew=1.5, ls="none")
    ax.axhline(0.95, color="#52514e", lw=0.8, ls=":")
    ax.set_ylim(min(0.85, df["Cov. 95%"].min() - 0.02), 1.0); ax.set_xlim(0, 1.1 * df["RMS σ (m)"].max()); ax.grid(True)
    ax.set_xlabel("RMS predicted σ (m)  ← sharper", **BOLD); ax.set_ylabel("Observed coverage of 95 % int.", **BOLD)
    ax.set_title("(d) Calibration and sharpness", fontsize=9, **BOLD)
    return [Line2D([], [], color="#52514e", marker="s", mfc="white", mew=1.5, ls="none", ms=6,
                   label=method_label("M0 network alone")),
            Line2D([], [], color="#52514e", marker="o", ls="none", ms=6, label=method_label("M0 full")),
            Line2D([], [], color="#52514e", ls=":", lw=0.8, label="Nominal 95 %")]


def plot_uncertainty_panel(items_a, sites: dict, spread_methods=("M0 full",), n_bins: int = 10,
                           save: str | None = None):
    """(a, b) Site A: σ, RMSE and coverage against distance to ATL08; (c) spread-skill per site;
    (d) coverage against RMS σ per site (cell 134). Returns the three tables."""
    T_dist = sigma_distance_table(items_a)
    T_ss = spread_skill_table(sites, spread_methods, n_bins)
    T_sh = sharpness_table(sites)
    fig = plt.figure(figsize=(7.4, 7.6), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.25], hspace=0.04)
    a1, a2 = top.subplots(1, 2)
    c1, c2 = bot.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    h_top = _draw_sigma_distance(a1, a2, T_dist)
    h_c = _draw_spread_skill(c1, T_ss)
    h_d = _draw_sharpness(c2, T_sh)
    top.legend(handles=h_top, loc="outside lower center", ncol=3, prop={"weight": "bold"}, frameon=False)
    seen, h_bot = set(), []
    for h in h_c[:len(sites)] + h_d + h_c[len(sites):]:
        if h.get_label() not in seen:
            seen.add(h.get_label()); h_bot.append(h)
    bot.legend(handles=h_bot, loc="outside lower center", ncol=3, prop={"weight": "bold"}, frameon=False)
    save_figure(fig, save); plt.show()
    return T_dist, T_ss, T_sh


# ---------- ablation variants (cell 129) ----------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300")
REF_STYLE = {"Uncorrected DSM": dict(color="#8a8a85", ls="--", marker=None),
             "Network only": dict(color="#52514e", ls=":", marker=None)}
FIXED_VARIANT_COLOURS = {
    "Network + ATL08 adaptation": "#2a78d6", "Network + ATL08 offset": "#1baf7a", "ATL08 mean only": "#e87ba4",
    "1 km level only": "#eb6834", "4 km level only": "#eda100", "16 km level only": "#6250d6", "Bilinear read": "#6250d6",
}
GAUSS_COLOURS = ("#eb6834", "#eda100", "#e87ba4")


def variant_style(names) -> dict:
    """Colour per ablation variant (fixed for named variants, so it is the same in every figure)."""
    st, k, g = {}, 0, 0
    marks = "osD^vP*"
    for i, n in enumerate(names):
        if n in REF_STYLE:
            st[n] = REF_STYLE[n]
            continue
        if n in FIXED_VARIANT_COLOURS:
            col = FIXED_VARIANT_COLOURS[n]
        elif n.startswith("Gaussian read"):
            col = GAUSS_COLOURS[g % len(GAUSS_COLOURS)]; g += 1
        else:
            col = SERIES[k % len(SERIES)]; k += 1
        st[n] = dict(color=col, ls="-", marker=marks[i % len(marks)])
    return st


def plot_distance_variants(P: dict, S: dict, OBS: dict, names=None, edges=(0, 0.25, 0.5, 1, 2, 4, np.inf),
                           min_px: int = 200, save: str | None = None) -> None:
    """P from an ablations.ablation_* call: (a) RMSE and (b) RMS σ vs distance to ATL08, one line per variant."""
    names = list(P) if names is None else names
    sel = score_mask(S, OBS, {n: P[n] for n in names})
    b = np.digitize(distance_to_obs_km(OBS["obs_mask"], S["pixel_m"])[sel], np.asarray(edges)[1:-1])
    keep = [j for j in range(len(edges) - 1) if (b == j).sum() >= min_px]
    pos, st = np.arange(len(keep)), variant_style(names)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    handles = []
    for n in names:
        e = (P[n]["mu"] - S["target"])[sel]
        kw = dict(color=st[n]["color"], ls=st[n]["ls"], marker=st[n]["marker"], ms=4, lw=1.4)
        h, = a1.plot(pos, [np.sqrt(np.mean(e[b == j] ** 2)) for j in keep], label=n, **kw)
        handles.append(h)
        if P[n]["sigma"] is not None:
            s = P[n]["sigma"][sel].astype(np.float64)
            a2.plot(pos, [np.sqrt(np.mean(s[b == j] ** 2)) for j in keep], **kw)
    for ax, t in ((a1, "(a) RMSE"), (a2, "(b) RMS predicted σ")):
        ax.set_xticks(pos); ax.set_xticklabels([f"{edges[j]:g}–{edges[j + 1]:g}".replace("inf", "∞") for j in keep])
        ax.set_xlabel("Distance to nearest ATL08 pixel (km)"); ax.set_ylabel("m"); ax.set_title(t); ax.grid(True, axis="y")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 3), frameon=False)
    save_figure(fig, save); plt.show()


# ---------- lambda sweeps (cell 131) ----------
DIST_COLOURS = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00")


def plot_lambda_sweeps(sweeps, ref_rmse: float | None = None, ncols: int = 2, save: str | None = None) -> None:
    """sweeps: DataFrames from ablations.lambda_sweep_site. Per panel: pooled RMSE (black) and RMSE per
    distance class against the λ multiplier (log); dashed = trained λ; dotted grey = ref_rmse (network only)."""
    n = len(sweeps)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.55 * nrows + 0.7), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    handles = {}
    for i, df in enumerate(sweeps):
        ax = axes[i // ncols][i % ncols]
        x = df.index.to_numpy(float)
        ax.axvline(1, color=C_REF, ls="--", lw=0.9, zorder=1)
        handles.setdefault("Trained λ (× 1)", Line2D([], [], color=C_REF, ls="--", lw=0.9))
        if ref_rmse is not None:
            ax.axhline(ref_rmse, color="#52514e", ls=":", lw=1.2, zorder=1)
            handles.setdefault("Network only (B6)", Line2D([], [], color="#52514e", ls=":", lw=1.2))
        for c, l in zip(DIST_COLOURS, df.attrs["dist_labels"]):
            h, = ax.plot(x, df[f"RMSE {l}"], color=c, marker="o", ms=3.5, lw=1.3, zorder=2)
            handles.setdefault(f"{l} from ATL08", h)
        h, = ax.plot(x, df["RMSE"], color="#000000", marker="s", ms=4, lw=2.0, zorder=3)
        handles.setdefault("All pixels", h)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off(); ax.grid(True, axis="y"); ax.set_axisbelow(True)
        ax.set_title(f"({PANEL[i]}) λ scaled on {df.attrs['which']}", fontsize=9, **BOLD)
        if i // ncols == nrows - 1 or i + ncols >= n:
            ax.set_xlabel("λ multiplier (1 = trained)", **BOLD); ax.tick_params(labelbottom=True)
        if i % ncols == 0:
            ax.set_ylabel("RMSE (m)", **BOLD)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    order = ["All pixels"] + [k for k in handles if k.endswith("from ATL08")] + \
            [k for k in ("Network only (B6)", "Trained λ (× 1)") if k in handles]
    fig.legend([handles[k] for k in order], order, loc="outside lower center", ncol=3, prop={"weight": "bold"},
               frameon=False)
    save_figure(fig, save); plt.show()


# ---------- ATL08 noise study (cell 128) ----------
SIG_EPS, EPS = r"$\sigma_\varepsilon$", r"$\varepsilon$"
COND_SHORT = {"No ATL08 (network only)": "No ATL08", "Real ATL08": "Real", "Lidar + empirical ATL08 ε": "Empirical ε",
              "Lidar, no noise": "No noise"}
COND_COLORS = {"No ATL08 (network only)": "#999999", "Real ATL08": "#0072B2", "Lidar + Gaussian ε": "#D55E00",
               "Lidar + empirical ATL08 ε": "#009E73", "Lidar, no noise": "#E69F00"}
SCALE_COLORS = {SEG: "#CC79A7", PIX: "#0072B2"}


def _cond_color(lab: str) -> str:
    return COND_COLORS.get("Lidar + Gaussian ε" if lab.startswith("Lidar + Gaussian") else lab, "#52514e")


def _tex(lab: str) -> str:
    """Table label -> figure label with math symbols."""
    return lab.replace("σ_ε", SIG_EPS).replace("ε", EPS) if "σ_ε" in lab else lab.replace("ε", EPS)


def plot_noise_distribution(real_b, canopy_b, cal_diff, cal_canopy, sigmas=(0.3, 1.0), lim: float = 8.0,
                            edges=NOISE_CLASS_EDGES, min_n: int = 50, save: str | None = None) -> None:
    """(a) density of ε = ATL08 − lidar (log), per segment and per 30 m pixel, with the Gaussian training
    noise; (b) robust SD and RMS of ε per canopy class against the constant σ_ε."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    bins, x = np.linspace(-lim, lim, 161), np.linspace(-lim, lim, 400)
    handles = []
    for scale, d in ((SEG, cal_diff), (PIX, real_b)):
        d = np.asarray(d, np.float64)
        inside = np.abs(d) <= lim
        h = a1.hist(d[inside], bins=bins, weights=np.full(inside.sum(), 1.0 / (d.size * (bins[1] - bins[0]))),
                    histtype="step", lw=1.4, color=SCALE_COLORS[scale], label=scale)[2][0]
        handles.append(h)
        print(f"{scale}: {np.mean(~inside):.2%} of ε beyond ±{lim:g} m (not drawn)")
    for s, ls in zip(sigmas, (":", "--", "-.")):
        h, = a1.plot(x, norm.pdf(x, 0, s), color="#0b0b0b", ls=ls, lw=1.0, label=f"Gaussian, {SIG_EPS} = {s:g} m")
        handles.append(h)
    a1.set_yscale("log"); a1.set_ylim(1e-4, 3); a1.set_xlim(-lim, lim)
    a1.set_xlabel(f"{EPS} = ATL08 − lidar ground (m)", **BOLD); a1.set_ylabel("Density (log)", **BOLD)
    a1.set_title(f"(a) Distribution of {EPS}", **BOLD); a1.grid(True, axis="y")
    labels = [f"{edges[j]:g}–{edges[j + 1]:g}".replace("inf", "∞") for j in range(len(edges) - 1)]
    pos = np.arange(len(labels))
    for k, (scale, d, c) in enumerate(((SEG, cal_diff, cal_canopy), (PIX, real_b, canopy_b))):
        d = np.asarray(d, np.float64)
        cls = np.digitize(np.nan_to_num(np.asarray(c), nan=0.0), np.asarray(edges)[1:-1])
        st = [residual_stats(d[cls == j]) if (cls == j).sum() >= min_n else None for j in range(len(labels))]
        a2.plot(pos + (k - 0.5) * 0.12, [s["robust_sd"] if s else np.nan for s in st], "o-", color=SCALE_COLORS[scale], ms=4, lw=1.4)
        a2.plot(pos + (k - 0.5) * 0.12, [s["RMSE"] if s else np.nan for s in st], "s", color=SCALE_COLORS[scale], ms=4, mfc="white")
    for s, ls in zip(sigmas, (":", "--", "-.")):
        a2.axhline(s, color="#0b0b0b", ls=ls, lw=1.0)
    a2.set_xticks(pos); a2.set_xticklabels(labels)
    a2.set_xlabel("Canopy height (m)", **BOLD); a2.set_ylabel(f"Spread of {EPS} (m)", **BOLD); a2.grid(True, axis="y")
    a2.set_title("(b) Spread by canopy class", **BOLD)
    handles += [Line2D([], [], color="#52514e", marker="o", ls="-", ms=4, label=f"Robust SD of {EPS}"),
                Line2D([], [], color="#52514e", marker="s", ls="none", ms=4, mfc="white", label=f"RMS {EPS}")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, prop={"weight": "bold"}, frameon=False)
    save_figure(fig, save); plt.show()


def plot_noise_conditions(runs: dict, canopy_scored, train_sigma: float | None = None, edges=NOISE_CLASS_EDGES,
                          min_px: int = 200, save: str | None = None) -> None:
    """runs from ablations.run_noise_conditions: (a) mean error per condition (± seed SD; > 0 = predicted
    ground too high); (b) RMSE per canopy class per condition. canopy_scored: S["canopy"][scored pixels]."""
    labs = [condition_label(k, train_sigma) for k in runs]
    cls = np.digitize(np.nan_to_num(np.asarray(canopy_scored), nan=0.0), np.asarray(edges)[1:-1])
    keep = [j for j in range(len(edges) - 1) if (cls == j).sum() >= min_px]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.4]))
    for i, (key, lab) in enumerate(zip(runs, labs)):
        r = [np.mean(x["err"].astype(np.float64)) for x in runs[key]]
        a1.bar(i, np.mean(r), width=0.7, color=_cond_color(lab), yerr=np.std(r) if len(r) > 1 else None, capsize=2,
               edgecolor="white", linewidth=0.4, label=_tex(lab))
        ys = [np.mean([np.sqrt(np.mean(x["err"][cls == j].astype(np.float64) ** 2)) for x in runs[key]]) for j in keep]
        a2.plot(np.arange(len(keep)), ys, "o-", color=_cond_color(lab), ms=4, lw=1.4)
    a1.set_xticks(range(len(labs)))
    a1.set_xticklabels([_tex(COND_SHORT.get(l, "Gaussian ε" if l.startswith("Lidar + Gaussian") else l)) for l in labs],
                       rotation=30, ha="right")
    a1.axhline(0, color="#52514e", lw=0.6)
    a1.set_ylabel("Mean error (m)\n(> 0: under-correction)", **BOLD); a1.set_title("(a) Mean error per condition", **BOLD)
    a1.grid(True, axis="y"); a1.set_axisbelow(True)
    a2.set_xticks(np.arange(len(keep)))
    a2.set_xticklabels([f"{edges[j]:g}–{edges[j + 1]:g}".replace("inf", "∞") for j in keep])
    a2.set_xlabel("Canopy height (m)", **BOLD); a2.set_ylabel("RMSE (m)", **BOLD)
    a2.set_title("(b) RMSE by canopy class", **BOLD); a2.grid(True, axis="y")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, prop={"weight": "bold"}, frameon=False)
    save_figure(fig, save); plt.show()


# ---------- ATL08 segments (cell 92) ----------
def plot_atl08(t, bbox_wgs84, save: str | None = None) -> None:
    """t from evaluate.summarize_atl08. Left: where the segments are, coloured by ground photons;
    right: median ground photons against canopy height, strong vs weak beams."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = a1.scatter(t.lon, t.lat, c=np.log10(t.n_te_ph.clip(lower=1)), s=1, cmap="viridis")
    fig.colorbar(sc, ax=a1, label="log10 ground photons per segment")
    lon0, lat0, lon1, lat1 = bbox_wgs84
    a1.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0], color="black", lw=1)
    a1.set_xlabel("longitude"); a1.set_ylabel("latitude"); a1.set_title("ATL08 segments with a ground height")
    a1.grid(alpha=0.3)
    for b, col in (("strong", "#2a78d6"), ("weak", "#eb6834")):
        s = t[(t.beam_type == b) & t.h_canopy.notna()]
        if len(s) < 50:
            continue
        bins = np.arange(0, 35, 2.5)
        k = np.digitize(s.h_canopy, bins) - 1
        med = [s.n_te_ph[k == i].median() if (k == i).sum() >= 20 else np.nan for i in range(len(bins) - 1)]
        a2.plot(bins[:-1] + 1.25, med, color=col, marker="o", ms=6, lw=2, label=f"{b} beams")
    a2.set_yscale("log"); a2.set_xlabel("ATL08 canopy height (m)"); a2.set_ylabel("median ground photons per segment")
    a2.set_title("Ground photons under canopy"); a2.grid(alpha=0.3); a2.legend()
    plt.tight_layout(); save_figure(fig, save); plt.show()