"""
Metrics on error / sigma arrays (numpy). No model, no rasters.

    error_metrics       RMSE, MAE, ME, NMAD, share |e| > t; with sigma: RMS sigma, Cov. 95%, NLL, CRPS
    nmad                robust spread 1.4826 * median |e - median(e)|
    crps_gaussian       closed-form CRPS of N(mu, sigma^2)
    coverage            observed coverage of central intervals at nominal levels
    error_tail_stats    share of |z| > z_tail and of the largest errors in the squared error
    residual_stats      n, median, mean, RMS, robust SD, share |d| > big (ATL08 - lidar residuals)
    qedges              bin edges with about equal counts

Replaces the per-cell variants _site_metrics (cell 106), _metrics (64, 129), _m (66, 122), _met (68),
metrics (103), _run_metrics (125-128), _crps_gauss (106), _crps (134), _stats (112, 125-128).
The torch versions used during training are in src/training/losses.py (gaussian_nll, point_metrics).

NLL is 0.5 * (log sigma^2 + e^2 / sigma^2), without the constant 0.5 log 2 pi, as in training.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

Z95 = float(norm.ppf(0.975))       # 1.959964...; the notebook used 1.96 in most places


def nmad(e) -> float:
    e = np.asarray(e, np.float64)
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def crps_gaussian(err, sigma) -> np.ndarray:
    """CRPS of a Gaussian N(mu, sigma^2) for the observation y, with err = mu - y (closed form), per pixel."""
    err, sigma = np.asarray(err, np.float64), np.asarray(sigma, np.float64)
    z = err / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def error_metrics(err, sigma=None, thresh: float | None = None, crps: bool = True) -> dict[str, float]:
    """err = prediction - truth at the scored pixels; sigma = predicted SD there (None for point methods).
    Keys: RMSE, MAE, ME, NMAD, [|e|>{thresh}m], and with sigma: RMS σ, Cov. 95%, NLL, [CRPS]."""
    e = np.asarray(err, np.float64)
    r = {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
         "ME": float(np.mean(e)), "NMAD": nmad(e)}
    if thresh is not None:
        r[f"|e|>{thresh:g}m"] = float(np.mean(np.abs(e) > thresh))
    if sigma is not None:
        s2 = np.asarray(sigma, np.float64) ** 2
        r["RMS σ"] = float(np.sqrt(np.mean(s2)))
        r["Cov. 95%"] = float(np.mean(np.abs(e) <= 1.96 * np.sqrt(s2)))
        r["NLL"] = float(np.mean(0.5 * (np.log(s2) + e ** 2 / s2)))
        if crps:
            r["CRPS"] = float(np.mean(crps_gaussian(e, np.sqrt(s2))))
    return r


def coverage(err, sigma, levels=(0.50, 0.68, 0.80, 0.90, 0.95, 0.99)) -> np.ndarray:
    """Observed share of |err| <= k sigma, with k the half-width of the central `levels` interval of N(0, 1)."""
    z = np.asarray(err, np.float64) / np.asarray(sigma, np.float64)
    return np.array([np.mean(np.abs(z) <= norm.ppf(0.5 + p / 2)) for p in levels])


def error_tail_stats(err, sigma, z_tail: float = 3.0, top: float = 0.01) -> dict[str, float]:
    """How much of the squared error sits in the tails (cell 48): share of pixels with |z| > z_tail and
    the share of the squared error they carry; the same for the largest `top` fraction of |error|."""
    e = np.asarray(err, np.float64)
    z = e / np.asarray(sigma, np.float64)
    se = e ** 2
    tail = np.abs(z) > z_tail
    q = np.quantile(np.abs(e), 1 - top)
    big = np.abs(e) >= q
    return {f"|z|>{z_tail:g}": float(tail.mean()), f"normal |z|>{z_tail:g}": float(2 * norm.sf(z_tail)),
            f"sq. error in |z|>{z_tail:g}": float(se[tail].sum() / se.sum()),
            f"top {top:.0%} |e| threshold (m)": float(q), f"sq. error in top {top:.0%}": float(se[big].sum() / se.sum())}


def residual_stats(d, big: float = 5.0) -> dict[str, float]:
    """Distribution of residuals d (e.g. ATL08 - lidar): n, median, mean, RMSE, robust_sd, share_gt{big}."""
    d = np.asarray(d, np.float64)
    return dict(n=int(d.size), median=float(np.median(d)), mean=float(np.mean(d)),
                RMSE=float(np.sqrt(np.mean(d ** 2))), robust_sd=nmad(d),
                **{f"share_gt{big:g}": float(np.mean(np.abs(d) > big))})


def qedges(x, n: int = 8) -> np.ndarray:
    """Edges giving about equal counts per bin (finite x only); the last edge nudged up to include the max."""
    e = np.unique(np.nanquantile(np.asarray(x, np.float64), np.linspace(0, 1, n + 1)))
    e[-1] += 1e-6
    return e
