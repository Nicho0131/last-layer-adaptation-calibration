"""
Evaluation tables: how good are the predictions, where, and how well calibrated is sigma.

Validation / test metrics over episodes during training: src.training.trainer.evaluate (not repeated here).

    method names        LABELS, method_role, method_label (names printed in tables and figures)
    helpers             distance_to_obs_km, signed_edge_distance, score_mask, set_map
    one site (P, S, OBS)    metrics_table, canopy_table, site_block_table, edge_profile_table,
                            anomaly_tables, anomaly_patches
    pooled items [(P, S, OBS)]  episode_items, table5, class_table, pooled_errors, calibration_table,
                            sigma_distance_table, spread_skill_table, sharpness_table
    Site A pixel analyses   collect_pixels, pixel_summary, coverage_table, breakdown_table,
                            block_table, semivariogram, variogram_reach, error_vs_train_distance
    ATL08 quality           summarize_atl08, compare_with_lidar, atl08_tail_stats, compare_filters,
                            sample_band_at
Tables are returned as pandas DataFrames and printed; the figures that draw them belong in a separate plotting module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from affine import Affine
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from scipy.stats import norm, spearmanr

from dataset.build_inputs import site_band
from src.evaluation.baselines import add_classical_baselines
from src.evaluation.inference import predict_episode
from src.evaluation.metrics import Z95, coverage, crps_gaussian, error_metrics, nmad, qedges
from src.preprocessing.icesat import add_segment_classes, along_track_deviation, apply_filter
from src.preprocessing.model_inputs import Site

CANOPY_EDGES = (0, 2, 5, 10, 20, np.inf)
DIST_EDGES_KM = (0, 0.25, 0.5, 1, 2, 4, 8, np.inf)
SLOPE_EDGES = (0, 5, 10, 20, np.inf)
LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


# =============================================================================
# method names (cell 106)
# =============================================================================
LABELS = {                                    # internal name -> name printed in tables and figures
    "uncorrected DSM": "Copernicus GLO-30 DSM",
    "FABDEM": "FABDEM",
    "FathomDEM": "FathomDEM",
    "network alone": "Network only",
    "+ global offset": "Network + global ATL08 offset",
    "full": "Network + ATL08 adaptation",
    "DSM + ATL08 offset": "Copernicus + global ATL08 offset",
    "ridge regression": "Regional ridge regression",
    "regression kriging": "Regression kriging",
}


def method_role(name: str) -> tuple[str, str | None]:
    """('full', 'M0') for 'M0 full'; ('FABDEM', None) for a baseline."""
    for r in ("network alone", "+ global offset", "full"):
        if name.endswith(" " + r):
            return r, name[: -len(r) - 1]
    return name, None


def method_label(name: str, multi_model: bool = False) -> str:
    r, m = method_role(name)
    base = LABELS.get(r, r)
    return f"{base} ({m})" if (m and multi_model) else base


def _show(df, fmt="{:.3f}", width=200, **kw):
    with pd.option_context("display.float_format", fmt.format, "display.width", width):
        print(df.to_string(**kw))


# =============================================================================
# helpers
# =============================================================================
def distance_to_obs_km(obs_mask, pixel_m: float) -> np.ndarray:
    """Distance (km) of every pixel to the nearest observed pixel; inf everywhere if there is none."""
    obs_mask = np.asarray(obs_mask, bool)
    if not obs_mask.any():
        return np.full(obs_mask.shape, np.inf)
    return distance_transform_edt(~obs_mask) * pixel_m / 1000


def signed_edge_distance(layer, thresh_m: float) -> np.ndarray:
    """Signed distance (px) to the edge of `layer` > thresh_m: > 0 inside the object, < 0 outside."""
    tall = np.where(np.isfinite(layer), layer, 0.0) > thresh_m
    return np.where(tall, distance_transform_edt(tall), -distance_transform_edt(~tall))


def score_mask(S: dict, OBS: dict, P: dict) -> np.ndarray:
    """Pixels every method is scored on: valid lidar, not an ATL08 pixel, finite in every method of P,
    and inside S["score"] if present (block episodes: the block, not its context ring)."""
    sel = S["valid"] & ~OBS["obs_mask"]
    if "score" in S:
        sel &= S["score"]
    for p in P.values():
        sel &= np.isfinite(p["mu"])
    return sel


def set_map(site: Site) -> np.ndarray:
    """Per pixel of Site A: 0 train, 1 val, 2 test, -1 gap (cell 57)."""
    H, W = site.hw
    s = np.full((H, W), -1, np.int8)
    for i, k in enumerate(("train", "val", "test")):
        y0, x0, y1, x1 = site.boxes[k]
        s[y0:y1, x0:x1] = i
    return s


# =============================================================================
# one site: tables (cells 106, 119, 132)
# =============================================================================
def metrics_table(P: dict, S: dict, OBS: dict, thresh: float = 2.0) -> pd.DataFrame:
    """Per method: RMSE, MAE, ME, NMAD, share |e| > thresh; and RMS σ, Cov. 95%, NLL, CRPS where a σ exists."""
    sel = score_mask(S, OBS, P)
    rows = {name: error_metrics((p["mu"] - S["target"])[sel], None if p["sigma"] is None else p["sigma"][sel],
                                thresh) for name, p in P.items()}
    df = pd.DataFrame(rows).T
    print(f"{sel.sum():,} scored pixels (valid lidar, off the ATL08 pixels, finite in every method)")
    _show(df, width=160, na_rep="—")
    return df


def canopy_table(P: dict, S: dict, OBS: dict, edges=CANOPY_EDGES, min_px: int = 200):
    """RMSE and bias per canopy-height class (m) per method. Returns (RMSE table, bias table)."""
    sel = score_mask(S, OBS, P)
    cls = np.digitize(S["canopy"][sel], np.asarray(edges)[1:-1])
    labels = [f"{edges[j]:g}-{edges[j + 1]:g} m" for j in range(len(edges) - 1)]
    rmse, bias = {}, {}
    for name, p in P.items():
        err = (p["mu"] - S["target"])[sel]
        ok = lambda j: (cls == j).sum() >= min_px
        rmse[name] = {labels[j]: np.sqrt(np.mean(err[cls == j] ** 2)) if ok(j) else np.nan for j in range(len(labels))}
        bias[name] = {labels[j]: err[cls == j].mean() if ok(j) else np.nan for j in range(len(labels))}
    n = pd.Series({labels[j]: int((cls == j).sum()) for j in range(len(labels))}, name="pixels")
    R, B = pd.DataFrame(rmse).T, pd.DataFrame(bias).T
    print("pixels per canopy class:\n" + n.to_string())
    print("\nRMSE (m) by canopy height:"); _show(R, width=160, na_rep="—")
    print("\nbias (m) by canopy height:"); _show(B, width=160, na_rep="—")
    return R, B


def site_block_table(P: dict, S: dict, OBS: dict, methods=("M0 full", "FABDEM"), block_km: float = 5.0,
                     tall_m: float = 5.0, bldg=None, bldg_m: float = 2.0, min_valid: float = 0.1) -> pd.DataFrame:
    """Per block of block_km: landscape (tall-canopy share, building share, elevation, std of true delta_h)
    and per method RMSE / bias (+ RMS σ and cover95 where the method has σ). Same scored pixels for all."""
    methods = [m for m in methods if m in P]
    sel = score_mask(S, OBS, P)
    B = int(round(block_km * 1000 / S["pixel_m"]))
    bldg = S.get("bldg") if bldg is None else bldg
    rows = []
    for by in range(0, S["H"], B):
        for bx in range(0, S["W"], B):
            sl = (slice(by, by + B), slice(bx, bx + B))
            ok = sel[sl]
            if ok.sum() < min_valid * B * B:
                continue
            c = S["canopy"][sl][S["dsm_ok"][sl]]
            r = dict(by=by, bx=bx, n=int(ok.sum()), canopy_frac=float(np.mean(c > tall_m)),
                     elev=float(S["dsm"][sl][S["dsm_ok"][sl]].mean()), dh_std=float(S["target"][sl][ok].std()))
            if bldg is not None:
                b = bldg[sl][np.isfinite(bldg[sl])]
                r["bldg_frac"] = float(np.mean(b > bldg_m)) if b.size else np.nan
            for name in methods:
                e = (P[name]["mu"][sl] - S["target"][sl])[ok]
                r[f"rmse | {name}"] = float(np.sqrt(np.mean(e ** 2)))
                r[f"bias | {name}"] = float(np.mean(e))
                if P[name]["sigma"] is not None:
                    s = P[name]["sigma"][sl][ok].astype(np.float64)
                    r[f"rms_sigma | {name}"] = float(np.sqrt(np.mean(s ** 2)))
                    r[f"cover95 | {name}"] = float(np.mean(np.abs(e) <= 1.96 * s))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs.update(B=B, block_km=block_km, tall_m=tall_m, methods=methods)
    print(f"{len(df)} blocks of {block_km:g} km ({B} px) with >= {min_valid:.0%} scored pixels")
    if len(methods) >= 2:
        a, b = methods[0], methods[1]
        d = df[f"rmse | {a}"] - df[f"rmse | {b}"]
        print(f"{method_label(a)} lower RMSE than {method_label(b)} in {(d < 0).sum()} of {len(df)} blocks "
              f"(median difference {d.median():+.2f} m)")
    cols = ["canopy_frac"] + (["bldg_frac"] if "bldg_frac" in df else []) + ["elev", "dh_std"] + \
           [c for c in df.columns if c.startswith(("rmse |", "rms_sigma |", "cover95 |"))]
    _show(df[cols].describe().loc[["mean", "50%", "min", "max"]].rename(index={"50%": "median"}), width=160)
    return df


def edge_profile_table(P: dict, S: dict, OBS: dict, layer, thresh_m: float, methods=("M0 full", "FABDEM"),
                       max_px: int = 30, min_px: int = 200) -> pd.DataFrame:
    """Error against the signed distance (px, > 0 inside) to the edge of layer > thresh_m (e.g. buildings
    > 2 m, canopy > 5 m), per method: RMSE, mean error and RMS σ (where the method has σ). One row per
    distance with >= min_px scored pixels (cells 61, 119)."""
    methods = [m for m in methods if m in P]
    signed = signed_edge_distance(layer, thresh_m)
    sel = score_mask(S, OBS, P) & (np.abs(signed) <= max_px)
    d = np.round(signed[sel]).astype(int)
    ks = [k for k in range(-max_px, max_px + 1) if (d == k).sum() >= min_px]
    rows = {k: {"pixels": int((d == k).sum())} for k in ks}
    for m in methods:
        e = (P[m]["mu"] - S["target"])[sel]
        sg = None if P[m]["sigma"] is None else P[m]["sigma"][sel].astype(np.float64)
        for k in ks:
            rows[k][f"RMSE | {m}"] = np.sqrt(np.mean(e[d == k] ** 2))
            rows[k][f"ME | {m}"] = np.mean(e[d == k])
            if sg is not None:
                rows[k][f"RMS σ | {m}"] = np.sqrt(np.mean(sg[d == k] ** 2))
    df = pd.DataFrame(rows).T
    df.index.name = "distance to edge (px)"
    print(f"edges > {thresh_m:g} m: {sel.sum():,} scored px within ±{max_px} px | "
          f"distances {min(ks) if ks else '-'} to {max(ks) if ks else '-'} px (>= {min_px} px each)")
    return df


def anomaly_tables(S: dict, OBS: dict, P: dict, feat_path: str, lo: float = 0.0, hi: float = 2.0,
                   tall_m: float = 5.0, ref=(2.0, 5.0)):
    """The canopy lo-hi m class against the reference class `ref`, then the anomaly class split by
    building / distance to tall canopy / slope / sign of delta_h (cell 132). Returns (T1, T2, extra)."""
    band = lambda b: site_band(feat_path, b)                            # 1-based bands of the GEE stack
    slope, lai, evi = band(2), band(6), band(7)
    assert slope.shape == (S["H"], S["W"]), "feature stack is not on the site grid"
    sel = score_mask(S, OBS, {m: P[m] for m in ("uncorrected DSM", "FABDEM", "M0 full") if m in P})
    bldg = S["bldg"]
    tall = S["canopy"] > tall_m
    d_tall = distance_transform_edt(~tall) * S["pixel_m"]               # m to the nearest canopy > tall_m
    dh, can = S["target"], S["canopy"]
    cls = sel & (can >= lo) & (can < hi)
    refm = sel & (can >= ref[0]) & (can < ref[1])
    e_dsm = -dh                                                         # DSM error = 0 - delta_h
    e_our = P["M0 full"]["mu"] - dh
    e_fab = P["FABDEM"]["mu"] - dh if "FABDEM" in P else None
    tot_sq_our = np.sum(e_our[sel] ** 2)

    def describe(m):
        r = dict(pixels=int(m.sum()), share_site=m.sum() / sel.sum(),
                 dh_mean=dh[m].mean(), dh_median=np.median(dh[m]), share_dh_below_m5=np.mean(dh[m] < -5),
                 share_dh_above_p2=np.mean(dh[m] > 2), bldg_share=np.mean(bldg[m] > 0),
                 slope_median=np.nanmedian(slope[m]), lai_median=np.nanmedian(lai[m]), evi_median=np.nanmedian(evi[m]),
                 dist_tall_median_m=np.median(d_tall[m]), share_within_60m_of_tall=np.mean(d_tall[m] <= 60),
                 RMSE_dsm=np.sqrt(np.mean(e_dsm[m] ** 2)), RMSE_ours=np.sqrt(np.mean(e_our[m] ** 2)),
                 share_of_our_sq_error=np.sum(e_our[m] ** 2) / tot_sq_our)
        if e_fab is not None:
            r["RMSE_fabdem"] = np.sqrt(np.mean(e_fab[m] ** 2))
        return r

    T1 = pd.DataFrame({f"canopy {lo:g}-{hi:g} m": describe(cls), f"canopy {ref[0]:g}-{ref[1]:g} m": describe(refm)})
    print("(1) the anomaly class against the reference class"); _show(T1)

    groups = {
        "building > 0 m": bldg > 0, "no building": bldg <= 0,
        "within 60 m of canopy > 5 m": d_tall <= 60, "more than 60 m from tall canopy": d_tall > 60,
        "slope >= 10°": slope >= 10, "slope < 10°": slope < 10,
        "delta_h < -5 m (DSM far above ground)": dh < -5, "-5 <= delta_h <= 2 m": (dh >= -5) & (dh <= 2),
        "delta_h > 2 m (DSM below ground)": dh > 2,
    }
    rows = {}
    for name, g in groups.items():
        m = cls & g
        if m.sum() < 50:
            continue
        rows[name] = dict(pixels=int(m.sum()), share_of_class=m.sum() / cls.sum(),
                          share_of_class_sq_err_ours=np.sum(e_our[m] ** 2) / np.sum(e_our[cls] ** 2),
                          dh_mean=dh[m].mean(), RMSE_dsm=np.sqrt(np.mean(e_dsm[m] ** 2)),
                          RMSE_ours=np.sqrt(np.mean(e_our[m] ** 2)),
                          **({"RMSE_fabdem": np.sqrt(np.mean(e_fab[m] ** 2))} if e_fab is not None else {}))
    T2 = pd.DataFrame(rows).T
    print("\n(2) inside the anomaly class: where does the error live?"); _show(T2)
    return T1, T2, dict(slope=slope, lai=lai, evi=evi, d_tall=d_tall, cls=cls)


def anomaly_patches(S: dict, OBS: dict, P: dict, extra: dict, dh_thresh: float = -5.0, min_px: int = 10,
                    top: int = 12) -> pd.DataFrame:
    """Connected patches of anomaly pixels with delta_h < dh_thresh (DSM far above the lidar ground),
    largest first: area, centre (km, to look up in aerial imagery), mean delta_h, building / LAI / EVI, errors."""
    m = extra["cls"] & (S["target"] < dh_thresh)
    lab, n = cc_label(m)
    if n == 0:
        print("no patches")
        return pd.DataFrame()
    sizes = np.bincount(lab.ravel())[1:]
    T = S["transform"]
    rows = []
    for k in np.argsort(sizes)[::-1][:top]:
        if sizes[k] < min_px:
            break
        pix = lab == k + 1
        rr, cc = np.nonzero(pix)
        rows.append(dict(area_ha=sizes[k] * S["pixel_m"] ** 2 / 1e4,
                         east_km=(T.c + (cc.mean() + 0.5) * T.a) / 1000, north_km=(T.f + (rr.mean() + 0.5) * T.e) / 1000,
                         dh_mean=S["target"][pix].mean(), bldg_mean=S["bldg"][pix].mean(),
                         lai=np.nanmean(extra["lai"][pix]), evi=np.nanmean(extra["evi"][pix]),
                         slope=np.nanmean(extra["slope"][pix]), dist_tall_m=extra["d_tall"][pix].mean(),
                         err_ours=np.mean(P["M0 full"]["mu"][pix] - S["target"][pix]),
                         err_fabdem=np.mean(P["FABDEM"]["mu"][pix] - S["target"][pix]) if "FABDEM" in P else np.nan))
    df = pd.DataFrame(rows)
    print(f"{n:,} patches with delta_h < {dh_thresh:g} m; the {len(df)} largest:")
    _show(df, fmt="{:.2f}")
    return df


# =============================================================================
# pooled items [(P, S, OBS)] (cells 106, 132-134)
# =============================================================================
def episode_items(model, episodes: list[dict], site: Site, name: str = "M0", classical: bool = True,
                  atl08_noise_var: float = 0.0, amp=None) -> list[tuple[dict, dict, dict]]:
    """Site A block episodes (simulated tracks) as (P, S, OBS), like a real site, so the same tables apply.
    Scored pixels: valid lidar, off-track, INSIDE the block (score_mask). P has "uncorrected DSM",
    "<name> network alone", "<name> full" (+ B3-B5 if classical). S also gets "slope" if the site has it."""
    canopy_full, bldg_full = site.extra["canopy"], site.extra["bldg"]
    slope_full = site.extra.get("slope")
    items = []
    for i, e in enumerate(episodes):
        y0, x0 = e["origin"]
        h, w = e["target"].shape[-2:]
        npy = lambda k: e[k][0, 0].detach().cpu().numpy()
        valid = npy("valid").astype(bool)
        crop = lambda a: a[y0:y0 + h, x0:x0 + w]
        S = dict(target=np.where(valid, npy("target"), 0.0).astype(np.float32), valid=valid,
                 score=npy("score_mask").astype(bool) if "score_mask" in e else np.ones((h, w), bool),
                 canopy=np.nan_to_num(crop(canopy_full)).clip(min=0).astype(np.float32),
                 bldg=np.nan_to_num(crop(bldg_full)).clip(min=0).astype(np.float32),
                 H=h, W=w, pixel_m=site.pixel_m, dsm=crop(site.dsm), transform=site.transform * Affine.translation(x0, y0))
        S["dsm_ok"] = np.isfinite(S["dsm"])
        if slope_full is not None:
            S["slope"] = crop(slope_full).astype(np.float32)
        OBS = dict(obs_dh=np.where(npy("obs_mask"), npy("obs_dh"), 0.0).astype(np.float32),
                   obs_mask=npy("obs_mask").astype(bool))
        P = {"uncorrected DSM": dict(mu=np.zeros((h, w), np.float32), sigma=None)}
        mu0, s0 = predict_episode(model, e, empty=True, amp=amp)
        mu1, s1 = predict_episode(model, e, empty=False, amp=amp)
        P[f"{name} network alone"] = dict(mu=mu0, sigma=s0)
        P[f"{name} full"] = dict(mu=mu1, sigma=s1)
        if classical and OBS["obs_mask"].sum() >= 30:
            add_classical_baselines(P, S, OBS, atl08_noise_var=atl08_noise_var)
        print(f"block {i}: {OBS['obs_mask'].sum():,} ICESat px | {score_mask(S, OBS, P).sum():,} scored px")
        items.append((P, S, OBS))
    return items


def table5(items, rows: dict, block_km: float = 2.0, n_boot: int = 1000, sub_per_block: int = 500, seed: int = 0,
           digits: int = 3, thresh: float = 2.0):
    """items: [(P, S, OBS), ...] pooled (one site, or the test blocks). rows: {"B0": "uncorrected DSM", ...}
    (methods missing from P are skipped). Point estimates use every scored pixel; the 95 % intervals come
    from resampling block_km x block_km blocks with replacement (n_boot times), identical resamples for all
    methods. NMAD / |e|95 intervals use up to sub_per_block random pixels per block (speed).
    block_km must exceed the residual autocorrelation range (~0.4 km) and give many blocks.
    Returns (table of strings for the paper, table of numbers)."""
    rng = np.random.default_rng(seed)
    rows = {k: v for k, v in rows.items() if all(v in it[0] for it in items)}
    E, SG, BID = {r: [] for r in rows}, {r: [] for r in rows}, []
    off = 0
    for P, S, OBS in items:
        sel = score_mask(S, OBS, {v: P[v] for v in rows.values()})
        bpx = max(1, int(round(block_km * 1000 / S["pixel_m"])))
        rr, cc = np.nonzero(sel)
        nbx = int(np.ceil(S["W"] / bpx))
        BID.append(off + (rr // bpx) * nbx + cc // bpx)
        off += int(np.ceil(S["H"] / bpx)) * nbx
        for r, m in rows.items():
            E[r].append((P[m]["mu"] - S["target"])[sel].astype(np.float64))
            SG[r].append(None if P[m]["sigma"] is None else P[m]["sigma"][sel].astype(np.float64))
    bid = np.concatenate(BID)
    ub, inv = np.unique(bid, return_inverse=True)
    nB = len(ub)
    draws = rng.integers(0, nB, (n_boot, nB))                           # same resamples for every method
    W = np.stack([np.bincount(d, minlength=nB) for d in draws])         # [n_boot, nB] block multiplicities
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=nB))]
    pick = []                                                           # per block: up to sub_per_block pixels
    for b in range(nB):
        idx = order[starts[b]:starts[b + 1]]
        pick.append(idx if len(idx) <= sub_per_block else rng.choice(idx, sub_per_block, replace=False))
    print(f"{len(bid):,} scored pixels in {nB} blocks of {block_km:g} km | {n_boot} block-bootstrap resamples")

    num, txt = {}, {}
    n_b = np.bincount(inv, minlength=nB).astype(float)
    for r in rows:
        e = np.concatenate(E[r])
        s = None if any(x is None for x in SG[r]) else np.concatenate(SG[r])
        sums = dict(se=np.bincount(inv, e ** 2, nB), s=np.bincount(inv, e, nB))
        big = (np.abs(e) > thresh).astype(float)
        point = dict(RMSE=np.sqrt(np.mean(e ** 2)), ME=np.mean(e), NMAD=nmad(e), e95=np.percentile(np.abs(e), 95),
                     big=big.mean())
        boot = dict(RMSE=np.sqrt(W @ sums["se"] / (W @ n_b)), ME=W @ sums["s"] / (W @ n_b),
                    big=W @ np.bincount(inv, big, nB) / (W @ n_b))
        if s is not None:
            cov = (np.abs(e) <= 1.96 * s).astype(float)
            crps = crps_gaussian(e, s)
            point.update(cov95=cov.mean(), CRPS=crps.mean())
            boot.update(cov95=W @ np.bincount(inv, cov, nB) / (W @ n_b), CRPS=W @ np.bincount(inv, crps, nB) / (W @ n_b))
        sub = [e[p] for p in pick]
        nm, q95 = np.empty(n_boot), np.empty(n_boot)
        for t, d in enumerate(draws):
            x = np.concatenate([sub[b] for b in d])
            nm[t] = nmad(x); q95[t] = np.percentile(np.abs(x), 95)
        boot.update(NMAD=nm, e95=q95)
        num[r] = {k: point[k] for k in point}
        num[r].update({f"{k} lo": np.percentile(boot[k], 2.5) for k in point})
        num[r].update({f"{k} hi": np.percentile(boot[k], 97.5) for k in point})
        f = lambda v: f"{v:.{digits}f}"
        txt[r] = {k: f"{f(point[k])} [{f(num[r][k + ' lo'])}, {f(num[r][k + ' hi'])}]" for k in point}
    cols = ["RMSE", "ME", "NMAD", "e95", "big", "cov95", "CRPS"]
    T_txt = pd.DataFrame(txt).T.reindex(columns=cols).fillna("–")
    T_txt.columns = ["RMSE (m)", "ME (m)", "NMAD (m)", "|e|95 (m)", f"|e| > {thresh:g} m", "Cov. 95%", "CRPS (m)"]
    T_txt.insert(0, "Method", [method_label(rows[r]) for r in T_txt.index])
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(T_txt.to_string())
    return T_txt, pd.DataFrame(num).T


def class_table(items, key: str, edges, methods, min_px: int = 200, unit: str = "°") -> pd.DataFrame:
    """Pooled over items: per class of S[key] (e.g. "slope") and method, RMSE and bias, plus each class's
    pixel share (cell 132, _class_stats)."""
    E = {m: [] for m in methods}
    V = []
    for P, S, OBS in items:
        sel = score_mask(S, OBS, {m: P[m] for m in methods}) & np.isfinite(S[key])
        V.append(S[key][sel])
        for m in methods:
            E[m].append((P[m]["mu"] - S["target"])[sel].astype(np.float64))
    v = np.concatenate(V)
    cls = np.digitize(v, np.asarray(edges)[1:-1])
    labels = [f"{edges[j]:g}–{edges[j + 1]:g}{unit}".replace(f"–inf{unit}", f"{unit}+") for j in range(len(edges) - 1)]
    rows = []
    for j in range(len(labels)):
        if (cls == j).sum() < min_px:
            continue
        r = {key: labels[j], "Pixels": int((cls == j).sum()), "Share": np.mean(cls == j)}
        for m in methods:
            e = np.concatenate(E[m])[cls == j]
            r[f"RMSE | {method_label(m)}"] = np.sqrt(np.mean(e ** 2))
            r[f"Bias | {method_label(m)}"] = e.mean()
        rows.append(r)
    df = pd.DataFrame(rows).set_index(key)
    _show(df.assign(Share=df["Share"].map("{:.1%}".format)))
    return df


def pooled_errors(items, method: str = "M0 full") -> tuple[np.ndarray, np.ndarray]:
    """Errors e = mu - truth and sigma, pooled over items [(P, S, OBS)] on the scored pixels."""
    E, SG = [], []
    for P, S, OBS in items:
        sel = score_mask(S, OBS, {method: P[method]}) & np.isfinite(P[method]["sigma"])
        E.append((P[method]["mu"] - S["target"])[sel].astype(np.float64))
        SG.append(P[method]["sigma"][sel].astype(np.float64))
    return np.concatenate(E), np.concatenate(SG)


def _pooled_with_distance(items, method: str):
    E, SG, D = [], [], []
    for P, S, OBS in items:
        sel = score_mask(S, OBS, {method: P[method]}) & np.isfinite(P[method]["sigma"])
        d = distance_to_obs_km(OBS["obs_mask"], S["pixel_m"])
        E.append((P[method]["mu"] - S["target"])[sel].astype(np.float64))
        SG.append(P[method]["sigma"][sel].astype(np.float64)); D.append(d[sel])
    return np.concatenate(E), np.concatenate(SG), np.concatenate(D)


def calibration_table(sites: dict, methods=("M0 full",), levels=LEVELS, z_tail: float = 3.0) -> pd.DataFrame:
    """sites: {"Site A (simulated ATL08)": items, ...}. One row per site (and method): pixels, RMSE, RMS σ,
    mean and SD of z, observed coverage at each nominal level, share of |z| > z_tail and the share of the
    squared error they carry, Gaussian NLL (with the 0.5 log 2π constant, as in the notebook)."""
    rows = {}
    for site, items in sites.items():
        for m in methods:
            e, s = pooled_errors(items, m)
            z = e / s
            tail = np.abs(z) > z_tail
            r = {"Pixels": len(e), "RMSE (m)": np.sqrt(np.mean(e ** 2)), "RMS σ (m)": np.sqrt(np.mean(s ** 2)),
                 "mean z": z.mean(), "SD z": z.std()}
            r.update({f"cov {p:.0%}": c for p, c in zip(levels, coverage(e, s, levels))})
            r["max |cov − nominal| (pp)"] = 100 * max(abs(r[f"cov {p:.0%}"] - p) for p in levels)
            r[f"|z| > {z_tail:g}"] = tail.mean()
            r[f"sq. error in |z| > {z_tail:g}"] = np.sum(e[tail] ** 2) / np.sum(e ** 2)
            r["NLL"] = np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * z ** 2)
            rows[site if len(methods) == 1 else f"{site} | {method_label(m)}"] = r
    df = pd.DataFrame(rows).T
    df["Pixels"] = df["Pixels"].astype(int)
    print(f"Gaussian reference: |z| > {z_tail:g} for {2 * norm.sf(z_tail):.2%} of pixels")
    _show(df, width=250)
    return df


def sigma_distance_table(items, methods=("M0 network alone", "M0 full"), edges=(0, 0.25, 0.5, 1, 2, np.inf),
                         min_px: int = 200) -> pd.DataFrame:
    """Per method and class of distance to the nearest ATL08 pixel: pixels, share, RMS σ, RMSE,
    95 % coverage, RMSE / RMS σ (cell 134)."""
    rows = []
    for m in methods:
        e, s, d = _pooled_with_distance(items, m)
        cls = np.digitize(d, np.asarray(edges)[1:-1])
        for j in range(len(edges) - 1):
            k = cls == j
            if k.sum() < min_px:
                continue
            rows.append({"Method": method_label(m), "key": m,
                         "Distance": f"{edges[j]:g}–{edges[j + 1]:g} km".replace("–inf km", "+ km"),
                         "Pixels": int(k.sum()), "Share": k.mean(), "RMS σ (m)": np.sqrt(np.mean(s[k] ** 2)),
                         "RMSE (m)": np.sqrt(np.mean(e[k] ** 2)), "Cov. 95%": np.mean(np.abs(e[k]) <= Z95 * s[k]),
                         "RMSE / RMS σ": np.sqrt(np.mean(e[k] ** 2)) / np.sqrt(np.mean(s[k] ** 2))})
    df = pd.DataFrame(rows)
    print("σ against distance to the nearest ATL08 pixel")
    _show(df.drop(columns="key").assign(Share=df["Share"].map("{:.1%}".format)), index=False)
    return df


def spread_skill_table(sites: dict, methods=("M0 full",), n_bins: int = 10) -> pd.DataFrame:
    """Pixels in n_bins equal-count bins by predicted σ, per site (and method): RMS σ and RMSE per bin, with
    Spearman ρ(σ, |e|) (cell 134)."""
    rows = []
    print(f"spread-skill, {n_bins} σ bins")
    for site, items in sites.items():
        for m in methods:
            e, s = pooled_errors(items, m)
            b = np.argsort(np.argsort(s, kind="stable")) * n_bins // len(s)   # equal-count bins by σ rank
            xs = np.array([np.sqrt(np.mean(s[b == j] ** 2)) for j in range(n_bins)])
            ys = np.array([np.sqrt(np.mean(e[b == j] ** 2)) for j in range(n_bins)])
            rho = spearmanr(s, np.abs(e)).correlation
            rows += [dict(Site=site, key=m, Method=method_label(m), Bin=j + 1, rho=rho,
                          **{"RMS σ (m)": xs[j], "RMSE (m)": ys[j], "RMSE / RMS σ": ys[j] / xs[j],
                             "Pixels": int((b == j).sum())}) for j in range(n_bins)]
            print(f"  {site}, {method_label(m)}: Spearman ρ(σ, |e|) = {rho:.3f} | RMSE / RMS σ from "
                  f"{(ys / xs).min():.2f} (bin {np.argmin(ys / xs) + 1}) to {(ys / xs).max():.2f} (bin {np.argmax(ys / xs) + 1})")
    return pd.DataFrame(rows)


def sharpness_table(sites: dict, methods=("M0 network alone", "M0 full")) -> pd.DataFrame:
    """Per site and method: 95 % coverage, RMS σ, mean width of the 95 % interval, RMSE and CRPS."""
    rows = []
    for site, items in sites.items():
        for m in methods:
            e, s = pooled_errors(items, m)
            rows.append({"Site": site, "Method": method_label(m), "key": m,
                         "Cov. 95%": np.mean(np.abs(e) <= Z95 * s), "RMS σ (m)": np.sqrt(np.mean(s ** 2)),
                         "95% width (m)": 2 * Z95 * s.mean(), "RMSE (m)": np.sqrt(np.mean(e ** 2)),
                         "CRPS (m)": crps_gaussian(e, s).mean()})
    df = pd.DataFrame(rows)
    print("calibration and sharpness")
    _show(df.drop(columns="key").set_index(["Site", "Method"]))
    return df


# =============================================================================
# Site A pixel analyses (cells 46-50, 57-60)
# =============================================================================
def collect_pixels(model, episodes: list[dict], site: Site, amp=None) -> dict[str, np.ndarray]:
    """Off-track pixels with valid lidar from every episode -> 1-D arrays: err, tgt, sigma, dist_km and the
    raw canopy / LAI / EVI / building height there (cell 46)."""
    feat, feat_ok = site.extra["features"], site.extra["features_valid"]
    bands = (("canopy", 4), ("lai", 5), ("evi", 6), ("bldg", 7))         # 0-based bands of the GEE stack
    out = {k: [] for k in ("err", "tgt", "sigma", "dist_km") + tuple(k for k, _ in bands)}
    for e in episodes:
        mu, sig = predict_episode(model, e, amp=amp)
        tgt, val, obs = (e[k][0, 0].cpu().numpy() for k in ("target", "valid", "obs_mask"))
        val, obs = val.astype(bool), obs.astype(bool)
        dist = distance_to_obs_km(obs, site.pixel_m)
        sel = val & ~obs
        oy, ox = e["origin"]
        h, w = tgt.shape
        out["err"].append((mu - tgt)[sel]); out["tgt"].append(tgt[sel])
        out["sigma"].append(sig[sel]); out["dist_km"].append(dist[sel])
        for key, b in bands:
            band, ok = feat[b, oy:oy + h, ox:ox + w], feat_ok[b, oy:oy + h, ox:ox + w]
            out[key].append(np.where(ok, band, np.nan)[sel])
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def pixel_summary(Px: dict) -> dict:
    """RMSE, MAE and their ratio (1.25 for normally distributed errors) of collect_pixels output."""
    err = Px["err"]
    rmse, mae = float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))
    print(f"{len(err):,} off-track test pixels | RMSE {rmse:.2f} m | MAE {mae:.2f} m | "
          f"RMSE/MAE {rmse / mae:.2f} (normally distributed errors: 1.25)")
    return dict(pixels=len(err), RMSE=rmse, MAE=mae, ratio=rmse / mae)


def coverage_table(err, sigma, levels=LEVELS) -> pd.DataFrame:
    """Nominal vs observed coverage of the intervals mu ± k σ (cell 47)."""
    obs = coverage(err, sigma, levels)
    df = pd.DataFrame({"nominal": levels, "k (σ)": [norm.ppf(0.5 + p / 2) for p in levels], "observed": obs,
                       "difference": obs - np.asarray(levels)}).set_index("nominal")
    _show(df)
    return df


def breakdown_table(Px: dict, key: str, edges=None, n_bins: int = 8) -> pd.DataFrame:
    """Error by classes of Px[key] (collect_pixels output; e.g. "canopy", "dist_km"): pixels, RMSE, MAE,
    DSM RMSE, RMS σ, cover95, share of pixels and of the squared error (cell 49).
    edges=None: equal-count classes (qedges)."""
    x = Px[key]
    ok = np.isfinite(x)
    err, tgt, sig, x = Px["err"][ok], Px["tgt"][ok], Px["sigma"][ok], x[ok]
    edges = qedges(x, n_bins) if edges is None else np.asarray(edges, float)
    b = np.digitize(x, edges) - 1
    se_total, n_total = np.sum(err ** 2), len(err)
    rows = {}
    for i in range(len(edges) - 1):
        m = b == i
        if not m.any():
            continue
        name = f"{edges[i]:.3g}–{edges[i + 1]:.3g}"
        rows[name] = dict(pixels=int(m.sum()), RMSE=np.sqrt(np.mean(err[m] ** 2)), MAE=np.mean(np.abs(err[m])),
                          DSM_RMSE=np.sqrt(np.mean(tgt[m] ** 2)), RMS_sigma=np.sqrt(np.mean(sig[m] ** 2)),
                          cover95=np.mean(np.abs(err[m]) <= 1.96 * sig[m]),
                          share_px=m.sum() / n_total, share_sq_err=np.sum(err[m] ** 2) / se_total)
    df = pd.DataFrame(rows).T
    df.index.name = key
    _show(df)
    return df


def block_table(site: Site, mu_full, sig_full, obs_full, block_km: float = 8.0, tall_m: float = 5.0,
                bldg_m: float = 2.0, min_valid: float = 0.1) -> pd.DataFrame:
    """Site A cut into blocks: per block its set (train / val / test, by the centre pixel), landscape
    (tall-canopy and building share, elevation, std of true delta_h) and RMSE, RMS σ, cover95 of the
    full-map prediction from inference.predict_site_a_full (cell 58)."""
    H, W = site.hw
    target, valid = site.target.cpu().numpy(), site.valid.cpu().numpy()
    err_full = np.where(valid & ~obs_full, mu_full - target, np.nan)     # off-track, valid only
    sets = set_map(site)
    feat, feat_ok = site.extra["features"], site.extra["features_valid"]
    canopy = np.where(feat_ok[4], feat[4], np.nan)
    bldg = np.where(feat_ok[7], feat[7], np.nan)
    B = int(round(block_km * 1000 / site.pixel_m))
    rows = []
    for by in range(0, H, B):
        for bx in range(0, W, B):
            sl = (slice(by, by + B), slice(bx, bx + B))
            ok = np.isfinite(err_full[sl])
            if ok.sum() < min_valid * B * B:
                continue
            c = canopy[sl][np.isfinite(canopy[sl])]
            b = bldg[sl][np.isfinite(bldg[sl])]
            e, s, t = err_full[sl][ok], sig_full[sl][ok], target[sl][ok]
            rows.append(dict(by=by, bx=bx, set=int(sets[min(by + B // 2, H - 1), min(bx + B // 2, W - 1)]),
                             canopy_frac=(c > tall_m).mean() if c.size else np.nan,
                             bldg_frac=(b > bldg_m).mean() if b.size else np.nan,
                             elev=site.dsm[sl][site.dsm_ok[sl]].mean(), dh_std=t.std(),
                             rmse=np.sqrt(np.mean(e ** 2)), rms_sigma=np.sqrt(np.mean(s ** 2)),
                             cover95=np.mean(np.abs(e) <= 1.96 * s), n=int(ok.sum())))
    df = pd.DataFrame(rows)
    df.attrs.update(B=B, block_km=block_km)
    print(f"{len(df)} blocks of {block_km:g} km ({B} px) with >= {min_valid:.0%} valid pixels")
    summary = df.groupby("set")[["canopy_frac", "bldg_frac", "elev", "dh_std", "rmse", "rms_sigma", "cover95"]].mean()
    summary.index = [{0: "train", 1: "val", 2: "test", -1: "gap"}[i] for i in summary.index]
    summary.insert(0, "blocks", df.groupby("set").size().values)
    _show(summary)
    print("NOTE: errors in the training strip are in-sample (the model was trained there)")
    return df


def _pair(a, sy: int, sx: int):
    """a[y, x] and a[y + sy, x + sx] over the overlapping area."""
    H, W = a.shape
    y0, y1 = max(0, -sy), H - max(0, sy)
    x0, x1 = max(0, -sx), W - max(0, sx)
    return a[y0:y1, x0:x1], a[y0 + sy:y1 + sy, x0 + sx:x1 + sx]


def semivariogram(a, pixel_m: float, max_km: float = 20.0, n_lags: int = 24, max_pairs: int = 2_000_000,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """gamma(h) = 0.5 * mean((a(x) - a(x+h))^2), averaged over 4 directions (E, S, SE, SW), NaN ignored.
    Returns lag distances (km) and semivariance (cell 59)."""
    rng = np.random.default_rng(seed)
    lags = np.unique(np.round(np.geomspace(1, max_km * 1000 / pixel_m, n_lags)).astype(int))
    dist_km, gamma = [], []
    for L in lags:
        d2, dd = [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = math.hypot(dy, dx)
            sy, sx = int(round(dy * L / n)), int(round(dx * L / n))
            p, q = _pair(a, sy, sx)
            d = (p - q)[np.isfinite(p) & np.isfinite(q)]
            if d.size > max_pairs:
                d = rng.choice(d, max_pairs, replace=False)
            d2.append(d ** 2); dd.append(np.full(d.size, math.hypot(sy, sx)))
        d2, dd = np.concatenate(d2), np.concatenate(dd)
        if d2.size:
            gamma.append(0.5 * d2.mean()); dist_km.append(dd.mean() * pixel_m / 1000)
    return np.array(dist_km), np.array(gamma)


def variogram_reach(h, g, sill: float, frac: float = 0.95) -> float:
    """First lag (km) where the semivariance reaches frac * sill (NaN if never)."""
    hit = g >= frac * sill
    return float(h[np.argmax(hit)]) if np.any(hit) else float("nan")


def error_vs_train_distance(model, val_set: list[dict], site: Site, bin_km: float = 1.0, min_px: int = 500,
                            amp=None) -> pd.DataFrame:
    """Leakage check (cell 60): RMSE and bias of the validation blocks against the distance (km) from the
    end of the training strip, in bins of bin_km."""
    H, W = site.hw
    along_x = W >= H
    errs, pos = [], []
    for ep in val_set:
        mu, _ = predict_episode(model, ep, amp=amp)
        sel = (ep["valid"][0, 0] & ~ep["obs_mask"][0, 0] & ep["score_mask"][0, 0]).cpu().numpy()
        yy, xx = np.nonzero(sel)
        errs.append(mu[sel] - ep["target"][0, 0].cpu().numpy()[sel])
        pos.append(xx + ep["origin"][1] if along_x else yy + ep["origin"][0])
    err, pos = np.concatenate(errs), np.concatenate(pos)
    train_end = site.boxes["train"][3] if along_x else site.boxes["train"][2]
    d_km = (pos - (train_end - 1)) * site.pixel_m / 1000
    edges = np.arange(np.floor(d_km.min()), np.ceil(d_km.max()) + 1, bin_km)
    idx = np.digitize(d_km, edges) - 1
    rows = {}
    for k in range(len(edges) - 1):
        e = err[idx == k]
        if len(e) >= min_px:
            rows[f"{edges[k]:.1f}–{edges[k + 1]:.1f}"] = dict(pixels=len(e), RMSE=np.sqrt(np.mean(e ** 2)), bias=e.mean())
    df = pd.DataFrame(rows).T
    df.index.name = "distance from training strip (km)"
    _show(df)
    print(f"overall RMSE {np.sqrt(np.mean(err ** 2)):.3f} m")
    return df


# =============================================================================
# ATL08 quality against the lidar (cells 92, 94, 95, 102)
# =============================================================================
FILTER_CANDIDATES = {
    "none": dict(),
    "uncertainty < 2 m": dict(max_unc=2.0),
    "ground photons >= 20": dict(min_ph=20),
    "strong beams": dict(strong_only=True),
    "night": dict(night_only=True),
    "along-track dev < 3 m": dict(max_dev=3.0),
    "unc < 2 + dev < 3": dict(max_unc=2.0, max_dev=3.0),
    "unc < 2 + ph >= 20 + dev < 3": dict(max_unc=2.0, min_ph=20, max_dev=3.0),
}


def summarize_atl08(df: pd.DataFrame) -> pd.DataFrame:
    """Prints segment counts, ground photons by beam / light / canopy class and per year; returns the
    segments with a ground height, with 'light' and 'canopy' class columns added (cell 92)."""
    ok = df.h_te.notna()
    print(f"{len(df):,} segments inside the tile from {df.file.nunique()} files | "
          f"{df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}")
    print(f"with a ground height: {ok.sum():,} ({ok.mean():.1%})")
    t = add_segment_classes(df[ok].copy())
    q = lambda s: f"{s.median():6.0f} {s.quantile(0.1):6.0f} {s.quantile(0.9):6.0f}"
    print(f"\nground photons per segment (median / 10th / 90th pct): {q(t.n_te_ph)}")
    print(f"segments with < 10 ground photons: {(t.n_te_ph < 10).mean():.1%}\n")
    print(f"{'beam':7s} {'light':6s} {'segments':>9s} {'ground ph med':>14s} {'unc med (m)':>12s}")
    for (b, light), s in t.groupby(["beam_type", "light"]):
        print(f"{b:7s} {light:6s} {len(s):9,d} {s.n_te_ph.median():14.0f} {s.h_te_unc.median():12.2f}")
    print(f"\n{'canopy':8s} {'segments':>9s} {'ground ph med':>14s} {'unc med (m)':>12s}")
    for c, s in t.groupby("canopy", observed=True):
        print(f"{c:8s} {len(s):9,d} {s.n_te_ph.median():14.0f} {s.h_te_unc.median():12.2f}")
    print(f"{'no value':8s} {t.h_canopy.isna().sum():9,d}   (no canopy height: open ground or too few canopy photons)")
    print("\nsegments with a ground height per year:", t.groupby(t.date.dt.year).size().to_dict())
    return t


def compare_with_lidar(t: pd.DataFrame, dtm_egm_path: str, geoid_epsg: int = 3855) -> pd.DataFrame:
    """ATL08 ground (converted to EGM2008) minus the 30 m lidar DTM (EGM2008) at each segment centre ->
    column 'diff' (cell 94). t: segments with 'canopy' / 'light' columns (summarize_atl08 output)."""
    import pyproj
    import rasterio

    to_geoid = pyproj.Transformer.from_crs("EPSG:4979", f"EPSG:4326+{geoid_epsg}", always_xy=True,
                                           allow_ballpark=False)
    _, _, h_geo = to_geoid.transform(t.lon.values, t.lat.values, t.h_te.values, errcheck=True)
    with rasterio.open(dtm_egm_path) as src:
        x, y = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(t.lon.values, t.lat.values)
        dtm = np.array([v[0] for v in src.sample(zip(x, y))], dtype=np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            dtm[dtm == src.nodata] = np.nan
    t = t.assign(diff=h_geo - dtm)
    t = t[np.isfinite(t["diff"])]
    rob = lambda d: f"{d.median():+6.2f} {np.sqrt(np.mean(d ** 2)):6.2f} {nmad(d):6.2f}"
    print(f"{len(t):,} segments over valid lidar | ATL08 ground - lidar DTM (m): median, RMSE, robust std")
    print(f"  all            {rob(t['diff'])}")
    for b, s in t.groupby("beam_type"):
        print(f"  {b:6s} beams   {rob(s['diff'])}")
    for c, s in t.groupby("canopy", observed=True):
        print(f"  canopy {str(c):7s} {rob(s['diff'])}   ({len(s):,})")
    return t


def atl08_tail_stats(cmp: pd.DataFrame, big_m: float = 5.0) -> None:
    """Share of large ATL08 - lidar differences and their sign, by canopy, beam, light, photons,
    uncertainty (cell 95, tail_stats)."""
    d = cmp["diff"]
    pct = lambda s: " ".join(f"{np.percentile(s, p):+6.1f}" for p in (1, 5, 50, 95, 99))

    def block(title, groups):
        print(f"\n{title:14s} {'n':>7s} {'|d|>%g' % big_m:>8s} {'d>+%g' % big_m:>8s} {'d<-%g' % big_m:>8s}   "
              f"percentiles 1 / 5 / 50 / 95 / 99 (m)")
        for k, s in groups:
            ds = s["diff"]
            print(f"{str(k):14s} {len(s):7,d} {np.mean(ds.abs() > big_m):8.1%} {np.mean(ds > big_m):8.1%} "
                  f"{np.mean(ds < -big_m):8.1%}   {pct(ds)}")

    print(f"all: {len(cmp):,} segments | |d| > {big_m:g} m: {np.mean(d.abs() > big_m):.1%} "
          f"(above lidar ground {np.mean(d > big_m):.1%}, below {np.mean(d < -big_m):.1%})")
    block("canopy", cmp.groupby("canopy", observed=True))
    block("beam", cmp.groupby("beam_type"))
    block("light", cmp.groupby("light"))
    block("ground photons", cmp.groupby(pd.cut(cmp.n_te_ph, [0, 5, 10, 20, 50, 100, np.inf]), observed=True))
    block("uncertainty m", cmp.groupby(pd.cut(cmp.h_te_unc, [0, 0.5, 1, 2, 3, 5, np.inf]), observed=True))


def compare_filters(cmp: pd.DataFrame, candidates=FILTER_CANDIDATES, big_m: float = 5.0,
                    window: int = 5) -> pd.DataFrame:
    """For each candidate filter: share of segments kept, RMSE / robust spread / median / share of large
    differences, and RMSE under canopy > 10 m (cell 95). Returns cmp with the 'dev' column added."""
    if "dev" not in cmp:
        cmp = cmp.assign(dev=along_track_deviation(cmp, window))
    print(f"{'filter':30s} {'kept':>7s} {'RMSE':>6s} {'robust':>7s} {'median':>7s} {'|d|>%g' % big_m:>7s} "
          f"{'RMSE canopy>10':>15s}")
    for name, kw in candidates.items():
        s = apply_filter(cmp, **kw)
        d = s["diff"]
        tall = s[s.h_canopy > 10]["diff"]
        print(f"{name:30s} {len(s) / len(cmp):7.1%} {np.sqrt(np.mean(d ** 2)):6.2f} "
              f"{nmad(d):7.2f} {d.median():+7.2f} {np.mean(d.abs() > big_m):7.1%} "
              f"{np.sqrt(np.mean(tall ** 2)) if len(tall) else np.nan:15.2f}")
    return cmp


def sample_band_at(df: pd.DataFrame, stack_path: str, band: int) -> np.ndarray:
    """Value of `band` (1-based) of a raster at each segment's lon / lat (NaN for nodata) (cell 102)."""
    import pyproj
    import rasterio

    with rasterio.open(stack_path) as src:
        tr = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tr.transform(df.lon.values, df.lat.values)
        v = np.array([s[band - 1] for s in src.sample(zip(x, y))], dtype=np.float64)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        v[v == nod] = np.nan
    return v
