"""
Ablations and sensitivity studies: switch parts of the trained model off or change them, and compare.

Switches (context managers; the trained model is restored on exit):
    only_levels(model, keep)          the read only sees the grid levels in `keep`
    fixed_blend(model, kind, ell)     bilinear / Gaussian weights instead of the learned attention
    fixed_lambda(model, value)        lambda_l = value on every level
    scaled_lambda(model, scale, lv)   lambda_l x scale on chosen levels
    constant_support(model, aux)      the variance head gets the same support inputs everywhere
    record_support(model, store)      collect the support inputs (for constant_support's mean)

Studies:
    ablation_table_site / ablation_table_blocks    Table 6: every switch on a real site / Site A blocks
    ablation_context / _global / _levels / _blend  the four Site B ablations with (T, P) for the figures
    lambda_table                                   Table 7: learned lambda and implied prior width
    lambda_sweep_site / lambda_sweep_blocks        RMSE against lambda x scale
    ATL08 noise study                              residual_pools, run_noise_conditions, noise_verdict,
                                                   noise_stats_table, noise_condition_table,
                                                   pretrain_noise_table

One year of ATL08: src.preprocessing.icesat.site_icesat(..., years=(2023,)).
"""

from __future__ import annotations

import math
import types
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd
import torch

from src.evaluation.baselines import atl08_mean_only, network_offset
from src.evaluation.evaluate import distance_to_obs_km, score_mask
from src.evaluation.inference import predict_episode, predict_site, run_calibrator
from src.evaluation.metrics import error_metrics, residual_stats

NOISE_CLASS_EDGES = (0, 2, 5, 10, 20, np.inf)           # canopy classes in the noise tables
EMP_EDGES = (0, 2, 5, 10, 15, 20, 30, np.inf)           # canopy classes of the empirical noise pools


# =============================================================================
# switches
# =============================================================================
@contextmanager
def only_levels(model, keep):
    """Inside the block, the read only sees the grid levels whose index is in `keep`
    (the others are emptied after the write)."""
    orig = type(model)._write_pass

    def patched(self, *a, **k):
        levels = orig(self, *a, **k)
        for li, L in enumerate(levels):
            if li not in keep:
                L["count"] = torch.zeros_like(L["count"])
                L["w"] = torch.zeros_like(L["w"])
        return levels

    model._write_pass = types.MethodType(patched, model)
    try:
        yield
    finally:
        del model._write_pass


def _level_geometry(pix_xy, L):
    gx = pix_xy[:, 0] / L["rx"] - 0.5
    gy = pix_xy[:, 1] / L["ry"] - 0.5
    x0, y0 = gx.floor().long(), gy.floor().long()
    xs = torch.stack([x0, x0 + 1, x0, x0 + 1], -1)
    ys = torch.stack([y0, y0, y0 + 1, y0 + 1], -1)
    inb = (xs >= 0) & (xs < L["Wg"]) & (ys >= 0) & (ys < L["Hg"])
    idx = ys.clamp(0, L["Hg"] - 1) * L["Wg"] + xs.clamp(0, L["Wg"] - 1)
    cnt = L["count"][idx]
    return idx, inb & (cnt > 0), (gx[:, None] - xs.float()).float(), (gy[:, None] - ys.float()).float(), cnt


def _fixed_forward(kind: str, ell: float = 0.5):
    """Replacement read: bilinear or Gaussian weights instead of attention, equal share per level with data."""
    def forward(q_feat, pix_xy, levels):
        alphas, has, ws, logcs, dists, masks = [], [], [], [], [], []
        for L in levels:
            idx, valid, dx, dy, cnt = _level_geometry(pix_xy, L)
            k = ((1 - dx.abs()).clamp(min=0) * (1 - dy.abs()).clamp(min=0) if kind == "bilinear"
                 else torch.exp(-(dx ** 2 + dy ** 2) / (2 * ell ** 2))) * valid
            ksum = k.sum(-1, keepdim=True)
            alphas.append(k / ksum.clamp_min(1e-12)); has.append(ksum.squeeze(-1) > 0); masks.append(valid)
            ws.append(L["w"][idx]); logcs.append(torch.log1p(cnt))
            dists.append((dx ** 2 + dy ** 2).sqrt() * L["res_km"])
        A = torch.stack(alphas, 1)
        Hh = torch.stack(has, 1).float()
        A = A * (Hh / Hh.sum(-1, keepdim=True).clamp_min(1)).unsqueeze(-1)      # equal share per level with data
        alpha = A.reshape(A.shape[0], -1)
        M = torch.cat(masks, -1)
        dW = (alpha.unsqueeze(-1) * torch.cat(ws, 1)).sum(1)
        support = (alpha * torch.cat(logcs, -1)).sum(-1)
        dmin = torch.cat(dists, -1).masked_fill(~M, 1e3).amin(-1).clamp(max=100.0)
        return dW, torch.stack([support, torch.log1p(dmin), M.any(-1).float()], -1)
    return forward


@contextmanager
def fixed_blend(model, kind: str, ell: float = 0.5):
    """kind "bilinear" or "gaussian" (width ell in node spacings) instead of the learned attention."""
    gat = model.grid_gat
    gat.forward = _fixed_forward(kind, ell)
    try:
        yield
    finally:
        del gat.forward


@contextmanager
def fixed_lambda(model, value: float = 1.0):
    """lambda_l = value on every level (log_lam written back afterwards)."""
    orig = model.log_lam.data.clone()
    try:
        with torch.no_grad():
            model.log_lam.data.fill_(math.log(value))
        yield
    finally:
        model.log_lam.data.copy_(orig)


@contextmanager
def scaled_lambda(model, scale: float, levels=None):
    """lambda_l x scale on the chosen levels (all if None); trained values restored afterwards."""
    orig = model.log_lam.data.clone()
    try:
        with torch.no_grad():
            for li in (range(len(orig)) if levels is None else levels):
                model.log_lam.data[li] = orig[li] + math.log(scale)
        yield
    finally:
        model.log_lam.data.copy_(orig)


@contextmanager
def constant_support(model, aux_mean: torch.Tensor):
    """The variance head gets the same support inputs (aux_mean, 3 values) at every pixel."""
    gat = model.grid_gat
    orig = type(gat).forward

    def forward(q_feat, pix_xy, levels):
        dW, aux = orig(gat, q_feat, pix_xy, levels)
        return dW, aux_mean.to(aux.device, aux.dtype).expand_as(aux)
    gat.forward = forward
    try:
        yield
    finally:
        del gat.forward


@contextmanager
def record_support(model, store: list):
    """Collect the support inputs of every read call (sums + count, for their mean: support_mean)."""
    h = model.grid_gat.register_forward_hook(
        lambda m, i, o: store.append(o[1].detach().float().sum(0).cpu().tolist() + [o[1].shape[0]]))
    try:
        yield
    finally:
        h.remove()


def support_mean(store: list) -> torch.Tensor:
    a = np.array(store)
    return torch.tensor(a[:, :3].sum(0) / a[:, 3].sum(), dtype=torch.float32)


# =============================================================================
# Table 6: every switch, on a real site or on Site A's test blocks (cell 122)
# =============================================================================
VARIANTS = (("Full model", None), ("1 km level only", ("levels", {0})), ("4 km level only", ("levels", {1})),
            ("16 km level only", ("levels", {2})), ("Bilinear read", ("blend", "bilinear")),
            ("Gaussian read", ("blend", "gaussian")), ("Fixed λ", ("lambda", 1.0)),
            ("No support features", ("support", None)))


def _variant_ctx(model, spec, aux_mean, ell):
    if spec is None:
        return nullcontext()
    kind, arg = spec
    return {"levels": lambda: only_levels(model, arg), "blend": lambda: fixed_blend(model, arg, ell),
            "lambda": lambda: fixed_lambda(model, arg), "support": lambda: constant_support(model, aux_mean)}[kind]()


def _m3(err, sigma) -> dict:
    m = error_metrics(err, sigma, crps=False)
    return dict(RMSE=m["RMSE"], cov95=m["Cov. 95%"], NLL=m["NLL"])


def _gain_table(res: dict, b6: dict):
    """Rows = variants; "Share of gain" = (RMSE network only - RMSE row) / (RMSE network only - RMSE full)."""
    full = res["Full model"]["RMSE"]
    df = pd.DataFrame(res).T
    df["Share of gain"] = (b6["RMSE"] - df["RMSE"]) / (b6["RMSE"] - full)
    print(f"B6 (network without context): RMSE {b6['RMSE']:.3f} m, Cov. 95 % {b6['cov95']:.3f}, NLL {b6['NLL']:.3f}")
    out = df.copy()
    for c in ("RMSE", "cov95", "NLL"):
        out[c] = out[c].map("{:.3f}".format)
    out["Share of gain"] = out["Share of gain"].map("{:.0%}".format)
    out.columns = ["RMSE (m)", "Cov. 95%", "NLL", "Share of gain"]
    print(out.to_string())
    return out, df


def ablation_table_site(model, S: dict, OBS: dict, OBS_1YR: dict | None = None, ell: float = 0.5, amp=None):
    """Every variant on a real site. Scored: valid lidar, off ALL ATL08 pixels (the full set), identical for
    every row, including "One year of ATL08" (OBS_1YR from site_icesat(..., years=...)).
    Returns (formatted table, numbers)."""
    sel = S["valid"] & ~OBS["obs_mask"]
    if "score" in S:
        sel &= S["score"]
    err = lambda mu: (mu - S["target"])[sel]
    run = lambda o: run_calibrator(model, S["Xn"], o["obs_dh"], o["obs_mask"], amp=amp)
    H, W = S["H"], S["W"]
    mu, sg = run_calibrator(model, S["Xn"], np.zeros((H, W), np.float32), np.zeros((H, W), bool), amp=amp)
    b6 = _m3(err(mu), sg[sel])
    store = []
    with record_support(model, store):                       # the full run also gives the mean support inputs
        mu, sg = run(OBS)
    aux = support_mean(store)
    res = {"Full model": _m3(err(mu), sg[sel])}
    for name, spec in VARIANTS[1:]:
        with _variant_ctx(model, spec, aux, ell):
            mu, sg = run(OBS)
        res[name] = _m3(err(mu), sg[sel])
        print(f"  {name:22s} RMSE {res[name]['RMSE']:.3f}")
    if OBS_1YR is not None:
        mu, sg = run(OBS_1YR)
        res["One year of ATL08"] = _m3(err(mu), sg[sel])
    return _gain_table(res, b6)


def ablation_table_blocks(model, episodes: list[dict], ell: float = 0.5, amp=None):
    """Every variant pooled over Site A block episodes (simulated tracks); scored inside each block only,
    off-track, valid lidar. Returns (formatted table, numbers)."""
    prep = []
    for e in episodes:
        npy = lambda k: e[k][0, 0].detach().cpu().numpy()
        sel = npy("valid").astype(bool) & ~npy("obs_mask").astype(bool)
        if "score_mask" in e:
            sel &= npy("score_mask").astype(bool)
        prep.append((e, npy("target"), sel))

    def pooled(ctx_fn=None, empty=False):
        E, SG = [], []
        for e, t, sel in prep:
            with (ctx_fn() if ctx_fn is not None else nullcontext()):
                mu, sg = predict_episode(model, e, empty, amp)
            E.append((mu - t)[sel]); SG.append(sg[sel])
        return _m3(np.concatenate(E), np.concatenate(SG))

    b6 = pooled(empty=True)
    store = []
    with record_support(model, store):
        res = {"Full model": pooled()}
    aux = support_mean(store)
    for name, spec in VARIANTS[1:]:
        res[name] = pooled(lambda spec=spec: _variant_ctx(model, spec, aux, ell))
        print(f"  {name:22s} RMSE {res[name]['RMSE']:.3f}")
    return _gain_table(res, b6)


def lambda_table(model, sigma_eps: float = 1.0) -> pd.DataFrame:
    """Table 7: lambda_l = exp(log_lam_l) and the implied prior width tau_l = sigma_eps / sqrt(lambda_l):
    the ridge fit is the MAP estimate under noise N(0, sigma_eps^2) and prior dW ~ N(0, tau^2 I)."""
    lam = model.log_lam.exp().detach().cpu().numpy()
    df = pd.DataFrame({"Level": [f"{r / 1000:g} km" for r in model.level_res_m], "λ": lam,
                       "τ (m)": sigma_eps / np.sqrt(lam)}).set_index("Level")
    print(f"nominal σ_ε = {sigma_eps:g} m")
    print(df.to_string(float_format="{:.3f}".format))
    return df


# =============================================================================
# the four Site B ablations (cell 129): each returns (table, P) for the figures
# =============================================================================
DSM, NET, FULL = "Uncorrected DSM", "Network only", "Network + ATL08 adaptation"


def base_predictions(model, S: dict, OBS: dict, cache: dict | None = None, amp=None) -> dict:
    """Uncorrected DSM, network only and the full model, predicted once and shared by all four ablations."""
    cache = {} if cache is None else cache
    if FULL not in cache:
        cache[DSM] = dict(mu=np.zeros((S["H"], S["W"]), np.float32), sigma=None)
        cache[NET] = predict_site(model, S, OBS, empty=True, amp=amp)
        cache[FULL] = predict_site(model, S, OBS, amp=amp)
    return cache


def variant_table(P: dict, S: dict, OBS: dict, names, extra: dict | None = None) -> pd.DataFrame:
    """RMSE, MAE, ME (+ RMS σ, Cov. 95%, NLL where σ exists) per name, on the same scored pixels.
    extra: {column: fn(df, name)} for derived columns."""
    sel = score_mask(S, OBS, {n: P[n] for n in names})
    keep = ["RMSE", "MAE", "ME", "RMS σ", "Cov. 95%", "NLL"]
    rows = {}
    for n in names:
        m = error_metrics((P[n]["mu"] - S["target"])[sel], None if P[n]["sigma"] is None else P[n]["sigma"][sel],
                          crps=False)
        rows[n] = {k: m[k] for k in keep if k in m}
    df = pd.DataFrame(rows).T
    for col, fn in (extra or {}).items():
        df[col] = [fn(df, n) for n in df.index]
    print(f"{sel.sum():,} scored pixels (valid lidar, off the ATL08 pixels)")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df.to_string(na_rep="–"))
    return df


def ablation_context(model, S: dict, OBS: dict, cache: dict | None = None, amp=None):
    """With vs without ATL08."""
    P = base_predictions(model, S, OBS, cache, amp)
    names = [DSM, NET, FULL]
    T = variant_table(P, S, OBS, names)
    print(f"ATL08 lowers RMSE by {T.loc[NET, 'RMSE'] - T.loc[FULL, 'RMSE']:+.3f} m vs the network alone")
    return T, {n: P[n] for n in names}


def ablation_global(model, S: dict, OBS: dict, cache: dict | None = None, amp=None):
    """ATL08 mean only (no network), network only, network + ONE scalar offset, and the full model."""
    P = dict(base_predictions(model, S, OBS, cache, amp))
    P["ATL08 mean only"], m_b, s_b = atl08_mean_only(S, OBS)
    P["Network + ATL08 offset"], b = network_offset(P[NET], OBS)
    print(f"ATL08 mean {m_b:+.3f} m (SD {s_b:.3f}) | offset of the network {b:+.3f} m")
    names = [DSM, "ATL08 mean only", NET, "Network + ATL08 offset", FULL]
    return variant_table(P, S, OBS, names), {n: P[n] for n in names}


def ablation_levels(model, S: dict, OBS: dict, cache: dict | None = None, amp=None):
    """One grid level at a time, with each level's share of the gain over the network alone."""
    P = dict(base_predictions(model, S, OBS, cache, amp))
    n_lv = len(model.level_res_m)
    lv_names = [f"{model.level_res_m[i] / 1000:g} km level only" for i in range(n_lv)]
    for i, n in enumerate(lv_names):
        P[n] = predict_site(model, S, OBS, ctx=lambda i=i: only_levels(model, {i}), amp=amp)
    all_lv = predict_site(model, S, OBS, ctx=lambda: only_levels(model, set(range(n_lv))), amp=amp)
    print(f"check: all-levels wrapper vs normal model, max |difference| "
          f"{np.nanmax(np.abs(all_lv['mu'] - P[FULL]['mu'])):.2e} m (should be ~0)")
    P["Network + ATL08 offset"], _ = network_offset(P[NET], OBS)
    names = [NET, "Network + ATL08 offset"] + lv_names + [FULL]
    share = lambda df, n: (df.loc[NET, "RMSE"] - df.loc[n, "RMSE"]) / (df.loc[NET, "RMSE"] - df.loc[FULL, "RMSE"])
    T = variant_table(P, S, OBS, names, extra={"Share of gain": share})
    print("share of gain = (RMSE network only − RMSE row) / (RMSE network only − RMSE full model)")
    return T, {n: P[n] for n in names}


def ablation_blend(model, S: dict, OBS: dict, cache: dict | None = None, ells=(0.25, 0.5, 1.0), amp=None):
    """Learned attention vs Gaussian (width ell node spacings) and bilinear weights."""
    P = dict(base_predictions(model, S, OBS, cache, amp))
    ref_before = P[FULL]["mu"]
    g_names = [f"Gaussian read ($\\ell$ = {ell:g} node)" for ell in ells]   # mathtext label for the figures
    for ell, n in zip(ells, g_names):
        P[n] = predict_site(model, S, OBS, ctx=lambda ell=ell: fixed_blend(model, "gaussian", ell), amp=amp)
    P["Bilinear read"] = predict_site(model, S, OBS, ctx=lambda: fixed_blend(model, "bilinear"), amp=amp)
    after = predict_site(model, S, OBS, amp=amp)
    print(f"check: trained attention before vs after the swaps, max |difference| "
          f"{np.nanmax(np.abs(after['mu'] - ref_before)):.2e} m (should be ~0)")
    names = [FULL] + g_names + ["Bilinear read"]
    vs = lambda df, n: df.loc[n, "RMSE"] - df.loc[FULL, "RMSE"]
    return variant_table(P, S, OBS, names, extra={"RMSE vs learned": vs}), {n: P[n] for n in names}


# =============================================================================
# lambda sweeps (cells 74, 131)
# =============================================================================
SCALES = (0.25, 1, 4, 16, 64, 256)


def _which(model, levels) -> str:
    if levels is None:
        return "all levels"
    return ", ".join(f"{model.level_res_m[i] / 1000:g} km" for i in levels) + (" level" if len(levels) == 1 else " levels")


def lambda_sweep_site(model, S: dict, OBS: dict, scales=SCALES, levels=None, dist_edges=(0, 0.25, 0.5, 1, 2, np.inf),
                      min_px: int = 200, amp=None) -> pd.DataFrame:
    """One dense prediction per multiplier on a real site. Rows: lambda multiplier; columns: RMSE, ME,
    cov95, NLL and RMSE per class of distance to the nearest ATL08 pixel. df.attrs: levels, lam0,
    dist_labels, which (for the figure)."""
    assert 1 in scales, "keep 1 (= the trained λ) in the scales: it is the reference"
    sel = score_mask(S, OBS, {})
    tgt = S["target"][sel].astype(np.float64)
    dist = distance_to_obs_km(OBS["obs_mask"], S["pixel_m"])[sel]
    b = np.digitize(dist, np.asarray(dist_edges)[1:-1])
    keep = [j for j in range(len(dist_edges) - 1) if (b == j).sum() >= min_px]
    lab = [f"{dist_edges[j]:g}–{dist_edges[j + 1]:g} km".replace("inf", "∞") for j in keep]
    lam0 = model.log_lam.exp().detach().cpu().numpy().copy()
    rows = {}
    for s in scales:
        with scaled_lambda(model, s, levels):
            mu, sg = run_calibrator(model, S["Xn"], OBS["obs_dh"], OBS["obs_mask"], amp=amp)
        e = mu[sel].astype(np.float64) - tgt
        m = error_metrics(e, sg[sel], crps=False)
        r = dict(RMSE=m["RMSE"], ME=m["ME"], cov95=m["Cov. 95%"], NLL=m["NLL"])
        r.update({f"RMSE {l}": np.sqrt(np.mean(e[b == j] ** 2)) for l, j in zip(lab, keep)})
        rows[s] = r
        print(f"  λ × {s:<5g} RMSE {r['RMSE']:.3f} m")
    back = np.abs(model.log_lam.exp().detach().cpu().numpy() - lam0).max()
    print(f"check: trained λ restored, max |difference| {back:.1e} (should be 0)")
    df = pd.DataFrame(rows).T
    df.index.name = "λ multiplier"
    df.attrs.update(levels=levels, lam0=lam0, dist_labels=lab, which=_which(model, levels))
    best = df["RMSE"].idxmin()
    print(f"\nλ scaled on: {df.attrs['which']} | trained λ {np.round(lam0, 3)} | lowest RMSE at λ × {best:g} "
          f"({df.loc[best, 'RMSE']:.3f} m vs {df.loc[1, 'RMSE']:.3f} m as trained)")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df.to_string())
    return df


def lambda_sweep_blocks(model, episodes: list[dict], pixel_m: float, scales=SCALES, levels=None,
                        dist_edges=(0, 0.5, 1, 2, np.inf), min_px: int = 200, amp=None) -> pd.DataFrame:
    """The same sweep pooled over Site A block episodes (cell 74), plus an "empty" row (no ICESat)."""
    assert 1 in scales, "keep 1 (= the trained λ) in the scales: it is the reference"
    prep = []
    for e in episodes:
        tgt = e["target"][0, 0].cpu().numpy()
        obs = e["obs_mask"][0, 0].cpu().numpy().astype(bool)
        sel = e["valid"][0, 0].cpu().numpy().astype(bool) & ~obs
        if "score_mask" in e:
            sel &= e["score_mask"][0, 0].cpu().numpy().astype(bool)
        prep.append((e, tgt, sel, distance_to_obs_km(obs, pixel_m)[sel]))
    b = np.digitize(np.concatenate([p[3] for p in prep]), np.asarray(dist_edges)[1:-1])
    labels = [f"{dist_edges[j]:g}–{dist_edges[j + 1]:g} km" for j in range(len(dist_edges) - 1)]

    def collect(empty):
        E, SG = [], []
        for e, tgt, sel, _ in prep:
            mu, sg = predict_episode(model, e, empty, amp)
            E.append((mu - tgt)[sel]); SG.append(sg[sel])
        err, sig = np.concatenate(E), np.concatenate(SG)
        m = error_metrics(err, sig, crps=False)
        r = dict(RMSE=m["RMSE"], ME=m["ME"], cov95=m["Cov. 95%"], NLL=m["NLL"])
        r.update({f"RMSE {l}": np.sqrt(np.mean(err[b == j] ** 2)) if (b == j).sum() >= min_px else np.nan
                  for j, l in enumerate(labels)})
        return r

    rows = {"empty": collect(True)}
    for s in scales:
        with scaled_lambda(model, s, levels):
            rows[s] = collect(False)
    df = pd.DataFrame(rows).T
    df.index.name = "λ multiplier"
    df.attrs.update(levels=levels, dist_labels=labels, which=_which(model, levels))
    best = min(scales, key=lambda s: rows[s]["RMSE"])
    gain = rows[1]["RMSE"] - rows[best]["RMSE"]
    print(f"λ sweep on {df.attrs['which']} | lowest RMSE at λ × {best:g}: {rows[best]['RMSE']:.3f} m vs "
          f"{rows[1]['RMSE']:.3f} m as trained ({gain:+.3f} m)")
    if best > 1 and gain > 0.05:
        print("-> more shrinkage helps: the node fits over-adapt (lambda under-regularised)")
    elif best < 1 and gain > 0.05:
        print("-> less shrinkage helps: the nodes are over-regularised")
    else:
        print("-> the trained lambda is (close to) the best: over-adaptation is not what limits the model")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df.to_string())
    return df


# =============================================================================
# ATL08 noise study (cells 101-104, 125-128)
# =============================================================================
def describe_noise(d, label: str) -> dict:
    """One line of statistics of ATL08 - lidar residuals d (cell 101)."""
    r = residual_stats(d)
    print(f"{label:22s} n {r['n']:7,d} | median {r['median']:+6.2f} | RMSE {r['RMSE']:5.2f} | "
          f"robust sd {r['robust_sd']:5.2f} | |d| > 5 m {r['share_gt5']:5.1%}")
    return r


def residual_pools(cal: pd.DataFrame, edges=EMP_EDGES, min_n: int = 200) -> list[np.ndarray]:
    """Measured ATL08 - lidar residuals per canopy class, to resample from (cell 102).
    cal: segments with 'diff' (evaluate.compare_with_lidar) and 'canopy_gee' (evaluate.sample_band_at).
    Classes with < min_n segments borrow the nearest class's pool."""
    d = cal["diff"].to_numpy(np.float64)
    c = np.nan_to_num(cal["canopy_gee"].to_numpy(np.float64), nan=0.0).clip(min=0)
    cls = np.digitize(c, np.asarray(edges)[1:-1])
    n_cls = len(edges) - 1
    raw = [d[cls == k] for k in range(n_cls)]
    ok = [k for k in range(n_cls) if len(raw[k]) >= min_n]
    assert ok, "no canopy class has enough calibration segments: lower min_n"
    pools = []
    print(f"{'canopy class':14s} {'own n':>7s}  {'sampled from':12s} {'median':>7s} {'robust sd':>9s}")
    for k in range(n_cls):
        use = k if k in ok else min(ok, key=lambda j: abs(j - k))
        p = raw[use]
        pools.append(p)
        src = "own" if use == k else f"class {use}"
        print(f"{edges[k]:g}-{edges[k + 1]:g} m{'':6s} {len(raw[k]):7,d}  {src:12s} {np.median(p):+7.2f} "
              f"{residual_stats(p)['robust_sd']:9.2f}")
    return pools


def gauss_values(target, obs_mask, rng, sigma: float = 1.0) -> np.ndarray:
    """Lidar delta_h + N(0, sigma^2) at the track pixels (the training noise model)."""
    out = np.zeros_like(target)
    out[obs_mask] = target[obs_mask] + sigma * rng.standard_normal(int(obs_mask.sum()))
    return out


def empirical_values(target, obs_mask, canopy, rng, pools, edges=EMP_EDGES) -> np.ndarray:
    """Lidar delta_h + one measured ATL08 residual per track pixel, drawn from its canopy class's pool."""
    out = np.zeros_like(target)
    cls = np.digitize(canopy[obs_mask], np.asarray(edges)[1:-1])
    noise = np.zeros(cls.size, np.float32)
    for k, pool in enumerate(pools):
        m = cls == k
        if m.any():
            noise[m] = rng.choice(pool, int(m.sum()), replace=True)
    out[obs_mask] = target[obs_mask] + noise
    return out


def run_noise_conditions(model, S: dict, OBS: dict, pools, train_sigma: float = 1.0, n_seeds: int = 5,
                         amp=None) -> dict[str, list[dict]]:
    """Same track pixels, same scored pixels, only the values at the tracks differ (cell 103):
       empty (network alone) | (i) real ATL08 | (ii) lidar + Gaussian train_sigma (n_seeds) |
       (iii) lidar + empirical residuals (n_seeds) | (iv) lidar, no noise.
    Returns RUNS: {condition: [dict(metrics, err, lv), ...]} (one dict per seed)."""
    score = S["valid"] & ~OBS["obs_mask"]
    tgt, om = S["target"], OBS["obs_mask"]
    runs: dict[str, list[dict]] = {}

    def record(name, obs_dh, obs_mask):
        mu, sg = run_calibrator(model, S["Xn"], obs_dh, obs_mask, amp=amp)
        err, s = (mu - tgt)[score], sg[score]
        runs.setdefault(name, []).append(dict(error_metrics(err, s), err=err, lv=2.0 * np.log(s)))

    record("empty (network alone)", np.zeros_like(tgt), np.zeros_like(om))
    record("(i) real ATL08", OBS["obs_dh"], om)
    for s in range(n_seeds):
        record(f"(ii) lidar + Gaussian {train_sigma:g} m", gauss_values(tgt, om, np.random.default_rng(s), train_sigma), om)
        record("(iii) lidar + empirical", empirical_values(tgt, om, S["canopy"], np.random.default_rng(100 + s), pools), om)
    record("(iv) lidar, no noise", tgt, om)

    print(f"scored on {score.sum():,} off-track pixels with valid lidar | {om.sum():,} track pixels written\n")
    print(f"{'condition':30s} {'RMSE':>14s} {'ME':>8s} {'RMS σ':>7s} {'Cov. 95%':>8s} {'NLL':>7s}")
    for name, rs in runs.items():
        r = np.array([x["RMSE"] for x in rs])
        sd = f" ± {r.std():.3f}" if len(rs) > 1 else " " * 8
        avg = lambda k: np.mean([x[k] for x in rs])
        print(f"{name:30s} {r.mean():6.3f}{sd} {avg('ME'):+8.3f} {avg('RMS σ'):7.3f} "
              f"{avg('Cov. 95%'):8.3f} {avg('NLL'):7.3f}")
    return runs


def noise_verdict(runs: dict, train_sigma: float = 1.0, tol_m: float = 0.05) -> dict:
    """The differences between the conditions and what they mean for retraining (cell 104)."""
    R = {n: np.mean([x["RMSE"] for x in rs]) for n, rs in runs.items()}
    C95 = {n: np.mean([x["Cov. 95%"] for x in rs]) for n, rs in runs.items()}
    k_empty, k_real, k_clean = "empty (network alone)", "(i) real ATL08", "(iv) lidar, no noise"
    k_gauss, k_emp = f"(ii) lidar + Gaussian {train_sigma:g} m", "(iii) lidar + empirical"
    tol = max(tol_m, 2 * np.std([x["RMSE"] for x in runs[k_gauss]]))
    out = dict(tolerance=tol, adapt_gain=R[k_empty] - R[k_real], gap=R[k_real] - R[k_gauss],
               emp_left=R[k_real] - R[k_emp], headroom=R[k_real] - R[k_clean])
    print(f"tolerance {tol:.3f} m\n")
    print(f"real ICESat gain over network alone   empty - (i)  {out['adapt_gain']:+.3f} m")
    print(f"real vs trained-for noise             (i) - (ii)   {out['gap']:+.3f} m")
    print(f"left unexplained by empirical noise   (i) - (iii)  {out['emp_left']:+.3f} m")
    print(f"headroom to perfect ICESat            (i) - (iv)   {out['headroom']:+.3f} m")
    print(f"Cov. 95%: real {C95[k_real]:.3f} | trained-for noise {C95[k_gauss]:.3f} | empirical {C95[k_emp]:.3f}\n")
    if out["adapt_gain"] < tol:
        print("NOTE: real ICESat barely improves on the network alone. Check that first (support pixels,\n"
              "      datum, the adaptation itself) before drawing conclusions about the noise model.")
    if out["gap"] <= tol:
        print("VERDICT: the Gaussian training noise is NOT limiting -> no retrain needed.")
    elif abs(out["emp_left"]) <= tol:
        print("VERDICT: real noise costs accuracy AND the empirical simulator reproduces it ->\n"
              "         retraining with empirical noise is justified.")
    else:
        print("VERDICT: real noise costs accuracy, but per-pixel empirical noise does NOT reproduce it ->\n"
              "         the difference is in something per-pixel resampling misses (errors correlated along a\n"
              "         track, a site-level bias).")
    if C95[k_real] < C95[k_gauss] - 0.03:
        print("ALSO: coverage drops with real ATL08 -> σ is overconfident with real data, even if RMSE is close.")
    return out


COND_LABELS = (("empty", "No ATL08 (network only)"), ("(i)", "Real ATL08"), ("(ii)", "Lidar + Gaussian ε"),
               ("(iii)", "Lidar + empirical ATL08 ε"), ("(iv)", "Lidar, no noise"))
SEG, PIX = "ATL08 segments", "30 m pixels (model input)"


def condition_label(key: str, train_sigma: float | None = None) -> str:
    """Name of a run_noise_conditions key in the paper."""
    for p, lab in COND_LABELS:
        if key.startswith(p):
            return f"Lidar + Gaussian ε, σ_ε = {train_sigma:g} m (as trained)" if (p == "(ii)" and train_sigma) else lab
    return key


def noise_stats_table(real_b, canopy_b, cal_diff, cal_canopy, edges=NOISE_CLASS_EDGES, min_n: int = 50):
    """The real noise ε = ATL08 - lidar per canopy class, at two scales (cell 128):
    real_b / canopy_b per 30 m track pixel (as the model receives it); cal_diff / cal_canopy per ATL08
    segment. Returns (formatted table, numbers)."""
    rows = []
    for scale, d, c in ((SEG, np.asarray(cal_diff), np.asarray(cal_canopy)), (PIX, np.asarray(real_b), np.asarray(canopy_b))):
        cls = np.digitize(np.nan_to_num(c, nan=0.0), np.asarray(edges)[1:-1])
        rows.append(dict(Scale=scale, Canopy="all", **residual_stats(d)))
        for j in range(len(edges) - 1):
            m = cls == j
            if m.sum() >= min_n:
                rows.append(dict(Scale=scale, Canopy=f"{edges[j]:g}–{edges[j + 1]:g} m".replace("inf", "∞"),
                                 **residual_stats(d[m])))
    df = pd.DataFrame(rows).set_index(["Scale", "Canopy"])
    show = df.copy()
    show["n"] = show["n"].map("{:,.0f}".format)
    for k in ("median", "mean"):
        show[k] = show[k].map("{:+.2f}".format)
    for k in ("RMSE", "robust_sd"):
        show[k] = show[k].map("{:.2f}".format)
    show["share_gt5"] = show["share_gt5"].map("{:.1%}".format)
    show.columns = ["n", "Median ε (m)", "Mean ε (m)", "RMS ε (m)", "Robust SD of ε (m)", "|ε| > 5 m"]
    print(show.to_string())
    return show, df


def noise_condition_table(runs: dict, train_sigma: float | None = None) -> pd.DataFrame:
    """One row per condition: mean over noise seeds; RMSE also ± seed SD where there are several seeds."""
    rows = {}
    for key, rs in runs.items():
        ms = [error_metrics(r["err"], np.exp(0.5 * r["lv"])) for r in rs]
        avg = {k: np.mean([m[k] for m in ms]) for k in ("RMSE", "ME", "RMS σ", "Cov. 95%", "NLL", "CRPS")}
        sd = np.std([m["RMSE"] for m in ms]) if len(ms) > 1 else None
        rows[condition_label(key, train_sigma)] = {
            "RMSE (m)": f"{avg['RMSE']:.3f}" + (f" ± {sd:.3f}" if sd is not None else ""),
            "ME (m)": f"{avg['ME']:+.3f}", "RMS σ (m)": f"{avg['RMS σ']:.3f}", "Cov. 95%": f"{avg['Cov. 95%']:.3f}",
            "NLL": f"{avg['NLL']:.3f}", "CRPS (m)": f"{avg['CRPS']:.3f}", "seeds": len(ms)}
    df = pd.DataFrame(rows).T
    print(df.to_string())
    return df


def pretrain_noise_table(models: dict, episodes: list[dict] | None = None, S: dict | None = None,
                         OBS: dict | None = None, amp=None) -> pd.DataFrame:
    """models: {"σ_ε = 0.3 m": model03, "σ_ε = 1.0 m": model} (the pre-training noise level). Columns:
    Site A test blocks (simulated tracks, scored inside the blocks) and/or a real site, each with ATL08 and
    without. Values: RMSE, cov95, NLL (cell 128)."""
    out = {}
    for name, mdl in models.items():
        row = {}
        if episodes is not None:
            for empty in (False, True):
                E, SG = [], []
                for e in episodes:
                    npy = lambda k: e[k][0, 0].detach().cpu().numpy()
                    sel = npy("valid").astype(bool) & ~npy("obs_mask").astype(bool)
                    if "score_mask" in e:
                        sel &= npy("score_mask").astype(bool)
                    mu, sg = predict_episode(mdl, e, empty, amp)
                    E.append((mu - npy("target"))[sel]); SG.append(sg[sel])
                tag = "Site A, no ATL08" if empty else "Site A, simulated ATL08"
                row.update({(tag, k): v for k, v in _m3(np.concatenate(E), np.concatenate(SG)).items()})
        if S is not None:
            sel = S["valid"] & ~OBS["obs_mask"]
            for empty in (False, True):
                p = predict_site(mdl, S, OBS, empty=empty, amp=amp)
                tag = "Site B, no ATL08" if empty else "Site B, real ATL08"
                row.update({(tag, k): v for k, v in _m3((p["mu"] - S["target"])[sel], p["sigma"][sel]).items()})
        out[name] = row
    df = pd.DataFrame(out).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "Pre-training noise"
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df.to_string())
    return df
