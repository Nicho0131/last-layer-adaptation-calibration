"""
Running a trained model: on a whole area, on evaluation episodes, and end-to-end on new data.

    predict_dense        the one place that calls model.forward for dense maps (no_grad, AMP, window batch)
    run_calibrator       numpy inputs -> mu, sigma maps (cell 82)
    predict_episode      one evaluation episode, with its tracks or an empty mask (cells 64, 66)
    site_arrays          Site (src.preprocessing.model_inputs) -> the S / OBS dicts used by every table
    predict_site         one prediction on S / OBS, optionally inside an ablation switch (cell 129)
    predict_methods      uncorrected DSM, reference DEMs, network only, + global offset, full (cell 106)
    predict_site_a_full  Site A as one map with one simulated track layout (cell 57)
    calibrate            checkpoint + features + ATL08 -> DTM, delta_h, sigma GeoTIFFs (cell 82)

Dicts used throughout src/evaluation:
    S    site arrays: Xn, dsm, dsm_ok, target, valid, canopy, bldg, transform, crs, H, W, pixel_m, stats
    OBS  observations: obs_dh, obs_mask
    P    {method name: dict(mu=[H, W], sigma=[H, W] or None)}
"""

from __future__ import annotations

import gc
from contextlib import nullcontext

import numpy as np
import torch

from dataset.reference_dems import read_baseline
from src.evaluation.baselines import network_offset
from src.preprocessing.model_inputs import Site, load_inference_inputs, write_predictions
from src.training.simulate import simulate_observations
from src.training.trainer import Amp, setup_amp


# =============================================================================
# the model call
# =============================================================================
def _amp_for(model, amp: Amp | None) -> Amp:
    """AMP as in the notebook (on for CUDA) unless one is given."""
    return amp if amp is not None else setup_amp(next(model.parameters()).device, verbose=False)


@torch.no_grad()
def predict_dense(model, x, obs_dh, obs_mask, origin_yx=(0, 0), dsm_hw=None, amp: Amp | None = None,
                  win_batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dense mu and sigma [H, W] (numpy, CPU) for x [1, C, H, W] with observations [1, 1, H, W].
    x may stay on the CPU (windows go to the model's device). win_batch temporarily overrides the
    model's windows per encoder call (more = faster, more GPU memory)."""
    model.eval()
    amp = _amp_for(model, amp)
    old = model.win_batch
    if win_batch is not None:
        model.win_batch = win_batch
    try:
        with amp.ctx():
            mu, lv = model(x, obs_dh, obs_mask, origin_yx=origin_yx, dsm_hw=dsm_hw)
    finally:
        model.win_batch = old
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return mu[0, 0].float().cpu().numpy(), np.exp(0.5 * lv[0, 0].float().cpu().numpy())


def run_calibrator(model, X_norm: np.ndarray, obs_dh: np.ndarray, obs_mask: np.ndarray, win_batch: int = 256,
                   amp: Amp | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Whole area from numpy arrays: X_norm [C, H, W], obs_dh / obs_mask [H, W] -> mu, sigma [H, W].
    Pass 1: closed-form last-layer adaptation from the observed pixels; pass 2: Hann-blended read."""
    x = torch.from_numpy(np.ascontiguousarray(X_norm, np.float32))[None]
    od = torch.from_numpy(np.ascontiguousarray(obs_dh, np.float32))[None, None]
    om = torch.from_numpy(np.ascontiguousarray(obs_mask, bool))[None, None]
    return predict_dense(model, x, od, om, (0, 0), tuple(X_norm.shape[1:]), amp, win_batch)


def predict_episode(model, ep: dict, empty: bool = False, amp: Amp | None = None) -> tuple[np.ndarray, np.ndarray]:
    """mu, sigma [h, w] for one episode (src.training.episodes), with its tracks or with an empty mask."""
    o = torch.zeros_like(ep["obs_mask"]) if empty else ep["obs_mask"]
    d = torch.zeros_like(ep["obs_dh"]) if empty else ep["obs_dh"]
    return predict_dense(model, ep["x"], d, o, ep["origin"], ep.get("dsm_hw"), amp)


# =============================================================================
# Site -> S / OBS dicts
# =============================================================================
def site_arrays(site: Site) -> tuple[dict, dict | None]:
    """The S and OBS dicts (numpy) of a Site. OBS is None when the site has no real observations (Site A)."""
    npy = lambda t: None if t is None else t.detach().cpu().numpy()
    H, W = site.hw
    S = dict(Xn=npy(site.Xn), dsm=site.dsm, dsm_ok=site.dsm_ok, target=npy(site.target), valid=npy(site.valid),
             transform=site.transform, crs=site.crs, prof=site.profile, H=H, W=W, pixel_m=site.pixel_m,
             stats=site.stats, name=site.name)
    for k in ("canopy", "bldg", "slope"):
        if k in site.extra:
            S[k] = np.asarray(site.extra[k], np.float32)
    OBS = None
    if site.obs_mask is not None:
        OBS = dict(obs_dh=npy(site.obs_dh).astype(np.float32), obs_mask=npy(site.obs_mask).astype(bool))
    return S, OBS


# =============================================================================
# predictions on S / OBS
# =============================================================================
def predict_site(model, S: dict, OBS: dict, empty: bool = False, ctx=None, amp: Amp | None = None) -> dict:
    """dict(mu, sigma) for the whole site; empty=True: no observations (network only).
    ctx: a zero-argument function returning a context manager (an ablation switch) to predict inside."""
    H, W = S["H"], S["W"]
    od = np.zeros((H, W), np.float32) if empty else OBS["obs_dh"]
    om = np.zeros((H, W), bool) if empty else OBS["obs_mask"]
    with (ctx() if ctx is not None else nullcontext()):
        mu, sg = run_calibrator(model, S["Xn"], od, om, amp=amp)
    return dict(mu=mu.astype(np.float32), sigma=sg.astype(np.float32))


def predict_methods(S: dict, OBS: dict, models: dict, baselines: dict | None = None, box=None,
                    amp: Amp | None = None) -> dict:
    """Every method on the same site (cell 106):
      "uncorrected DSM"          mu = 0
      <baseline>                 mu = DEM - DSM, no sigma (baselines: {"FABDEM": path, ...} on the site grid)
      "<model> network alone"    empty observation mask
      "<model> + global offset"  network alone + mean(ATL08 delta_h - mu) over the ATL08 pixels
      "<model> full"             closed-form last-layer adaptation with the ATL08 pixels
    models: {"M0": model, ...}. box: the box given to prepare_site, so the baselines are cut the same way."""
    P = {"uncorrected DSM": dict(mu=np.zeros((S["H"], S["W"]), np.float32), sigma=None)}
    for name, path in (baselines or {}).items():
        dem = read_baseline(path, S, box)
        P[name] = dict(mu=np.where(np.isfinite(dem), dem - S["dsm"], np.nan).astype(np.float32), sigma=None)
    for name, m in models.items():
        P[f"{name} network alone"] = net = predict_site(m, S, OBS, empty=True, amp=amp)
        P[f"{name} + global offset"], b = network_offset(net, OBS)
        P[f"{name} full"] = full = predict_site(m, S, OBS, amp=amp)
        print(f"{name}: global offset {b:+.3f} m | max |full - network alone| "
              f"{np.abs(full['mu'] - net['mu']).max():.2f} m")
    return P


def predict_site_a_full(model, site: Site, seed: int = 0, noise_m: float = 1.0, win_batch: int = 128,
                        amp: Amp | None = None, sim: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mu, sigma and the simulated ICESat mask over the whole of Site A: one track layout, one TTT write,
    one dense read (cell 57). Errors in the training strip are in-sample."""
    gen = torch.Generator().manual_seed(seed)
    obs, obs_dh = simulate_observations(site.target.cpu(), site.valid.cpu(), gen, site.pixel_m, noise_m,
                                        **(sim or {}))
    dev = site.Xn.device
    mu, sg = predict_dense(model, site.Xn[None], obs_dh.to(dev)[None, None].float(), obs.to(dev)[None, None],
                           (0, 0), site.hw, amp, win_batch)
    return mu, sg, obs.numpy()


# =============================================================================
# end to end on a new area
# =============================================================================
def calibrate(ckpt_path: str, feat_path: str, atl08_files=None, dsm_path: str | None = None,
              out_prefix: str | None = None, geoid_epsg: int = 3855, points=None, device=None,
              win_batch: int = 256) -> dict:
    """Checkpoint + 8-band feature stack + ATL08 -> DTM = DSM + mu, delta_h = mu and sigma (cell 82).
    Writes <out_prefix>_dtm.tif, _delta_h.tif, _sigma.tif if out_prefix is given. Returns the arrays."""
    from models.initialize_calibrator import load_calibrator

    model, stats, cfg = load_calibrator(ckpt_path, device)
    site = load_inference_inputs(feat_path, stats, cfg["pixel_m"], atl08_files=atl08_files, points=points,
                                 dsm_path=dsm_path, geoid_epsg=geoid_epsg)
    mu, sigma = run_calibrator(model, site.Xn.numpy(), site.obs_dh.numpy(), site.obs_mask.numpy(), win_batch)
    t0, t1 = stats["target_min"], stats["target_max"]
    mu = np.where(site.dsm_ok, mu, np.nan)
    sigma = np.where(site.dsm_ok, sigma, np.nan)
    dtm = np.where(site.dsm_ok, site.dsm + np.clip(mu, t0, t1), np.nan)
    if out_prefix:
        write_predictions(site, mu, sigma, out_prefix)
    return dict(dtm=dtm, delta_h=mu, sigma=sigma, obs_dh=site.obs_dh.numpy(), obs_mask=site.obs_mask.numpy(),
                dsm=site.dsm, profile=site.profile)
