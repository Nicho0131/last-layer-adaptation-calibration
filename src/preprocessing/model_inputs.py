"""
Model-ready inputs for one area: training (Site A), test sites, and inference on new areas.

The dataset/ functions make arrays from rasters on disk; this module packs them into a Site:
standardised inputs, target, valid mask and (for real ICESat) the observations, as torch tensors,
plus the grid information needed to map predictions back to GeoTIFFs.

    load_site_a(cfg)                     Site A: training, validation and test strips (cells 19-21)
    load_test_site(cfg, "tile_1", stats) a LiDAR test site with real ATL08 (cells 100, 106)
    load_inference_inputs(...)           a new area: features + ATL08, no LiDAR (cell 82, calibrate)
    model_inputs(site)                   the tensors ProbabilisticCalibrator.forward expects

Cutting training / evaluation EPISODES out of a Site (crops, simulated tracks, query pixels)
belongs in src/training/episodes.py, not here.

"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dataset.build_inputs import (build_inputs, means_stds, prepare_site, prepare_site_a, read_raster,
                                  standardize)
from src.preprocessing.icesat import load_atl08, rasterize_obs, site_icesat


# =============================================================================
# the Site container
# =============================================================================
@dataclass
class Site:
    """One area, ready for the model.

    Xn       [C, H, W] float32  standardised inputs (Site A's statistics, always)
    target   [H, W] float32     delta_h = DTM - DSM, 0 where not valid      (None for inference)
    valid    [H, W] bool        LiDAR target exists and is inside the range (None for inference)
    obs_dh   [H, W] float32     real ATL08 delta_h, 0 where no observation  (None for Site A)
    obs_mask [H, W] bool        True where a real ATL08 observation exists  (None for Site A)
    dsm, dsm_ok                 raw DSM (numpy) for DTM = DSM + mu
    transform, crs, profile     the grid, for writing GeoTIFFs
    boxes                       Site A only: {"train", "val", "test"} -> (y0, x0, y1, x1)
    extra                       canopy, buildings, features ... for analysis
    """
    name: str
    Xn: torch.Tensor
    dsm: np.ndarray
    dsm_ok: np.ndarray
    transform: Any
    crs: Any
    profile: dict
    pixel_m: float
    stats: dict
    target: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    obs_dh: torch.Tensor | None = None
    obs_mask: torch.Tensor | None = None
    boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def hw(self) -> tuple[int, int]:
        return tuple(self.Xn.shape[-2:])

    @property
    def device(self) -> torch.device:
        return self.Xn.device

    def to(self, device) -> "Site":
        """Move the tensors (Xn, target, valid, obs) to `device`; numpy arrays stay as they are."""
        for k in ("Xn", "target", "valid", "obs_dh", "obs_mask"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        return self

    def summary(self) -> str:
        H, W = self.hw
        s = (f"{self.name}: {W} x {H} px at {self.pixel_m:g} m "
             f"({W * self.pixel_m / 1000:.1f} x {H * self.pixel_m / 1000:.1f} km) | on {self.device}")
        if self.valid is not None:
            s += f" | valid LiDAR {self.valid.float().mean().item():.1%}"
        if self.obs_mask is not None:
            s += f" | ATL08 pixels {int(self.obs_mask.sum()):,}"
        return s


def _fits_on_gpu(t: torch.Tensor, max_frac: float = 0.2) -> bool:
    """The notebook's rule: put the whole area on the GPU only if Xn uses < 20 % of its memory."""
    return (torch.cuda.is_available()
            and t.numel() * t.element_size() < max_frac * torch.cuda.get_device_properties(0).total_memory)


def _place(site: Site, device) -> Site:
    """device=None: GPU if the area fits (else CPU). Otherwise exactly `device`."""
    if device is None:
        device = "cuda" if _fits_on_gpu(site.Xn) else "cpu"
    return site.to(device)


# =============================================================================
# Site A: training / validation / test
# =============================================================================
def load_site_a(cfg: dict, device=None, write_stats: bool = False, check_stats: dict | None = None) -> Site:
    """Site A with its train / val / test boxes (cells 19-21).

    write_stats: also (re)write stats.json; True only when preparing a new training run.
    check_stats: the stats stored in a checkpoint; raises if this data gives different means / stds
                 (i.e. the rasters changed since that model was trained).
    """
    A = prepare_site_a(cfg, write_stats=write_stats)
    if check_stats is not None:
        m0, s0 = means_stds(check_stats)
        m1, s1 = means_stds(A["stats"])
        if not (np.allclose(m0, m1, rtol=1e-4) and np.allclose(s0, s1, rtol=1e-4)):
            raise ValueError("Site A statistics differ from the checkpoint's: the rasters have changed")

    feat, feat_ok = A["features"], A["features_valid"]
    site = Site(
        name="site_a",
        Xn=torch.from_numpy(A["Xn"]),
        target=torch.from_numpy(A["target"]),
        valid=torch.from_numpy(A["valid"]),
        dsm=A["dsm"], dsm_ok=A["dsm_ok"],
        transform=A["profile"]["transform"], crs=A["profile"]["crs"], profile=A["profile"],
        pixel_m=A["pixel_m"], stats=A["stats"],
        boxes=dict(train=A["train_box"], val=A["val_box"], test=A["test_box"]),
        # 0-based bands of the GEE stack: 1 slope, 4 canopy, 7 building height (cells 121, 132)
        extra=dict(features=feat, features_valid=feat_ok,
                   slope=np.where(feat_ok[1], feat[1], np.nan),
                   canopy=np.where(feat_ok[4], feat[4], 0.0),
                   bldg=np.where(feat_ok[7], feat[7], 0.0)),
    )
    _place(site, device)
    print(site.summary())
    return site


# =============================================================================
# test sites with LiDAR and real ATL08
# =============================================================================
def load_test_site(cfg: dict, site_key: str, stats: dict, model_pixel_m: float, box=None,
                   with_atl08: bool = True, device="cpu") -> Site:
    """A sites: entry of config.yaml (e.g. "tile_1" = Site B), standardised with Site A's statistics,
    with the real ATL08 observations rasterised on its grid (filter from config atl08.filter).
    Kept on the CPU by default: the model moves windows to the GPU itself."""
    sc = cfg["sites"][site_key]
    S = prepare_site(sc, stats, model_pixel_m, box=box)
    profile = dict(S["prof"], transform=S["transform"], height=S["H"], width=S["W"])   # cropped grid if box
    site = Site(
        name=site_key,
        Xn=torch.from_numpy(S["Xn"]),
        target=torch.from_numpy(S["target"]),
        valid=torch.from_numpy(S["valid"]),
        dsm=S["dsm"], dsm_ok=S["dsm_ok"],
        transform=S["transform"], crs=S["crs"], profile=profile,
        pixel_m=S["pixel_m"], stats=stats,
        extra=dict(canopy=S["canopy"], bldg=S["bldg"], box=box),
    )
    if with_atl08:
        a = cfg["atl08"]
        obs = site_icesat(S, sc["atl08_dir"], flt=a["filter"], window=a["along_track_window"],
                          geoid_epsg=cfg["datum"]["dsm_geoid_epsg"])
        site.obs_dh = torch.from_numpy(obs["obs_dh"])
        site.obs_mask = torch.from_numpy(obs["obs_mask"])
        site.extra["n_segments"] = obs["n_segments"]
    _place(site, device)
    print(site.summary())
    return site


# =============================================================================
# inference on a new area (no LiDAR)
# =============================================================================
def load_inference_inputs(feat_path: str, stats: dict, model_pixel_m: float, atl08_files=None,
                          points=None, dsm_path: str | None = None, geoid_epsg: int = 3855,
                          max_unc_m: float = 5.0, name: str = "area") -> Site:
    """The data part of calibrate() (cell 82), without running the model.

    feat_path:   8-band GEE stack on the DSM grid; the DSM is band 1 unless dsm_path is given.
    atl08_files: list of .h5 files or a glob pattern (read with load_atl08), or
    points:      (x, y, h_ground) already in the DSM's CRS and geoid.
    """
    import rasterio

    feat, feat_ok, prof = read_raster(feat_path)
    assert feat.shape[0] == 8, f"expected 8 feature bands, got {feat.shape[0]}"
    if dsm_path is None:
        dsm, dsm_ok = feat[0], feat_ok[0]
    else:
        dsm, dsm_ok, p2 = read_raster(dsm_path, 1)
        assert dsm.shape == feat.shape[1:] and p2["transform"].almost_equals(prof["transform"]), \
            "DSM and features are on different grids"
    crs = rasterio.crs.CRS.from_user_input(prof["crs"])
    px = abs(prof["transform"].a)
    assert crs.is_projected, "rasters must be in a projected CRS (metres)"
    assert abs(px / model_pixel_m - 1) < 0.05, f"pixel {px:.2f} m, model trained at {model_pixel_m:.2f} m"
    print(f"DSM {dsm.shape[0]} x {dsm.shape[1]} px at {px:.1f} m | {crs.to_string()}")

    X, X_ok = build_inputs(dsm, dsm_ok, feat, feat_ok)                 # same mapping as training
    Xn = standardize(X, X_ok, *means_stds(stats))                      # TRAINING statistics
    del X, feat

    if points is not None:
        x, y, h = points
    else:
        files = sorted(glob.glob(atl08_files)) if isinstance(atl08_files, str) else list(atl08_files or [])
        x, y, h = load_atl08(files, crs.to_wkt(), geoid_epsg, max_unc_m)
    obs_dh, obs_mask = rasterize_obs(x, y, h, dsm, dsm_ok, prof["transform"],
                                     (stats["target_min"], stats["target_max"]))
    print(f"ICESat pixels: {int(obs_mask.sum()):,} ({obs_mask.mean():.3%} of the DSM)")

    site = Site(name=name, Xn=torch.from_numpy(np.ascontiguousarray(Xn)),
                dsm=dsm, dsm_ok=dsm_ok, transform=prof["transform"], crs=crs, profile=prof,
                pixel_m=px, stats=stats,
                obs_dh=torch.from_numpy(obs_dh), obs_mask=torch.from_numpy(obs_mask))
    print(site.summary())
    return site


# =============================================================================
# what the model's forward() takes
# =============================================================================
def model_inputs(site: Site, obs_dh: torch.Tensor | None = None, obs_mask: torch.Tensor | None = None
                 ) -> dict[str, Any]:
    """Whole-area inputs for ProbabilisticCalibrator.forward:
    x [1, C, H, W], obs_dh [1, 1, H, W], obs_mask [1, 1, H, W], origin_yx, dsm_hw.
    obs_dh / obs_mask default to the site's own (real ATL08); pass others to test e.g. simulated tracks.
    With no observations at all, the model runs as "network only"."""
    obs_dh = site.obs_dh if obs_dh is None else obs_dh
    obs_mask = site.obs_mask if obs_mask is None else obs_mask
    if obs_mask is None:                                   # no ATL08: empty observations
        obs_mask = torch.zeros(site.hw, dtype=torch.bool, device=site.device)
        obs_dh = torch.zeros(site.hw, dtype=torch.float32, device=site.device)
    return dict(x=site.Xn[None], obs_dh=obs_dh[None, None].float(), obs_mask=obs_mask[None, None].bool(),
                origin_yx=(0, 0), dsm_hw=site.hw)


def write_predictions(site: Site, mu: np.ndarray, sigma: np.ndarray, out_prefix: str) -> dict[str, str]:
    """DTM = DSM + clip(mu) and the two maps -> <out_prefix>_dtm.tif, _delta_h.tif, _sigma.tif (cell 82)."""
    from dataset.build_inputs import save_tif

    t0, t1 = site.stats["target_min"], site.stats["target_max"]
    mu = np.where(site.dsm_ok, mu, np.nan)
    sigma = np.where(site.dsm_ok, sigma, np.nan)
    dtm = np.where(site.dsm_ok, site.dsm + np.clip(mu, t0, t1), np.nan)
    out = {}
    for kind, arr in (("dtm", dtm), ("delta_h", mu), ("sigma", sigma)):
        out[kind] = save_tif(f"{out_prefix}_{kind}.tif", arr, site.profile)
        print("saved", out[kind])
    return out