"""
Simulated ICESat-2 observations for training and evaluation on Site A.

    simulate_icesat_mask    where ICESat-2 would have ground points over an h x w region
    simulate_observations   track mask & valid LiDAR, with Gaussian noise on delta_h
    sim_kwargs              config.yaml icesat_sim -> keyword arguments of simulate_icesat_mask


"""

from __future__ import annotations

import math

import torch


def simulate_icesat_mask(h: int, w: int, pixel_m: float = 30.0, rgt_spacing_km=(3.0, 40.0),
                         max_heading: float = 15.0, pair_sep_km: float = 3.3, beam_sep_m: float = 90.0,
                         along_step_m: float = 100.0, p_beam_drop: float = 0.2, p_cloud_gap: float = 0.3,
                         gap_len_km=(1.0, 8.0), generator: torch.Generator | None = None,
                         device="cpu") -> torch.Tensor:
    """bool [h, w] mask of simulated ICESat-2 ground points over an h x w pixel region.

    Reference ground tracks (log-uniform spacing in rgt_spacing_km) at a random heading within
    +-max_heading of north-south; each track = 3 beam pairs pair_sep_km apart, each pair = 2 beams
    beam_sep_m apart; one point every along_step_m. Each beam is dropped with p_beam_drop and gets a
    cloud gap (gap_len_km long) with p_cloud_gap.
    """
    rnd = lambda: float(torch.rand((), generator=generator))
    lo, hi = rgt_spacing_km
    S = math.exp(math.log(lo) + rnd() * (math.log(hi) - math.log(lo))) * 1000 / pixel_m   # px
    a = math.radians((2 * rnd() - 1) * max_heading)
    dy, dx = math.cos(a), math.sin(a)            # along track (north-south)
    ny, nx = -math.sin(a), math.cos(a)           # across track (east-west)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    R = 0.5 * math.hypot(h, w)
    P = pair_sep_km * 1000 / pixel_m
    b = beam_sep_m / pixel_m
    step = along_step_m / pixel_m

    phase = rnd() * S
    offsets = []
    for k in range(math.floor((-R - P - phase) / S), math.ceil((R + P - phase) / S) + 1):
        c = phase + k * S
        for p in (-P, 0.0, P):
            for s in (-b / 2, b / 2):
                if abs(c + p + s) <= R + 1:
                    offsets.append(c + p + s)

    t = torch.arange(-R, R + step, step, device=device)
    mask = torch.zeros(h, w, device=device, dtype=torch.bool)
    for o in offsets:
        if rnd() < p_beam_drop:
            continue
        keep = torch.ones_like(t, dtype=torch.bool)
        if rnd() < p_cloud_gap:
            L = (gap_len_km[0] + rnd() * (gap_len_km[1] - gap_len_km[0])) * 1000 / pixel_m
            t0 = -R + rnd() * 2 * R
            keep &= ~((t >= t0) & (t < t0 + L))

        tt = t[keep]
        yy = (cy + o * ny + tt * dy).round().long()
        xx = (cx + o * nx + tt * dx).round().long()
        inb = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        mask[yy[inb], xx[inb]] = True

    return mask


def simulate_observations(target: torch.Tensor, valid: torch.Tensor, generator: torch.Generator,
                          pixel_m: float, noise_m: float = 1.0, **sim) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated ICESat over a crop: obs_mask = tracks & valid LiDAR, obs_dh = target + N(0, noise_m^2).
    target, valid: [h, w]. Returns obs_mask [h, w] bool, obs_dh [h, w] float (noise everywhere,
    only read where obs_mask is True). Random order: tracks first, then noise (as in the notebook)."""
    h, w = valid.shape
    tracks = simulate_icesat_mask(h, w, pixel_m=pixel_m, generator=generator, device=valid.device, **sim)
    obs_mask = tracks & valid.bool()
    noise = torch.randn(h, w, generator=generator).to(target.device) * noise_m
    return obs_mask, target + noise


def sim_kwargs(icesat_sim: dict | None) -> dict:
    """config.yaml `icesat_sim:` section -> keyword arguments of simulate_icesat_mask."""
    if not icesat_sim:
        return {}
    c = dict(icesat_sim)
    if "max_heading_deg" in c:
        c["max_heading"] = c.pop("max_heading_deg")
    for k in ("rgt_spacing_km", "gap_len_km"):
        if k in c:
            c[k] = tuple(c[k])
    return c
