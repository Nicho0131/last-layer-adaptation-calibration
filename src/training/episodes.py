"""
Episodes: what the model sees in one training or evaluation step on Site A, MAML inspired method.

An episode is a crop of a Site with simulated ICESat-2 observations on it:
    x [1, C, h, w], target / valid / obs_mask / obs_dh [1, 1, h, w], origin (y, x) of the crop in the
    full DSM, dsm_hw (H, W) of the full DSM (the memory grid is anchored to it).

    sample_episode     random training crop inside a box (cell 26)
    sample_queries     off-track pixels with valid LiDAR to score (cell 26)
    split_into_blocks  equal blocks of ~block_km covering a box (cell 30)
    block_episode      one block + context, scored only inside the block (cell 30)
    make_eval_set      one fixed episode per block of the val / test strip (cell 30)
    make_eval_sets     the validation and test sets from config.yaml
    ensure_icesat      redraw tracks for blocks with too few ICESat pixels (cell 73)

"""

from __future__ import annotations

import numpy as np
import torch

from src.preprocessing.model_inputs import Site
from src.training.simulate import simulate_icesat_mask, simulate_observations, sim_kwargs


# =============================================================================
# training
# =============================================================================
def sample_episode(site: Site, box, gen: torch.Generator, crop_px: int = 2048, min_valid_frac: float = 0.5,
                   noise_m: float = 1.0, device="cuda", sim: dict | None = None,
                   max_tries: int = 50) -> dict:
    """A random crop_px x crop_px crop inside box (y0, x0, y1, x1) with simulated ICESat tracks.
    Crops with less than min_valid_frac valid LiDAR are redrawn (up to max_tries; the last is kept)."""
    y0b, x0b, y1b, x1b = box
    ch, cw = min(crop_px, y1b - y0b), min(crop_px, x1b - x0b)
    for _ in range(max_tries):
        oy = y0b + int(torch.randint(0, y1b - y0b - ch + 1, (1,), generator=gen))
        ox = x0b + int(torch.randint(0, x1b - x0b - cw + 1, (1,), generator=gen))
        val = site.valid[oy:oy + ch, ox:ox + cw]
        if val.float().mean() >= min_valid_frac:
            break

    x = site.Xn[:, oy:oy + ch, ox:ox + cw]
    tgt = site.target[oy:oy + ch, ox:ox + cw]
    obs, obs_dh = simulate_observations(tgt, val, gen, site.pixel_m, noise_m, **(sim or {}))
    return dict(
        x=x[None].to(device),
        target=tgt[None, None].to(device),
        valid=val[None, None].to(device),
        obs_mask=obs[None, None].to(device),
        obs_dh=obs_dh[None, None].to(device),
        origin=(oy, ox),
        dsm_hw=site.hw,
    )


def sample_queries(valid: torch.Tensor, obs_mask: torch.Tensor, n: int) -> torch.Tensor:
    """n off-track pixels with valid LiDAR per sample -> [B, n, 2] (y, x), drawn with replacement.
    Uses the global torch RNG (saved in checkpoints by the trainer)."""
    out = []
    for b in range(valid.shape[0]):
        cand = (valid[b, 0].bool() & ~obs_mask[b, 0].bool()).nonzero()
        if len(cand) == 0:
            raise ValueError(f"sample {b} has no valid off-track pixels to query")
        out.append(cand[torch.randint(len(cand), (n,), device=cand.device)])
    return torch.stack(out)


# =============================================================================
# evaluation blocks
# =============================================================================
def split_into_blocks(box, pixel_m: float, block_km: float = 8.0) -> list[tuple[int, int, int, int]]:
    """Equal-size blocks (y0, x0, y1, x1) tiling box, each about block_km on a side."""
    y0, x0, y1, x1 = box
    bp = block_km * 1000 / pixel_m
    ny, nx = max(1, round((y1 - y0) / bp)), max(1, round((x1 - x0) / bp))
    ey = np.linspace(y0, y1, ny + 1).round().astype(int)
    ex = np.linspace(x0, x1, nx + 1).round().astype(int)
    return [(int(ey[i]), int(ex[j]), int(ey[i + 1]), int(ex[j + 1])) for i in range(ny) for j in range(nx)]


def block_episode(site: Site, block, box, gen: torch.Generator, ctx_px: int = 64, noise_m: float = 1.0,
                  device="cuda", sim: dict | None = None) -> dict:
    """Crop = block + ctx_px of context, clipped to box. score_mask = pixels inside the block."""
    by0, bx0, by1, bx1 = block
    y0b, x0b, y1b, x1b = box
    cy0, cx0 = max(y0b, by0 - ctx_px), max(x0b, bx0 - ctx_px)
    cy1, cx1 = min(y1b, by1 + ctx_px), min(x1b, bx1 + ctx_px)
    ch, cw = cy1 - cy0, cx1 - cx0
    x = site.Xn[:, cy0:cy1, cx0:cx1]
    tgt = site.target[cy0:cy1, cx0:cx1]
    val = site.valid[cy0:cy1, cx0:cx1]
    score = torch.zeros(ch, cw, dtype=torch.bool, device=val.device)
    score[by0 - cy0:by1 - cy0, bx0 - cx0:bx1 - cx0] = True
    obs, obs_dh = simulate_observations(tgt, val, gen, site.pixel_m, noise_m, **(sim or {}))
    return dict(x=x[None].to(device), target=tgt[None, None].to(device), valid=val[None, None].to(device),
                obs_mask=obs[None, None].to(device), obs_dh=obs_dh[None, None].to(device),
                score_mask=score[None, None].to(device), origin=(cy0, cx0), dsm_hw=site.hw, block=block)


def make_eval_set(site: Site, box, seed: int, block_km: float = 8.0, ctx_px: int = 64,
                  min_scored_px: int = 500, noise_m: float = 1.0, n_layouts: int = 1, device="cuda",
                  sim: dict | None = None) -> list[dict]:
    """One fixed episode per block of box (n_layouts > 1: several track layouts per block).
    Blocks with fewer than min_scored_px valid off-track pixels inside the block are skipped."""
    gen = torch.Generator().manual_seed(seed)
    eps = []
    for b_id, block in enumerate(split_into_blocks(box, site.pixel_m, block_km)):
        for k in range(n_layouts):
            e = block_episode(site, block, box, gen, ctx_px, noise_m, device, sim)
            n_scored = int((e["valid"] & ~e["obs_mask"] & e["score_mask"]).sum())
            if n_scored < min_scored_px:
                continue
            e.update(block_id=b_id, layout=k, n_scored=n_scored)
            eps.append(e)
    return eps


def make_eval_sets(site: Site, cfg: dict, device="cuda") -> tuple[list[dict], list[dict]]:
    """VAL_SET and TEST_SET of the notebook, from config.yaml (evaluation, episodes, icesat_sim)."""
    ev, sim = cfg["evaluation"], sim_kwargs(cfg.get("icesat_sim"))
    kw = dict(block_km=ev["block_km"], ctx_px=ev["ctx_px"], min_scored_px=ev["min_scored_px"],
              noise_m=cfg["episodes"]["icesat_noise_m"], device=device, sim=sim)
    val_set = make_eval_set(site, site.boxes["val"], seed=ev["seed"], **kw)
    test_set = make_eval_set(site, site.boxes["test"], seed=ev["seed"], **kw)
    for name, S in (("val", val_set), ("test", test_set)):
        print(f"{name}: {len(S)} blocks | block size (px) "
              f"{[(e['block'][2] - e['block'][0], e['block'][3] - e['block'][1]) for e in S]} | "
              f"ICESat px {[int(e['obs_mask'].sum()) for e in S]}")
    return val_set, test_set


def ensure_icesat(episodes: list[dict], pixel_m: float, min_px: int = 200, noise_m: float = 1.0,
                  seed: int = 77, max_tries: int = 100, sim: dict | None = None) -> list[dict]:
    """Episodes with at least min_px simulated ICESat pixels: blocks below that get new tracks
    (redrawn up to max_tries) and new noise; the others are returned unchanged (cell 73)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    print(f"{'block':>5s} {'ICESat px before':>17s} {'after':>7s} {'of which inside block':>22s} {'tries':>6s}")
    for i, e in enumerate(episodes):
        n_before = int(e["obs_mask"].sum())
        if n_before >= min_px:
            out.append(e)
            print(f"{i:5d} {n_before:17,d} {'kept':>7s}")
            continue

        val = e["valid"][0, 0].bool()
        tgt = e["target"][0, 0]
        ch, cw = val.shape
        for t in range(1, max_tries + 1):
            tracks = simulate_icesat_mask(ch, cw, pixel_m=pixel_m, generator=gen, device=val.device,
                                          **(sim or {}))
            obs = tracks & val
            if int(obs.sum()) >= min_px:
                break
        else:
            raise RuntimeError(f"block {i}: fewer than {min_px} ICESat px after {max_tries} draws; lower min_px")

        noise = (torch.randn(ch, cw, generator=gen) * noise_m).to(tgt.device)
        e2 = dict(e)                                           # shallow copy
        e2["obs_mask"] = obs[None, None]
        e2["obs_dh"] = (tgt + noise)[None, None]
        e2["tracks_redrawn"] = True
        inside = int((obs & e["score_mask"][0, 0].bool()).sum()) if "score_mask" in e else -1
        print(f"{i:5d} {n_before:17,d} {int(obs.sum()):7,d} {inside:22,d} {t:6d}")
        out.append(e2)
    return out
