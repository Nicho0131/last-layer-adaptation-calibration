"""
Training loop, validation, mixed precision and checkpointing.

    setup_amp       bf16 on compute capability >= 8.0, else fp16 + GradScaler (cell 24)
    evaluate        mean metrics + NLL over evaluation episodes, off-track valid pixels (cell 30)
    atomic_save, rng_state, set_rng_state   checkpoint helpers (cell 30)
    train_meta      the meta-training loop with gradient accumulation and resume (cell 33)
    main            pre-training from config.yaml

Checkpoints hold: model, opt, sched, scaler, rng, step, best, history, cfg (model settings), stats.
Read them with models.initialize_calibrator.load_calibrator / load_checkpoint.

"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from models.initialize_calibrator import build_model, get_device, load_checkpoint
from src.preprocessing.model_inputs import Site, load_site_a
from src.training.episodes import make_eval_sets, sample_episode, sample_queries
from src.training.losses import episode_loss, gaussian_nll, point_metrics
from src.training.simulate import sim_kwargs


# =============================================================================
# mixed precision
# =============================================================================
@dataclass
class Amp:
    device_type: str            # "cuda" or "cpu"
    dtype: torch.dtype
    enabled: bool

    @property
    def scaler_on(self) -> bool:
        """A GradScaler is needed only for fp16 (bf16 has fp32's range)."""
        return self.enabled and self.dtype == torch.float16

    def ctx(self):
        """One place that switches autocast on or off for training, evaluation and deployment."""
        return torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled)


def setup_amp(device, use_amp: bool = True, verbose: bool = True) -> Amp:
    device = torch.device(device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cap = torch.cuda.get_device_capability(device)
        torch.backends.cudnn.benchmark = True          # fixed window size -> tune once
        if verbose:
            print(f"GPU: {props.name} | compute capability {cap} | {props.total_memory / 1e9:.1f} GB | "
                  f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    else:
        cap = (0, 0)
        if verbose:
            print("No GPU: running on the CPU (slow). Colab: Runtime > Change runtime type > GPU.")
    amp = Amp(device_type=device.type,
              dtype=torch.bfloat16 if cap >= (8, 0) else torch.float16,
              enabled=use_amp and device.type == "cuda")
    if verbose:
        print(f"AMP: {amp.enabled} ({str(amp.dtype).replace('torch.', '')}) | GradScaler: {amp.scaler_on}")
    return amp


# =============================================================================
# evaluation on episodes (validation during training, and the test strip)
# =============================================================================
@torch.no_grad()
def evaluate(model, episodes: list[dict], amp: Amp | None = None) -> dict[str, float]:
    """Dense prediction per episode, scored at valid off-track pixels; metrics averaged over episodes.
    Returns rmse, rmse_dsm, mae, bias, cover95, sigma, nll."""
    model.eval()
    ctx = amp.ctx if amp is not None else contextlib.nullcontext
    tot, n = {}, 0
    for ep in episodes:
        with ctx():
            mu, lv = model(ep["x"], ep["obs_dh"], ep["obs_mask"], origin_yx=ep["origin"], dsm_hw=ep["dsm_hw"])
        dev = ep["target"].device
        mu, lv = mu[0, 0].float().to(dev), lv[0, 0].float().to(dev)       # [1, 1, H, W] -> [H, W]
        sel = ep["valid"][0, 0].bool() & ~ep["obs_mask"][0, 0].bool()      # off-track valid pixels
        yv = ep["target"][0, 0][sel].float()
        m = point_metrics(mu[sel], lv[sel], yv)
        m["nll"] = gaussian_nll(mu[sel], lv[sel], yv).item()
        for k, v in m.items():
            tot[k] = tot.get(k, 0.0) + v
        n += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {k: v / n for k, v in tot.items()}


# =============================================================================
# checkpoint helpers
# =============================================================================
def atomic_save(obj, path: str) -> None:
    """Write to a temp file, then rename, so a disconnect cannot leave a half-written file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(gen: torch.Generator) -> dict:
    st = dict(gen=gen.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(gen: torch.Generator, st: dict) -> None:
    # RNG states must be CPU byte tensors; map_location may have moved them to the GPU
    gen.set_state(st["gen"].cpu())
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


# =============================================================================
# the training loop
# =============================================================================
def train_meta(model, site: Site, val_set: list[dict], model_cfg: dict, stats: dict, ckpt_dir: str,
               amp: Amp, steps: int = 3000, lr: float = 3e-4, lr_min_factor: float = 0.05,
               warmup_steps: int = 150, beta: float = 0.0, accum: int = 4, grad_clip: float = 1.0,
               crop_px: int = 2048, min_valid_frac: float = 0.5, noise_m: float = 1.0, n_queries: int = 4096,
               sim: dict | None = None, print_every: int = 5, val_every: int = 100,
               ckpt_every_min: float = 10, resume: bool = True, seed: int = 0) -> dict:
    """Meta-training on random episodes from site.boxes["train"].

    Each optimiser step averages `accum` episodes. Warmup steps use MSE on mu (episode_loss).
    Validation NLL every val_every steps -> best.pt; last.pt every val_every steps, every
    ckpt_every_min minutes and on Ctrl-C / interrupt. resume=True continues from last.pt,
    including the episode RNG, so the sequence of episodes is the same as without the break.
    Returns history (train_step, train_loss, val_step, val_loss).
    """
    device = next(model.parameters()).device
    train_box = site.boxes["train"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * lr_min_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp.scaler_on)
    last_path, best_path = f"{ckpt_dir}/last.pt", f"{ckpt_dir}/best.pt"
    gen = torch.Generator().manual_seed(seed)
    start, best = 0, float("inf")
    history = dict(train_step=[], train_loss=[], val_step=[], val_loss=[])

    if resume and os.path.exists(last_path):
        ck = load_checkpoint(last_path, device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        set_rng_state(gen, ck["rng"])
        start, best = ck["step"] + 1, ck["best"]
        history = ck.get("history", history)
        print(f"resumed at step {start} (best validation loss {best:.4f})")

    def save(path, step):
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), rng=rng_state(gen), step=step, best=best,
                         history=history, cfg=model_cfg, stats=stats), path)

    t_ckpt = time.time()
    running, n_run, step = 0.0, 0, start - 1
    try:
        for step in range(start, steps):
            model.train()
            warm = step < warmup_steps
            opt.zero_grad(set_to_none=True)
            step_loss, ok = 0.0, True

            # ---- gradient accumulation over `accum` episodes ----
            for _ in range(accum):
                ep = sample_episode(site, train_box, gen, crop_px, min_valid_frac, noise_m, device, sim)
                q = sample_queries(ep["valid"], ep["obs_mask"], n_queries)
                y = ep["target"][0, 0][q[0, :, 0], q[0, :, 1]].float()
                with amp.ctx():
                    mu, lv = model(ep["x"], ep["obs_dh"], ep["obs_mask"], q,
                                   origin_yx=ep["origin"], dsm_hw=ep["dsm_hw"])
                loss = episode_loss(mu[0].float(), lv[0].float(), y, warmup=warm, beta=beta)
                if not torch.isfinite(loss):
                    ok = False
                    break
                scaler.scale(loss / accum).backward()
                step_loss += loss.item() / accum
                del ep, q, y, mu, lv, loss

            if not ok:
                opt.zero_grad(set_to_none=True)
                print(f"step {step + 1}: non-finite loss, step skipped")
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not amp.scaler_on and not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                print(f"step {step + 1}: non-finite gradient, step skipped")
                continue
            scale_before = scaler.get_scale() if amp.scaler_on else 1.0
            scaler.step(opt)
            scaler.update()
            if not (amp.scaler_on and scaler.get_scale() < scale_before):   # skip if the step was skipped
                sched.step()

            history["train_step"].append(step + 1)
            history["train_loss"].append(step_loss)
            running += step_loss
            n_run += 1
            if (step + 1) % print_every == 0:
                print(f"step {step + 1:6d} | loss {running / n_run:.4f}")
                running, n_run = 0.0, 0

            if (step + 1) % val_every == 0 or step + 1 == steps:
                v = evaluate(model, val_set, amp)
                history["val_step"].append(step + 1)
                history["val_loss"].append(v["nll"])
                print(f"   validation loss {v['nll']:.4f}")
                if v["nll"] < best:
                    best = v["nll"]
                    save(best_path, step)
                    print("   saved best.pt")
                save(last_path, step)
                t_ckpt = time.time()

            if time.time() - t_ckpt > ckpt_every_min * 60:
                save(last_path, step)
                t_ckpt = time.time()

    except KeyboardInterrupt:
        print("interrupted: saving last.pt")
        save(last_path, max(step, start - 1))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return history


# =============================================================================
# pre-training from config.yaml
# =============================================================================
def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description="Pre-train the calibrator on Site A.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", action="store_true", help="continue from <checkpoints_dir>/<run_name>/last.pt")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t, e = cfg["training"], cfg["episodes"]

    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = get_device()
    amp = setup_amp(device, t["use_amp"])

    site = load_site_a(cfg, write_stats=True)                 # writes stats.json
    model_cfg = dict(cfg["model"], pixel_m=site.pixel_m)      # pixel size from the raster, as in the notebook
    model = build_model(model_cfg, device)
    val_set, _ = make_eval_sets(site, cfg, device)

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"], t["run_name"])
    history = train_meta(
        model, site, val_set, model_cfg, site.stats, ckpt_dir, amp,
        steps=t["total_steps"], lr=t["lr"], lr_min_factor=t["lr_min_factor"],
        warmup_steps=t["warmup_steps"], beta=t["beta"], accum=t["accum_steps"], grad_clip=t["grad_clip"],
        crop_px=e["crop_px"], min_valid_frac=e["min_valid_frac"], noise_m=e["icesat_noise_m"],
        n_queries=e["n_queries"], sim=sim_kwargs(cfg.get("icesat_sim")),
        print_every=t["print_every"], val_every=t["val_every"], ckpt_every_min=t["ckpt_every_min"],
        resume=args.resume or t["resume"], seed=t["seed"])
    print(f"done: {len(history['train_step'])} steps logged | checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
