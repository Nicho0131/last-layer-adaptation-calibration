"""
Build, load and prepare ProbabilisticCalibrator models.


Usage:
    import yaml
    from models.initialize_calibrator import build_model, load_calibrator, load_for_finetune

    cfg = yaml.safe_load(open("config.yaml"))
    model = build_model(cfg["model"])                                   # new, untrained
    model, stats, model_cfg = load_calibrator(cfg["sites"]["tile_1"]["checkpoint"])   # inference
    model, stats, model_cfg, groups = load_for_finetune(
        cfg["finetune"]["paths"]["source_checkpoint"],
        lr=cfg["finetune"]["training"]["lr"],
        freeze_encoder=False, encoder_lr=1e-5)

"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .probabilistic_calibrator import ProbabilisticCalibrator


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def get_device(device: str | torch.device | None = None) -> torch.device:
    """The given device, or CUDA when available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# settings older checkpoints stored in "cfg" that are not model arguments (dropped when loading);
# e.g. the no_augmentation checkpoint has solve_on_cpu (notebook cell 128 removed it by hand)
LEGACY_KEYS = {"solve_on_cpu"}


def _check_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of model_cfg; raises a clear error for keys ProbabilisticCalibrator does not accept
    (e.g. a typo in config.yaml), instead of a confusing TypeError from __init__."""
    allowed = set(inspect.signature(ProbabilisticCalibrator.__init__).parameters) - {"self"}
    model_cfg = {k: v for k, v in model_cfg.items() if k not in LEGACY_KEYS}
    unknown = set(model_cfg) - allowed
    if unknown:
        raise KeyError(f"unknown model setting(s) {sorted(unknown)}; "
                       f"ProbabilisticCalibrator accepts {sorted(allowed)}")
    cfg = dict(model_cfg)
    if "level_res_m" in cfg:                       # YAML gives a list; the model stores a tuple
        cfg["level_res_m"] = tuple(float(r) for r in cfg["level_res_m"])
    return cfg


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# -----------------------------------------------------------------------------
# new model
# -----------------------------------------------------------------------------
def build_model(model_cfg: dict[str, Any], device: str | torch.device | None = None,
                pixel_m: float | None = None, verbose: bool = True) -> ProbabilisticCalibrator:
    """Untrained model from the `model:` section of config.yaml.

    pixel_m: pass the pixel size read from the DSM raster to override the config value
             (the notebook used PIXEL_M = abs(transform.a)).
    """
    cfg = _check_model_cfg(model_cfg)
    if pixel_m is not None:
        cfg["pixel_m"] = float(pixel_m)
    model = ProbabilisticCalibrator(**cfg).to(get_device(device))
    if verbose:
        print(f"built ProbabilisticCalibrator | parameters: {count_parameters(model):,}")
    return model


# -----------------------------------------------------------------------------
# from a checkpoint
# -----------------------------------------------------------------------------
def load_checkpoint(path: str, device: str | torch.device | None = None) -> dict[str, Any]:
    """Raw checkpoint dict written by train_meta: model, opt, sched, scaler, rng, step, best,
    history, cfg, stats.
    weights_only=False because the file holds RNG states and lists. Only load your own checkpoints."""
    return torch.load(path, map_location=get_device(device), weights_only=False)


def load_calibrator(ckpt_path: str, device: str | torch.device | None = None,
                    for_training: bool = False, verbose: bool = True
                    ) -> tuple[ProbabilisticCalibrator, dict[str, Any], dict[str, Any]]:
    """Model with the checkpoint's own architecture and weights, plus its training statistics.

    Returns (model, stats, model_cfg).
      stats:     the stats.json contents saved with the model (means / stds, target range, boxes).
                 Standardise new inputs with THESE, never with the new area's statistics.
      model_cfg: the architecture the checkpoint was trained with.
    for_training=False: eval mode and gradient checkpointing off (cell 82).
    for_training=True:  train mode and the stored grad_checkpoint setting.
    """
    dev = get_device(device)
    ck = load_checkpoint(ckpt_path, dev)
    cfg = _check_model_cfg(ck["cfg"])
    if not for_training:
        cfg["grad_checkpoint"] = False             # no gradients at inference
    model = ProbabilisticCalibrator(**cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.train(for_training)
    if verbose:
        best = ck.get("best", float("nan"))
        print(f"loaded {ckpt_path} | step {ck.get('step', -1) + 1} | best val loss {best:.4f} | "
              f"{'train' if for_training else 'eval'} mode on {dev}")
    return model, ck["stats"], cfg


# -----------------------------------------------------------------------------
# fine-tuning
# -----------------------------------------------------------------------------
def finetune_param_groups(model: ProbabilisticCalibrator, lr: float, freeze_encoder: bool = True,
                          encoder_lr: float | None = None) -> list[dict[str, Any]]:
    """Optimizer parameter groups for fine-tuning; also sets requires_grad.

    freeze_encoder=True  (cell 159): encoder frozen, everything else trained at `lr`.
    freeze_encoder=False (cell 162): encoder trained at `encoder_lr` (e.g. 1e-5),
                                     everything else at `lr` (e.g. 1e-4).
    Pass the result to torch.optim.AdamW(groups).
    """
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    for p in rest:
        p.requires_grad_(True)

    if freeze_encoder:
        for p in enc:
            p.requires_grad_(False)
        return [{"params": rest, "lr": lr}]

    for p in enc:
        p.requires_grad_(True)
    return [{"params": enc, "lr": encoder_lr if encoder_lr is not None else lr},
            {"params": rest, "lr": lr}]


def load_for_finetune(ckpt_path: str, lr: float, freeze_encoder: bool = True,
                      encoder_lr: float | None = None, device: str | torch.device | None = None,
                      verbose: bool = True
                      ) -> tuple[ProbabilisticCalibrator, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Pre-trained model in train mode, ready for fine-tuning.

    Returns (model, pretrain_stats, model_cfg, param_groups).
    Keep using pretrain_stats' means / stds to standardise the new area's inputs.
    """
    model, stats, cfg = load_calibrator(ckpt_path, device, for_training=True, verbose=verbose)
    groups = finetune_param_groups(model, lr, freeze_encoder, encoder_lr)
    if verbose:
        print(f"fine-tuning: encoder {'frozen' if freeze_encoder else 'trainable'} | "
              f"trainable parameters {count_parameters(model, trainable_only=True):,} "
              f"of {count_parameters(model):,}")
    return model, stats, cfg, groups


# -----------------------------------------------------------------------------
# sanity check: python -m models.initialize_calibrator config.yaml
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    m = build_model(config["model"])
    groups = finetune_param_groups(m, lr=1e-4, freeze_encoder=False, encoder_lr=1e-5)
    print(f"param groups: {[len(g['params']) for g in groups]} tensors | lrs {[g['lr'] for g in groups]}")