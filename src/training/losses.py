"""
Losses and point metrics for the probabilistic calibrator (mean mu, log-variance lv).

    gaussian_nll    0.5 * (lv + (y - mu)^2 / exp(lv)), optionally beta-NLL weighted
    episode_loss    warmup: MSE on mu + NLL on lv only (mu detached); afterwards: (beta-)NLL
    point_metrics   RMSE, uncorrected-DSM RMSE, MAE, bias, 95 % coverage, mean sigma
    
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood (without the constant 0.5 log 2pi).
    beta > 0: beta-NLL (Seitzer et al. 2022), each term weighted by sigma^(2 beta), weight detached."""
    nll = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))
    if beta > 0:
        nll = nll * torch.exp(lv).detach() ** beta
    return nll.mean()


def episode_loss(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor, warmup: bool = False,
                 beta: float = 0.0) -> torch.Tensor:
    """Training loss of one episode. During warmup the mean is fitted with MSE and the variance head
    with NLL on a detached mean, so a poor early mu cannot inflate sigma."""
    if warmup:
        return F.mse_loss(mu, y) + gaussian_nll(mu.detach(), lv, y)
    return gaussian_nll(mu, lv, y, beta=beta)


@torch.no_grad()
def point_metrics(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Metrics at scored pixels. rmse_dsm = error with no correction (mu = 0).
    cover95 is ~0.95 if sigma is calibrated."""
    err, sig = mu - y, torch.exp(0.5 * lv)
    return dict(
        rmse=err.pow(2).mean().sqrt().item(),
        rmse_dsm=y.pow(2).mean().sqrt().item(),
        mae=err.abs().mean().item(),
        bias=err.mean().item(),
        cover95=(err.abs() < 1.96 * sig).float().mean().item(),
        sigma=sig.mean().item())
