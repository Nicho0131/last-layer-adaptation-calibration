# ===================================
# Last-layer adapted calibration model for ICESat-2 DTM correction
# 
# Key components:
# - DSMEncoder: CNN encoder for the input DSM stack
# - Trunk: CNN trunk for the latent feature map
# - Grid_Diffuser: Graph Attention Network (GATv2) style diffuser for coarse grid nodes
# - ProbabilisticCalibrator: Main model class that combines the above components
# 
# ====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


# ---- helper classes and features -------
class ChannelNorm(nn.Module):
  """ layernorm over channels at each pixel. Unlike GroupNorm does not use statistics over the window, a pixel's features does not change with the window"""
  def __init__(self, c):
    super().__init__()
    self.ln = nn.LayerNorm(c)

  def forward(self, x):
    return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, C, H W]

def hann_2d(ph, pw, device, floor=0.1):
  """Seperable hann window with a floor to ensure edge pixels never get 0 weight"""
  wy = torch.hann_window(ph, periodic=False, device=device)
  wx = torch.hann_window(pw, periodic=False, device=device)
  wy, wx = floor + (1 - floor) * wy, floor + (1 - floor) * wx
  return wy[:, None] * wx[None, :]

def window_starts(n, p, stride):
  """window start position that covers (0, n) completely, last window flush with edge"""
  if n <= p:
    return [0]
  s = list(range(0, n - p + 1, stride))
  if s[-1] != n - p:
    s.append(n - p)
  return s


# ==============
# FEATURE ENCODING
# - CNN-conv encoding into a latent space for feature map
# ==============
class DSMEncoder(nn.Module):
  "Processes Geotiff into a dense topological feature map F"
  def __init__(self, in_channels=9, feature_dim=64):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        ChannelNorm(32),
        nn.GELU(),
        # Dilated convolution to expand physical receptive field
        nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
        ChannelNorm(64),
        nn.GELU(),
        nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
    )

  def forward(self, x):
    return self.net(x)

class Trunk(nn.Module):
  """
  Seperated: The two convolutional layers prior to the final layer. Previous version TTT is not implemented well
  """
  def __init__(self, feature_dim=64, hidden_dim=64):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, padding=1),
        ChannelNorm(feature_dim * 2),
        nn.GELU(),
        nn.Conv2d(feature_dim*2, hidden_dim, 3, padding=1),
        ChannelNorm(hidden_dim),
    )

  def forward(self, f):
    return self.net(f)

# ================
# DIFFUSER MECHANISM USING GAT AND MULTILAYER COARSE GRID
# ================
class Grid_Diffuser(nn.Module):
  """
  Diffuser class motivated by Graph Attention Networks (GAT)
  - Coarse Memory Grid motivated by Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (NVIDIA 2022)
  Changed implementation 10/10/26 - additive GATv2-style attention
  """
  def __init__(self, feature_dim=64, n_levels=3, hidden_dim=32):
    super().__init__()
    self.q = nn.Linear(feature_dim, hidden_dim)
    self.k = nn.Linear(feature_dim, hidden_dim)
    self.level_emb = nn.Parameter(torch.zeros(n_levels, hidden_dim))
    self.count_proj = nn.Linear(1, hidden_dim) # attention trusts well-sampled nodes
    self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
    self.sim_head = nn.Linear(hidden_dim, 1)
    self.gamma_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

  def forward(self, q_feat, pix_xy, levels):
    """
    q_feat: [B, F], pix_xy: [B, 2] = (x,y) pixel indices
    levels: list of dicts with k [G, hidden], w [G, Dh], count [G]
    Returns dW [N, Dh] and aux [B, 3] = (support, log1p(dist_to_data), has_data)
    """
    q = self.q(q_feat)
    scores, ws, logcs, dist, mask = [], [], [], [], []
    for li, L in enumerate(levels):
      # node i = sum pixel [i*r, (i+1)*r], sits at (i + 0.5)*r
      gx = pix_xy[:, 0] / L['rx'] - 0.5
      gy = pix_xy[:, 1] / L['ry'] - 0.5
      x0, y0 = gx.floor().long(), gy.floor().long()
      xs = torch.stack([x0, x0+1, x0, x0+1], -1) # TL, TR, BL, BR
      ys = torch.stack([y0, y0, y0+1, y0+1], -1)
      inb = (xs >= 0) & (xs < L['Wg']) & (ys >= 0) & (ys < L['Hg'])
      idx = ys.clamp(0, L['Hg'] - 1) * L['Wg'] + xs.clamp(0, L['Wg'] - 1)

      count = L['count'][idx]
      valid = inb & (count > 0) # empty nodes dont vote for diffusion
      d2 = (gx[:, None] - xs.float()) ** 2 + (gy[:, None] - ys.float()) ** 2
      lc = torch.log1p(count)

      e = self.mlp(q[:, None] + L['k'][idx] + self.level_emb[li] + self.count_proj(lc.unsqueeze(-1)))
      s = self.sim_head(e).squeeze(-1) - self.gamma_head(e).squeeze(-1) * d2

      scores.append(s)
      ws.append(L['w'][idx])
      logcs.append(lc)
      dist.append(d2.sqrt() * L['res_km'])
      mask.append(valid)

    S = torch.cat(scores, -1).float()
    M = torch.cat(mask, -1)
    alpha = torch.softmax(S.masked_fill(~M, -1e4), -1) * M # all zero if no valid node
    dW = (alpha.unsqueeze(-1) * torch.cat(ws, 1)).sum(1)

    support = (alpha * torch.cat(logcs, -1)).sum(-1)
    dmin = torch.cat(dist, -1).masked_fill(~M, 1e3).amin(-1).clamp(max=100.0)
    aux = torch.stack([support, torch.log1p(dmin), M.any(-1).float()], -1)

    return dW, aux

# =====
# TTT-layer: previously undefined - which is an oversight
# =====
def ttt_fit_final_layer(h_obs, r_obs, node_flat, n_nodes, lam, weight=None, chunk=8192):
  """
  Per-node n: dW_n = argmin sum_{i in n} w_i (r_i - dW . h_i)^2 + lam * |dW|^2
  Closed form via accumulated normal equations (no padding).
  Memory ~ U*Dh*Dh + chunk*Dh*Dh floats, independent of points per node.
  """
  N, Dh = h_obs.shape
  dev, dt = h_obs.device, torch.float32

  with torch.autocast(device_type=dev.type, enabled=False):
      if weight is None:
          weight = torch.ones(N, device=dev, dtype=dt)
      count = torch.zeros(n_nodes, device=dev, dtype=dt).index_add(0, node_flat, weight.detach().to(dt))
      w_grid = torch.zeros(n_nodes, Dh, device=dev, dtype=dt)
      if N == 0:
          return w_grid, count

      h = h_obs.to(dt)
      r = r_obs.to(dt)
      w = weight.to(dt)

      # compact to nodes that actually have data
      uniq, inv = torch.unique(node_flat, return_inverse=True)
      U = uniq.numel()

      A = torch.zeros(U, Dh, Dh, device=dev, dtype=dt)
      b = torch.zeros(U, Dh, device=dev, dtype=dt)
      for s in range(0, N, chunk):
          hc, rc, wc, ic = h[s:s + chunk], r[s:s + chunk], w[s:s + chunk], inv[s:s + chunk]
          hw = hc * wc[:, None]                                    # w_i * h_i
          A = A.index_add(0, ic, hw[:, :, None] * hc[:, None, :])  # sum w h h^T
          b = b.index_add(0, ic, hw * rc[:, None])                 # sum w h r

      A = A + lam.to(dt) * torch.eye(Dh, device=dev, dtype=dt)

      # CPU solve as before (batched GPU solver issue on T4 + CUDA 13)
      dW = torch.linalg.solve(A.cpu(), b.cpu()).to(dev)

      return w_grid.index_copy(0, uniq, dW), count

# =====================
# FINAL ARCHITECTURE
# - hann window sliding structure
# - fixed node_res in meters (the spatial length of one grid node)
# v2 - updated for using sigma to bound loss_variance window for more efficient and less hardcoded predictions
# =====================

class ProbabilisticCalibrator(nn.Module):
  def __init__(self, in_channels=9, feature_dim=64, hidden_dim=64, pixel_m=30.0,
                level_res_m=(1000.0, 4000.0, 16000.0), diffuser_hid=32,
                patch_size=64, overlap=32, win_batch=64, chunk=64*64,
                sigma_min=0.1, sigma_max=30.0, grad_checkpoint=True):
      super().__init__()
      self.feature_dim = feature_dim
      self.hidden_dim = hidden_dim
      self.pixel_m = pixel_m
      self.level_res_m = tuple(level_res_m)
      self.patch_size = patch_size
      self.stride = patch_size - overlap
      self.win_batch = win_batch  # windows per encoder call
      self.chunk = chunk  # pixels per diffuser call
      self.grad_checkpoint = grad_checkpoint
      # predicted sigma is bounded to [sigma_min, sigma_max] metres
      self.lv_min = 2 * math.log(sigma_min)
      self.lv_max = 2 * math.log(sigma_max)

      self.encoder = DSMEncoder(in_channels, feature_dim)
      self.decoder_body = Trunk(feature_dim, hidden_dim)
      self.grid_gat = Grid_Diffuser(feature_dim, len(level_res_m), diffuser_hid)

      # Final layer on [h, 1]. Shared default; TTT adds dW(x) from the grid.
      self.final_mu = nn.Linear(hidden_dim + 1, 1, bias=False)
      nn.init.zeros_(self.final_mu.weight)
      # Variance head: basis + how much ICESat supports this pixel
      self.final_logvar = nn.Sequential(nn.Linear(hidden_dim + 3, 32), nn.GELU(), nn.Linear(32, 1))
      # Learned weight decay of the TTT fit, one per grid level
      self.log_lam = nn.Parameter(torch.zeros(len(level_res_m)))

  def _device(self):
      return next(self.parameters()).device

  # ---------- geometry ----------
  def _grid_levels(self, dsm_h, dsm_w):
      """Grid anchored to the full DSM: node count and spacing fixed by the DSM extent."""
      out = []
      for res_m in self.level_res_m:
          res_px = res_m / self.pixel_m
          Hg, Wg = max(1, math.ceil(dsm_h / res_px)), max(1, math.ceil(dsm_w / res_px))
          rx, ry = dsm_w / Wg, dsm_h / Hg
          out.append(dict(Hg=Hg, Wg=Wg, rx=rx, ry=ry, G=Hg * Wg,
                          res_km=0.5 * (rx + ry) * self.pixel_m / 1000.0))
      return out

  @staticmethod
  def _node_index(g, gy, gx):
      """Global pixel (y, x) -> flat node index on level g."""
      ny = (gy.float() / g['ry']).floor().long().clamp(0, g['Hg'] - 1)
      nx = (gx.float() / g['rx']).floor().long().clamp(0, g['Wg'] - 1)
      return ny * g['Wg'] + nx

  def _windows(self, H, W, data_dev, dev):
      """Window corners, Hann weight and the summed weight
      per pixel `total`"""
      ph, pw = min(self.patch_size, H), min(self.patch_size, W)
      wins = [(y, x) for y in window_starts(H, ph, self.stride)
                      for x in window_starts(W, pw, self.stride)]
      w2d = hann_2d(ph, pw, dev)
      w2d_d = w2d.to(data_dev)
      total = torch.zeros(H, W, device=data_dev)
      for y, x in wins:
          total[y:y + ph, x:x + pw] += w2d_d
      return wins, ph, pw, w2d, total

  @staticmethod
  def _stack(t, wl, ph, pw, dev):
      """Cut windows wl = [(y0, x0), ...] out of t [..., H, W] -> [g, ..., ph, pw]"""
      return torch.stack([t[..., y:y + ph, x:x + pw] for y, x in wl]).to(dev)

  @staticmethod
  def _coords(wl, ph, pw, oy, ox, dev):
      """Global DSM coordinates of every pixel of windows wl -> gy, gx [g, ph, pw]."""
      y0 = torch.tensor([y for y, _ in wl], device=dev)[:, None, None]
      x0 = torch.tensor([x for _, x in wl], device=dev)[:, None, None]
      gy = (y0 + torch.arange(ph, device=dev)[None, :, None] + oy).expand(-1, ph, pw)
      gx = (x0 + torch.arange(pw, device=dev)[None, None, :] + ox).expand(-1, ph, pw)
      return gy, gx

  def _encode_raw(self, xg):
      f = self.encoder(xg)
      h = self.decoder_body(f)
      h1 = torch.cat([h, torch.ones_like(h[:, :1])], 1)    # append 1 -> bias term
      return f.float(), h1.float()

  def _encode(self, xg):
      """xg [g, C, ph, pw] -> f [g, D, ph, pw], h1 [g, Dh, ph, pw]."""
      if self.grad_checkpoint and self.training and torch.is_grad_enabled():
          return checkpoint(self._encode_raw, xg, use_reentrant=False)
      return self._encode_raw(xg)

  # ---------- pass 1: WRITE ----------
  def _write_pass(self, xb, dh, mask, wins, ph, pw, w2d, total, geoms, oy, ox):
      """xb [C, H, W], dh [H, W], mask [H, W] bool. Returns the grid levels."""
      dev = self._device()
      D, Dh = self.feature_dim, self.hidden_dim + 1
      fsum = [torch.zeros(g['G'], D, device=dev) for g in geoms]
      fwt = [torch.zeros(g['G'], device=dev) for g in geoms]
      H_l, T_l, O_l = [], [], []                 # basis, target, omega at ICESat pixels
      N_l = [[] for _ in geoms]                  # node index per level at ICESat pixels

      for s in range(0, len(wins), self.win_batch):
          wl = wins[s:s + self.win_batch]
          f, h1 = self._encode(self._stack(xb, wl, ph, pw, dev))
          omega = (w2d[None] / self._stack(total, wl, ph, pw, dev)).reshape(-1)  # partition of unity
          gy, gx = self._coords(wl, ph, pw, oy, ox, dev)
          nodes = [self._node_index(g, gy.reshape(-1), gx.reshape(-1)) for g in geoms]

          # node features for the GAT: omega-weighted average of pixel features per node
          f_flat = f.permute(0, 2, 3, 1).reshape(-1, D)
          for li in range(len(geoms)):
              fsum[li] = fsum[li].index_add(0, nodes[li], omega[:, None] * f_flat)
              fwt[li] = fwt[li].index_add(0, nodes[li], omega)

          # ICESat pixels inside these windows (if any)
          m = self._stack(mask, wl, ph, pw, dev).reshape(-1)
          if m.any():
              idx = m.nonzero(as_tuple=True)[0]
              H_l.append(h1.permute(0, 2, 3, 1).reshape(-1, Dh)[idx])
              T_l.append(self._stack(dh, wl, ph, pw, dev).reshape(-1)[idx].float())
              O_l.append(omega[idx])
              for li in range(len(geoms)):
                  N_l[li].append(nodes[li][idx])

      if H_l:
          h_obs, dh_obs, w_obs = torch.cat(H_l), torch.cat(T_l), torch.cat(O_l)
          r_obs = dh_obs - (h_obs * self.final_mu.weight.float()).sum(-1)   # residual vs shared layer
      else:
          h_obs = torch.zeros(0, Dh, device=dev)
          r_obs, w_obs = torch.zeros(0, device=dev), torch.zeros(0, device=dev)

      levels = []
      for li, g in enumerate(geoms):
          node = torch.cat(N_l[li]) if H_l else torch.zeros(0, dtype=torch.long, device=dev)
          w, cnt = ttt_fit_final_layer(h_obs, r_obs, node, g['G'], self.log_lam[li].exp(), w_obs)
          k = self.grid_gat.k(fsum[li] / fwt[li].clamp_min(1e-6)[:, None])
          levels.append(dict(g, k=k, w=w, count=cnt))
      return levels

  # ---------- pass 2: READ ----------
  def _predict_points(self, f_pts, h1_pts, pix_xy, levels):
      dW, aux = self.grid_gat(f_pts, pix_xy, levels)
      mu = ((self.final_mu.weight + dW) * h1_pts).sum(-1)
      raw = self.final_logvar(torch.cat([h1_pts[:, :-1], aux], -1)).squeeze(-1)
      logvar = self.lv_min + (self.lv_max - self.lv_min) * torch.sigmoid(raw.float())
      return mu.float(), logvar

  def _read_dense(self, xb, wins, ph, pw, w2d, total, levels, oy, ox):
      """Every pixel of the region. Returns mu, logvar [H, W]."""
      dev, ddev = self._device(), total.device
      D, Dh = self.feature_dim, self.hidden_dim + 1
      H, W = total.shape
      mu_c = torch.zeros(H, W, device=ddev)
      lv_c = torch.zeros(H, W, device=ddev)
      for s in range(0, len(wins), self.win_batch):           # left -> right, top -> bottom
          wl = wins[s:s + self.win_batch]
          g = len(wl)
          f, h1 = self._encode(self._stack(xb, wl, ph, pw, dev))
          gy, gx = self._coords(wl, ph, pw, oy, ox, dev)
          pix = torch.stack([gx, gy], -1).reshape(-1, 2).float()
          f_flat = f.permute(0, 2, 3, 1).reshape(-1, D)
          h_flat = h1.permute(0, 2, 3, 1).reshape(-1, Dh)
          mus, lvs = [], []
          for c in range(0, f_flat.shape[0], self.chunk):
              mu, lv = self._predict_points(f_flat[c:c + self.chunk], h_flat[c:c + self.chunk],
                                            pix[c:c + self.chunk], levels)
              mus.append(mu); lvs.append(lv)
          mu_w = (torch.cat(mus).view(g, ph, pw) * w2d).to(ddev)     # Hann-weighted
          lv_w = (torch.cat(lvs).view(g, ph, pw) * w2d).to(ddev)
          for j, (y, x) in enumerate(wl):
              mu_c[y:y + ph, x:x + pw] += mu_w[j]
              lv_c[y:y + ph, x:x + pw] += lv_w[j]
      return mu_c / total, lv_c / total

  def _read_queries(self, xb, q, wins, ph, pw, w2d, levels, oy, ox):
      """Only the query pixels q [Q, 2] (y, x in region coords), Hann-blended across the
      windows that contain them. Returns mu, logvar [Q] on the model device."""
      dev = self._device()
      q = q.to(dev).long()
      qy, qx = q[:, 0], q[:, 1]
      Q = q.shape[0]
      starts = torch.tensor(wins, device=dev)                  # [nW, 2]
      inw = ((qy[None] >= starts[:, 0, None]) & (qy[None] < starts[:, 0, None] + ph) &
              (qx[None] >= starts[:, 1, None]) & (qx[None] < starts[:, 1, None] + pw))
      wi, qi = inw.nonzero(as_tuple=True)                      # (window, query) pairs
      need = torch.unique(wi)                                  # windows holding a query
      local = torch.full((len(wins),), -1, dtype=torch.long, device=dev)

      acc_mu = torch.zeros(Q, device=dev)
      acc_lv = torch.zeros(Q, device=dev)
      acc_w = torch.zeros(Q, device=dev)
      for s in range(0, need.numel(), self.win_batch):
          sel = need[s:s + self.win_batch]
          wl = [wins[i] for i in sel.tolist()]
          f, h1 = self._encode(self._stack(xb, wl, ph, pw, dev))
          local.fill_(-1)
          local[sel] = torch.arange(sel.numel(), device=dev)
          pm = local[wi] >= 0                                  # pairs in this batch
          wj, pq = local[wi[pm]], qi[pm]
          ly = qy[pq] - starts[wi[pm], 0]
          lx = qx[pq] - starts[wi[pm], 1]
          f_pts = f.permute(0, 2, 3, 1)[wj, ly, lx]
          h_pts = h1.permute(0, 2, 3, 1)[wj, ly, lx]
          pix = torch.stack([qx[pq] + ox, qy[pq] + oy], -1).float()
          mu, lv = self._predict_points(f_pts, h_pts, pix, levels)
          wq = w2d[ly, lx]
          acc_mu = acc_mu.index_add(0, pq, wq * mu)
          acc_lv = acc_lv.index_add(0, pq, wq * lv)
          acc_w = acc_w.index_add(0, pq, wq)
      return acc_mu / acc_w, acc_lv / acc_w

  # ---------- forward ----------
  def forward(self, x, obs_dh, obs_mask, query_yx=None, origin_yx=(0, 0), dsm_hw=None):
      """
      x: [B, C, H, W]  standardised input stack (whole DSM, or a crop of it).
                              May live on the CPU; windows go to the model's device.
      obs_dh: [B, 1, H, W]  delta_h at ICESat pixels (anything elsewhere, may be NaN)
      obs_mask: [B, 1, H, W]  bool, True where an ICESat point exists
      query_yx: [B, Q, 2]     optional (y, x) pixels to predict, in x's coordinates.
                              None -> dense mu / logvar over all of x.
      origin_yx: (y, x) offset of x inside the full DSM ((0, 0) when x is the whole DSM)
      dsm_hw: (H, W) of the full DSM the grid is anchored to (default: x.size)
      Returns mu, logvar as [B, Q] on the model device (query mode)
                      or [B, 1, H, W] on x's device (dense mode).
      DTM = raw_dsm + mu
      """
      B, _, H, W = x.shape
      oy, ox = origin_yx
      dsm_h, dsm_w = dsm_hw if dsm_hw is not None else (H, W)
      obs_mask = obs_mask.bool()
      obs_dh = torch.where(obs_mask, obs_dh, torch.zeros_like(obs_dh))  # NaN-proof

      geoms = self._grid_levels(dsm_h, dsm_w)
      wins, ph, pw, w2d, total = self._windows(H, W, x.device, self._device())

      mus, lvs = [], []
      for b in range(B):
          levels = self._write_pass(x[b], obs_dh[b, 0], obs_mask[b, 0],
                                    wins, ph, pw, w2d, total, geoms, oy, ox)
          if query_yx is not None:
              mu, lv = self._read_queries(x[b], query_yx[b], wins, ph, pw, w2d, levels, oy, ox)
          else:
              mu, lv = self._read_dense(x[b], wins, ph, pw, w2d, total, levels, oy, ox)
              mu, lv = mu[None], lv[None]
          mus.append(mu); lvs.append(lv)
      return torch.stack(mus), torch.stack(lvs)
