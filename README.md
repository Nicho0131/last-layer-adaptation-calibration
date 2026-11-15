# DSM-to-DTM calibration with test-time last-layer adaptation

A probabilistic neural network that turns a global digital surface model (Copernicus GLO-30 DSM, 30 m) into a
digital terrain model (bare ground), together with a per-pixel calibration uncertainty. At prediction time it adapts to
a new area using the sparse ICESat-2 ATL08 ground heights available there, with a closed-form update of its last
layer and no retraining.

For every 30 m pixel the model predicts the correction Δh = DTM − DSM as a Gaussian, mean μ and standard deviation σ:

```
DTM = DSM + μ          uncertainty of the ground height = σ
```

## Overview

A DSM records the top of whatever covers the ground: trees, buildings, crops. Removing that bias needs (1) knowing
what the land cover looks like, which a network can learn from global raster layers, and (2) local ground truth,
which ICESat-2 provides only along narrow tracks kilometres apart. This model combines both:

1. **Encode.** A convolutional encoder reads 9 input channels per pixel: DSM, slope, aspect (sine and cosine),
   terrain ruggedness, canopy height (ETH 2020), LAI (MODIS), EVI (Sentinel-2) and building height (GHSL).
   Large areas are processed in overlapping 64 × 64 pixel windows that are blended with Hann weights, so the result
   does not depend on where the windows fall. A shared trunk turns the features into a basis *h* for the final layer.
2. **Write (test-time adaptation).** Over the whole area, a multiresolution grid of nodes is laid out at 1, 4 and
   16 km spacing. At each node, the pixels with an ICESat-2 observation give a weighted ridge regression of the
   residual (observed Δh minus the shared prediction) on *h*. Its closed-form solution is a node-specific correction
   ΔW of the final layer. The ridge strength λ is learned per grid level.
3. **Read.** Every pixel gathers the corrections ΔW of the nearby nodes on all levels with a learned attention
   (GATv2-style, from the pixel's features and its position within the grid cell). The mean is
   μ = (W + ΔW(x)) · [h, 1]. A variance head predicts σ from *h* and from how much ICESat-2 data supports the pixel
   (its support, its distance to data, whether any node has data). σ is bounded between 0.1 and 30 m.
4. **Train.** The network is meta-trained on airborne LiDAR in the UK (Digimap, converted from ODN to EGM2008
   heights). Each episode is a random 2048 × 2048 pixel crop with **simulated ICESat-2 tracks**: reference ground
   tracks at random spacing and heading, three beam pairs, dropped beams, cloud gaps, and Gaussian noise on the
   observed Δh. The model is scored on held-out off-track pixels, so it learns to use sparse tracks the way it will
   meet them on real data. The loss is an MSE warm-up on μ, then a β-weighted Gaussian negative log-likelihood.

Because the adaptation is a closed-form solve inside the forward pass, a new area only needs its GEE feature stack
and the ATL08 files that cover it.

## Main features

- **Probabilistic output:** a mean correction and a calibrated σ for every pixel, evaluated with coverage,
  reliability diagrams, spread–skill and CRPS.
- **Test-time adaptation without retraining:** a closed-form ridge fit of the final layer on a 1 / 4 / 16 km node
  grid, with learned per-level regularisation.
- **Attention read-out across scales:** each pixel draws on the node corrections of all three levels, weighted by a
  learned attention, with learned support features for the uncertainty.
- **Window-independent dense prediction:** per-pixel channel normalisation and Hann-blended overlapping windows make
  large rasters seamless.
- **Realistic ICESat-2 simulation for training:** beam-pair geometry, track spacing, beam dropouts, cloud gaps and
  observation noise.
- **Complete data pipeline:** downloads from Google Earth Engine and NASA earthaccess, LiDAR mosaicking and
  aggregation to 30 m, the ODN → EGM2008 datum shift via OSGM15, the ATL08 quality filter, and reference DEMs
  (FABDEM, FathomDEM) put on the same grid.
- **Evaluation and baselines:** held-out strips and blocks with simulated tracks (Site A); a separate site with real
  ATL08 and LiDAR (Site B); comparison against the uncorrected DSM, FABDEM, a global ATL08 offset, regional ridge
  regression, regression kriging and the network without ATL08. Includes block-bootstrap confidence intervals,
  breakdowns by distance to ATL08, canopy, slope and object edges, and a co-registration check.
- **Ablations and sensitivity studies:** single grid levels, fixed (bilinear or Gaussian) read-out instead of
  attention, fixed or scaled λ, no support features, one year of ATL08, and the effect of real versus simulated
  ATL08 noise.
- **Reproducible training on Colab:** mixed precision (bf16 or fp16), gradient checkpointing, gradient accumulation,
  atomic checkpoints with the full RNG state, so a run resumes exactly after a disconnect.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.yaml                        # all paths and settings (data, model, training, evaluation)
├── last_layer_adaptation_calibration.ipynb   # the original research notebook
│
├── dataset/                           # data preparation, run in this order
│   ├── __init__.py
│   ├── data_download.py               # Copernicus DSM + GEE feature stack, ICESat-2 ATL08
│   ├── prepare_rasters.py             # merge LiDAR tiles, align to the 30 m grid, ODN -> EGM2008
│   ├── build_inputs.py                # model inputs, target, train/val/test split, stats.json
│   └── reference_dems.py              # FABDEM / FathomDEM on a site's grid
│
├── models/
│   ├── __init__.py
│   ├── probabilistic_calibrator.py    # the network: encoder, trunk, node grid, closed-form fit, read-out
│   └── initialize_calibrator.py       # build a model from the config, load one from a checkpoint
│
└── src/
    ├── preprocessing/
    │   ├── icesat.py                  # read, filter and rasterise real ATL08
    │   └── model_inputs.py            # one area as a Site, ready for the model
    ├── training/
    │   ├── simulate.py                # simulated ICESat-2 tracks
    │   ├── episodes.py                # training crops and evaluation blocks
    │   ├── losses.py                  # Gaussian NLL, β-NLL, warm-up loss
    │   └── trainer.py                 # training loop, validation, mixed precision, checkpoints
    └── evaluation/
        ├── inference.py               # predictions, and calibrate() for a new area
        ├── metrics.py                 # RMSE, NMAD, coverage, NLL, CRPS
        ├── evaluate.py                # tables: per site, per class, calibration, ATL08 quality
        ├── baselines.py               # global offset, ridge, regression kriging, co-registration
        ├── ablations.py               # ablation switches, λ sweeps, ATL08 noise study
        └── plots.py                   # all figures
```

The rasters, ATL08 files and checkpoints are not part of the repository. `dataset/` recreates them, and their
locations are set in `config.yaml`.

