# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
FCDA_main.py
============
FCDA-Net : Fractal-Conditioned Dual-Gate Attention Network
============================================================
Brain Tumour Classification — 4 classes
    glioma | meningioma | notumor | pituitary

Dataset  : /nfsshare/users/raghavan/Brainz/Brain tumor dataset/
Save dir : FCDA_2026/

-------------------------------------------------------------------------------
CHANGE LOG 
-------------------------------------------------------------------------------
1. BACKBONE SCOPE NARROWED.
   The study now runs on three backbones only:
       DenseNet121, MobileNetV3-Large, ConvNeXt-Tiny
   ResNet50 has been removed from BACKBONE_REGISTRY. The leave-one-out (LOO)
   ablation runs on ConvNeXt-Tiny only.

2. FIXED THE BLACK FRACTAL-MAP BUG.
   The old `to_map()` normalised every fractal-dimension (FD) map with its
   own per-image min/max: `(m - m.min()) / (m.max() - m.min())`. Whenever a
   map was (near-)constant across the image — common early in training, or
   whenever the DLFD regression saturates — this collapsed the whole map to
   0, which `cmap="magma"` renders as solid black. The fix normalises D
   maps against the THEORETICAL fixed range [0, 3] that `D_map.clamp(0,3)`
   already guarantees, so a flat-but-nonzero map still shows its true tone
   instead of collapsing to black. FSA attention maps (already in [0,1]
   from a sigmoid) use a robust 2nd/98th-percentile contrast stretch
   instead of raw min/max, so a few outlier pixels no longer wash out the
   rest of the map to near-black/near-white.

3. ABLATION STUDY NARROWED.
   The ablation includes only three variants on
   ConvNeXt-Tiny:
       - FCDA w/o FCG  (channel gate removed)
       - FCDA w/o FSA  (spatial attention removed)
       - FCDA_classicalFBC (fixed, non-differentiable box-counting)
   All other ablation variants have been removed.

-------------------------------------------------------------------------------
CORE IDEA 
-------------------------------------------------------------------------------
Radiological literature has long used the fractal dimension (FD) of tumour
margins / internal texture as a hand-crafted, OFFLINE biomarker of lesion
irregularity. This work makes local FD estimation DIFFERENTIABLE and
LEARNABLE, and folds it directly into a CNN feature extractor as a dual gate:

  1. Differentiable Local Fractal Dimension (DLFD) extractor
     - soft edge map from backbone features
     - soft box-occupancy at multiple scales via a learnable sigmoid
       threshold/temperature (replaces the classic hard box-counting test)
     - per-pixel slope of log N(eps) vs log(1/eps) fit in closed form
       (weighted least squares) -> a local FD map, fully backprop-able

  2. Fractal Channel Gate (FCG)   - channel recalibration from FD statistics
  3. Fractal Spatial Attention (FSA) - spatial mask from the local FD map
  4. Residual bypass (gradient highway), matching the TCDA v2 design
  5. Fractal-consistency regulariser - variance-floor term preventing the
     FD map from collapsing to a constant (a dead / bypassed gate)

A CLASSICAL, NON-DIFFERENTIABLE fixed box-counting variant is included as a
control: identical FCG/FSA wiring, but the FD map is computed with fixed
threshold/temperature and detached from the autograd graph (no_grad). This
isolates the benefit of making the fractal representation learnable, which
is the paper's central methodological claim.

-------------------------------------------------------------------------------
MODELS EVALUATED
-------------------------------------------------------------------------------

Main comparison (all 3 backbones):
    BASE_{DenseNet121, MobileNetV3L, ConvNeXtTiny}
    FCDA_{DenseNet121, MobileNetV3L, ConvNeXtTiny}   (full module)

Leave-one-out ablation (on ConvNeXt-Tiny only):
    FCDA_ConvNeXtTiny                  - full model (reference row)
    FCDA_ConvNeXtTiny_noFCG            - channel gate removed
    FCDA_ConvNeXtTiny_noFSA            - spatial attention removed
    FCDA_ConvNeXtTiny_classicalFBC     - fixed, non-differentiable box-counting
    BASE_ConvNeXtTiny                  - backbone only

SEEDS       : 42, 43, 44, 45, 46
STATISTICS  : Wilcoxon signed-rank + pooled McNemar
VISUALS     : GradCAM + DLFD maps + FSA attention maps, saved per seed
"""

import gc
import math
import random
import time
import itertools
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Global figure style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":        "bold",
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "figure.titlesize":   15,
    "figure.titleweight": "bold",
    "xtick.major.width":  1.4,
    "ytick.major.width":  1.4,
    "axes.linewidth":     1.4,
    "lines.linewidth":    2.2,
})

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAIN_PATH    = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH     = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"
SAVE_DIR      = Path("FCDA_2026")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

gpu_id        = 1
DEVICE        = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
IMG_SIZE      = 224
LR            = 3e-4
WARMUP_EPOCHS = 3
EPOCHS        = 25
PATIENCE      = 15
DESC_DIM      = 512
FD_SCALES     = [1, 2, 4, 8]          # box sizes for multi-scale DLFD
LABEL_SMOOTH  = 0.08
CUTMIX_ALPHA  = 1.0
MIXUP_ALPHA   = 0.4
EPS           = 1e-8
ALPHA_STAT    = 0.05
BATCH         = 16
STOCH_DEPTH_P = 0.10
LAMBDA_FRAC   = 0.05                   # weight of fractal-consistency loss
FRAC_MARGIN   = 0.15                   # target minimum std of local FD map

SEEDS = [42, 43, 44, 45, 46]

# --- CHANGE 1: backbone scope narrowed to the three requested backbones ---
BACKBONE_REGISTRY = {
    "DenseNet121":    "densenet121.ra_in1k",
    "MobileNetV3L":   "mobilenetv3_large_100.ra_in1k",
    "ConvNeXtTiny":   "convnext_tiny.in12k_ft_in1k",
}
BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())

# The LOO ablation runs on ConvNeXt-Tiny only.
ABLATION_BACKBONE = "ConvNeXtTiny"

# --------------------------------------------------------------------------
# Model registry — MAIN comparison + LOO ablation, all built from one
# configuration table so create_model() has a single source of truth.
# --------------------------------------------------------------------------
def _cfg(kind, backbone, **kw):
    base = dict(kind=kind, backbone=backbone,
                use_fcg=True, use_fsa=True, multiscale=True,
                learnable=True, bypass=True, fractal_loss=True,
                classical=False)
    base.update(kw)
    return base

MODEL_CONFIGS = {}
for _b in BACKBONE_NAMES:
    MODEL_CONFIGS[f"BASE_{_b}"] = _cfg("BASE", _b)
    MODEL_CONFIGS[f"FCDA_{_b}"] = _cfg("FCDA", _b)

# LOO ablation rows — ABLATION_BACKBONE only, narrowed to 3 variants
_R = ABLATION_BACKBONE
MODEL_CONFIGS[f"FCDA_{_R}_noFCG"]         = _cfg("FCDA", _R, use_fcg=False)
MODEL_CONFIGS[f"FCDA_{_R}_noFSA"]         = _cfg("FCDA", _R, use_fsa=False)
MODEL_CONFIGS[f"FCDA_{_R}_classicalFBC"]  = _cfg("FCDA", _R, learnable=False,
                                                  classical=True, fractal_loss=False)

MAIN_MODELS = ([f"BASE_{b}" for b in BACKBONE_NAMES] +
               [f"FCDA_{b}" for b in BACKBONE_NAMES])
LOO_MODELS  = ([f"FCDA_{_R}"] +
               [f"FCDA_{_R}_noFCG", f"FCDA_{_R}_noFSA",
                f"FCDA_{_R}_classicalFBC",
                f"BASE_{_R}"])
ALL_MODELS = list(dict.fromkeys(MAIN_MODELS + LOO_MODELS))   # de-duplicated, ordered

PROPOSED = f"FCDA_{_R}"   # anchor for the superiority table

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
}

# --------------------------------------------------------------------------
# Directory helpers
# --------------------------------------------------------------------------
def get_dirs(model_name: str) -> dict:
    root = SAVE_DIR / model_name
    for d in [root / "metrics", root / "curves", root / "gradcam", root / "fd_maps"]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "role.txt"
    if not note.exists():
        cfg = MODEL_CONFIGS[model_name]
        note.write_text(f"Model : {model_name}\nConfig: {cfg}\n")
    return {"root": root, "metrics": root/"metrics",
            "curves": root/"curves", "gradcam": root/"gradcam",
            "fd_maps": root/"fd_maps"}


def get_paths(model_name: str, seed: int, dirs: dict) -> dict:
    return {
        "test_metrics": dirs["metrics"] / f"test_metrics_seed{seed}.csv",
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"] / f"seed{seed}",
        "fd_base":      dirs["fd_maps"] / f"seed{seed}",
        "checkpoint":   dirs["root"]    / f"checkpoint_seed{seed}.pt",
    }


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# --------------------------------------------------------------------------
# Augmentation helpers (unchanged from prior pipeline)
# --------------------------------------------------------------------------
def rand_bbox(size, lam):
    W, H  = size[2], size[3]
    cut_r = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_r); cut_h = int(H * cut_r)
    cx = random.randint(0, W); cy = random.randint(0, H)
    x1 = max(cx - cut_w//2, 0); y1 = max(cy - cut_h//2, 0)
    x2 = min(cx + cut_w//2, W); y2 = min(cy + cut_h//2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1.0 - (x2-x1)*(y2-y1) / (x.size(2)*x.size(3) + EPS)
    return mixed, y, y[idx], lam


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def aug_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)


def get_lr(epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


# ==========================================================================
# NOVEL MODULE — Differentiable Local Fractal Dimension (DLFD) extractor
# ==========================================================================
class DLFDExtractor(nn.Module):
    """
    Differentiable, multi-scale, learnable local fractal-dimension estimator.

    Classic box-counting: N(eps) = #boxes of size eps that intersect the
    edge set; D = slope of log N(eps) vs log(1/eps). Both the box-membership
    test and the counting step are non-differentiable in the classic form.

    Here:
      - Edge strength E is a soft gradient-magnitude map of L2-normalised
        backbone features (differentiable by construction).
      - Box "occupancy" is a soft indicator sigmoid(tau * (E - theta))
        instead of a hard threshold; theta, tau are learnable (unless
        `learnable=False`, in which case they are fixed buffers — used for
        the ablation / classical-descriptor control).
      - Soft occupancy is average-pooled per scale, giving a smooth box
        "membership density" map per eps; all scales are resampled to a
        common coarse grid so the log-log regression can be solved
        per-pixel in closed form (weighted least squares slope).
      - `classical=True` additionally detaches the whole computation from
        autograd (torch.no_grad) and disables learnability, faithfully
        reproducing a *fixed, non-differentiable* box-counting descriptor
        wired into the same downstream attention module — the control
        condition that isolates the benefit of making FD learnable.
    """

    def __init__(self, scales=None, learnable=True, classical=False):
        super().__init__()
        self.scales    = list(scales) if scales else list(FD_SCALES)
        self.classical = classical
        init_theta, init_tau = 0.10, 12.0
        if learnable and not classical:
            self.theta = nn.Parameter(torch.tensor(init_theta))
            self.tau   = nn.Parameter(torch.tensor(init_tau))
        else:
            self.register_buffer("theta", torch.tensor(init_theta))
            self.register_buffer("tau",   torch.tensor(init_tau))

    @staticmethod
    def _soft_edge(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        E = (gx**2 + gy**2).mean(dim=1, keepdim=True)          # (B,1,H,W)
        mx = E.amax(dim=(2, 3), keepdim=True).clamp(min=EPS)
        return E / mx

    def _compute(self, feat):
        norm = feat.norm(p=2, dim=(2, 3), keepdim=True).clamp(min=EPS)
        x = feat / norm
        E = self._soft_edge(x)                                  # (B,1,H,W)
        H, W = E.shape[2], E.shape[3]
        grid = max(min(H, W) // max(self.scales), 4)

        log_counts, log_inv_eps = [], []
        for eps in self.scales:
            occ    = torch.sigmoid(self.tau * (E - self.theta))
            pooled = F.avg_pool2d(occ, kernel_size=eps, stride=eps, ceil_mode=True)
            pooled = F.interpolate(pooled, size=(grid, grid),
                                    mode="bilinear", align_corners=False)
            log_counts.append(torch.log(pooled.clamp(min=EPS) * (eps ** 2)))
            log_inv_eps.append(math.log(1.0 / eps))

        y = torch.stack(log_counts, dim=0)                       # (S,B,1,g,g)
        x_pts = torch.tensor(log_inv_eps, device=feat.device).view(-1, 1, 1, 1, 1)

        if len(self.scales) == 1:
            D_map = y[0].clamp(0.0, 3.0)
        else:
            x_mean = x_pts.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            cov = ((x_pts - x_mean) * (y - y_mean)).sum(dim=0)
            var = ((x_pts - x_mean) ** 2).sum(dim=0) + EPS
            D_map = (cov / var).clamp(0.0, 3.0)                   # (B,1,g,g)

        D_full = F.interpolate(D_map, size=(H, W), mode="bilinear",
                                align_corners=False)
        return D_full, D_map

    def forward(self, feat):
        if self.classical:
            with torch.no_grad():
                return self._compute(feat)
        return self._compute(feat)


# ==========================================================================
# NOVEL MODULE — Fractal-Conditioned Dual-Gate Attention (FCDA)
# ==========================================================================
class FCDA(nn.Module):
    """
    Fractal Channel Gate (FCG) + Fractal Spatial Attention (FSA), conditioned
    on the DLFD local fractal-dimension map, with a residual bypass and an
    optional fractal-consistency regulariser (variance-floor penalty that
    discourages the FD map from collapsing to a constant / dead gate).

    Every sub-component can be switched off independently for the
    leave-one-out ablation study via the constructor flags.
    """

    def __init__(self, nf, use_fcg=True, use_fsa=True, multiscale=True,
                 learnable=True, bypass=True, fractal_loss=True,
                 classical=False):
        super().__init__()
        self.use_fcg       = use_fcg
        self.use_fsa       = use_fsa
        self.use_bypass    = bypass
        self.use_frac_loss = fractal_loss

        scales = FD_SCALES if multiscale else [FD_SCALES[0]]
        self.dlfd = DLFDExtractor(scales=scales, learnable=learnable,
                                   classical=classical)

        if use_fcg:
            self.fcg_mlp = nn.Sequential(
                nn.Linear(3, max(nf // 8, 8)), nn.ReLU(inplace=True),
                nn.Linear(max(nf // 8, 8), nf),
            )
        if use_fsa:
            self.fsa_conv = nn.Conv2d(1, 1, kernel_size=1)

        self.proj_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nf, DESC_DIM, bias=False), nn.LayerNorm(DESC_DIM),
        )
        if bypass:
            self.bypass_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(nf, DESC_DIM, bias=False),
            )
            nn.init.zeros_(self.bypass_proj[2].weight)
        self.norm = nn.LayerNorm(DESC_DIM)

    def forward(self, feat):
        D_full, D_map = self.dlfd(feat)                 # (B,1,H,W), (B,1,g,g)

        gated = feat
        if self.use_fcg:
            mu = D_map.mean(dim=(2, 3))
            sd = D_map.std(dim=(2, 3))
            mx = D_map.amax(dim=(2, 3))
            stats_vec = torch.cat([mu, sd, mx], dim=1)          # (B,3)
            g_c = torch.sigmoid(self.fcg_mlp(stats_vec)).unsqueeze(-1).unsqueeze(-1)
            gated = gated * g_c

        if self.use_fsa:
            a_s = torch.sigmoid(self.fsa_conv(D_full))          # (B,1,H,W)
            gated = gated * a_s
        else:
            a_s = torch.ones_like(D_full)

        desc = self.proj_pool(gated)
        if self.use_bypass:
            desc = desc + self.bypass_proj(feat)
        out = self.norm(desc)

        frac_loss = None
        if self.use_frac_loss:
            frac_loss = F.relu(FRAC_MARGIN - D_map.std(dim=(2, 3))).mean()

        return out, D_full.detach(), a_s.detach(), frac_loss


# ==========================================================================
# Backbone factory
# ==========================================================================
def _build_backbone(timm_name):
    bb = timm.create_model(timm_name, pretrained=True, features_only=True)
    nf = bb.feature_info[-1]["num_chs"]
    return bb, nf


def _last_conv(bb):
    last = None
    for _, m in bb.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in backbone.")
    return last


# ==========================================================================
# Model variants
# ==========================================================================
class BaselineModel(nn.Module):
    """Backbone + GAP + Dropout(0.4) + Linear."""

    def __init__(self, timm_name, nc):
        super().__init__()
        self.bb, nf       = _build_backbone(timm_name)
        self.target_layer = _last_conv(self.bb)
        self.gap          = nn.AdaptiveAvgPool2d(1)
        self.drop         = nn.Dropout(0.4)
        self.head         = nn.Linear(nf, nc)

    def forward(self, x):
        feat   = self.bb(x)[-1]
        pooled = self.gap(feat).flatten(1)
        return self.head(self.drop(pooled)), None, None, None


class FCDAModel(nn.Module):
    """
    Backbone + FCDA + Dropout(0.3) + Linear.

    Stochastic depth: during training, with probability STOCH_DEPTH_P,
    bypass FCDA entirely and fall back to a plain GAP descriptor (same
    regularisation logic as the previous TCDA pipeline).
    """

    def __init__(self, timm_name, nc, use_fcg=True, use_fsa=True,
                 multiscale=True, learnable=True, bypass=True,
                 fractal_loss=True, classical=False):
        super().__init__()
        self.bb, nf       = _build_backbone(timm_name)
        self.target_layer = _last_conv(self.bb)
        self.fcda = FCDA(nf, use_fcg=use_fcg, use_fsa=use_fsa,
                          multiscale=multiscale, learnable=learnable,
                          bypass=bypass, fractal_loss=fractal_loss,
                          classical=classical)
        self.gap_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nf, DESC_DIM, bias=False), nn.LayerNorm(DESC_DIM),
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(DESC_DIM, nc)

    def forward(self, x):
        feat = self.bb(x)[-1]
        if self.training and random.random() < STOCH_DEPTH_P:
            desc = self.gap_proj(feat)
            return self.head(self.drop(desc)), None, None, None
        desc, D_map, a_s, frac_loss = self.fcda(feat)
        return self.head(self.drop(desc)), D_map, a_s, frac_loss


def create_model(name, nc):
    cfg = MODEL_CONFIGS[name]
    timm_name = BACKBONE_REGISTRY[cfg["backbone"]]
    if cfg["kind"] == "BASE":
        return BaselineModel(timm_name, nc)
    return FCDAModel(
        timm_name, nc,
        use_fcg=cfg["use_fcg"], use_fsa=cfg["use_fsa"],
        multiscale=cfg["multiscale"], learnable=cfg["learnable"],
        bypass=cfg["bypass"], fractal_loss=cfg["fractal_loss"],
        classical=cfg["classical"],
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ==========================================================================
# GradCAM
# ==========================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.act = self.grad = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "act", o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def generate(self, x, cls=None):
        self.model.zero_grad()
        logits = self.model(x)[0]
        if cls is None:
            cls = logits.argmax(1).item()
        logits[:, cls].sum().backward()
        w   = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        mn, mx = cam.min(), cam.max()
        return ((cam - mn) / (mx - mn + EPS)).cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove(); self._bh.remove()


# ==========================================================================
# Image utilities
# ==========================================================================
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def denorm(t):
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# --- CHANGE 2: fixed the black-fractal-map normalisation bug --------------
def to_map(t, vmin=None, vmax=None, percentile_stretch=False):
    """
    (1,1,H,W) tensor -> normalised (H,W) numpy array in [0,1] for display.

    OLD BEHAVIOUR (buggy): per-image min/max normalisation. Whenever a map
    was (near-)constant, (m - m.min()) / (m.max() - m.min() + EPS) collapsed
    to all-zero, which renders as solid BLACK under cmap="magma".

    NEW BEHAVIOUR:
      - For DLFD maps (percentile_stretch=False, the default), normalise
        against the FIXED theoretical range [0, 3] that D_map.clamp(0,3)
        already enforces. A flat-but-nonzero map now shows its true tone
        (e.g. mid-magma for D˜1.5) instead of collapsing to black.
      - For attention / probability-like maps already in [0,1]
        (percentile_stretch=True), use a robust 2nd/98th-percentile
        contrast stretch so a few outlier pixels don't wash out the rest
        of the map.
    """
    m = t.squeeze().detach().cpu().numpy()
    if percentile_stretch:
        lo, hi = np.percentile(m, [2, 98])
        if hi - lo < 1e-6:
            lo, hi = float(m.min()), float(m.max()) + 1e-6
    else:
        lo = 0.0 if vmin is None else vmin
        hi = 3.0 if vmax is None else vmax
    out = (m - lo) / (hi - lo + EPS)
    return np.clip(out, 0.0, 1.0)


# ==========================================================================
# Test metrics
# ==========================================================================
def save_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob, out_dir):
    rows   = []
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    per_auc = []
    for ci, cls in enumerate(classes):
        r           = report[cls]
        fpr, tpr, _ = roc_curve((y_true == ci).astype(int), y_prob[:, ci])
        cls_auc     = auc(fpr, tpr)
        per_auc.append(cls_auc)
        cm_b = confusion_matrix((y_true==ci).astype(int), (y_pred==ci).astype(int))
        tn = cm_b[0,0] if cm_b.shape==(2,2) else 0
        fp = cm_b[0,1] if cm_b.shape==(2,2) else 0
        rows.append({"model": model_name, "seed": seed, "class": cls,
                     "precision":   round(r["precision"], 4),
                     "recall":      round(r["recall"],    4),
                     "f1_score":    round(r["f1-score"],  4),
                     "support":     int(r["support"]),
                     "auc":         round(cls_auc,         4),
                     "specificity": round(tn/(tn+fp+EPS),  4)})
    acc   = accuracy_score(y_true, y_pred)
    mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mrec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    mf1   = f1_score(y_true, y_pred,        average="macro", zero_division=0)
    mauc  = float(np.mean(per_auc))
    for tag, vals in [
        ("MACRO_AVG", {"precision": mprec, "recall": mrec,
                       "f1_score": mf1, "auc": mauc}),
        ("ACCURACY",  {"f1_score": acc}),
    ]:
        row = {"model": model_name, "seed": seed, "class": tag,
               "precision": "", "recall": "", "f1_score": "",
               "support": int(y_true.shape[0]), "auc": "", "specificity": ""}
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)
    out_path = Path(out_dir) / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    [saved] {out_path.name}")
    return acc, mf1, mprec, mrec, mauc


# ==========================================================================
# Single seed: train + evaluate
# ==========================================================================
def run_one_seed(seed: int, model_name: str) -> dict:

    done_csv = SAVE_DIR / "all_results.csv"
    if done_csv.exists():
        ex   = pd.read_csv(done_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty:
            print(f"  =>  {model_name} seed={seed} already done — skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs  = get_dirs(model_name)
    paths = get_paths(model_name, seed, dirs)

    print(f"\n{'='*72}")
    print(f"  Seed {seed}  |  {model_name}")
    print(f"{'='*72}")

    # -- transforms --------------------------------------------------------
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(30),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.RandomAffine(degrees=0, shear=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # -- datasets ----------------------------------------------------------
    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    nc           = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.15, stratify=full_train.targets, random_state=seed,
    )
    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(full_val_ref, val_idx),
                              batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    # -- model + optimizer -------------------------------------------------
    model     = create_model(model_name, nc).to(DEVICE)
    bb_params   = list(model.bb.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not any(id(p)==id(q) for q in bb_params)]
    optimizer = optim.AdamW([
        {"params": bb_params,   "lr": LR * 0.1},
        {"params": head_params, "lr": LR},
    ], weight_decay=1e-2)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, label_smooth=LABEL_SMOOTH):
            super().__init__()
            self.gamma = gamma
            self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smooth,
                                             reduction="none")
        def forward(self, logits, target):
            ce   = self.ce(logits, target)
            pt   = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

    criterion = FocalLoss()

    best_loss = float("inf"); best_state = None; patience_ctr = 0
    history = {k: [] for k in ["epoch","train_loss","train_acc","val_loss","val_acc"]}

    # -- training loop -----------------------------------------------------
    for epoch in range(EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE); torch.cuda.empty_cache()
        gc.collect()

        cur_lr_head = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR)
        cur_lr_bb   = cur_lr_head * 0.1
        optimizer.param_groups[0]["lr"] = cur_lr_bb
        optimizer.param_groups[1]["lr"] = cur_lr_head

        use_cutmix = (epoch % 2 == 0)
        cm_alpha   = CUTMIX_ALPHA * 0.5 * (1 - math.cos(math.pi * min(epoch/20.0, 1.0)))
        apply_aug  = cm_alpha > 0.05

        model.train()
        tl = tc = tt = 0

        for bidx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if apply_aug and x.size(0) > 1:
                    if use_cutmix:
                        x_a, ya, yb, lam = cutmix_data(x, y, cm_alpha)
                    else:
                        x_a, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
                    out    = model(x_a)
                    logits = out[0]
                    loss   = aug_criterion(criterion, logits, ya, yb, lam)
                else:
                    out    = model(x)
                    logits = out[0]
                    loss   = criterion(logits, y)

                frac_loss = out[3]
                if frac_loss is not None:
                    loss = loss + LAMBDA_FRAC * frac_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                tl += loss.item() * x.size(0)
                tc += (logits.argmax(1) == y).sum().item()
                tt += y.size(0)
            except RuntimeError as e:
                print(f"    ! batch {bidx} skipped: {e}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

        train_loss = tl / max(tt, 1)
        train_acc  = tc / max(tt, 1)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                logits = model(x)[0]
                vl += criterion(logits, y).item() * x.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += y.size(0)
        val_loss = vl / max(vt, 1)
        val_acc  = vc / max(vt, 1)

        for k, v in zip(
            ["epoch","train_loss","train_acc","val_loss","val_acc"],
            [epoch+1, round(train_loss,6), round(train_acc,6),
             round(val_loss,6), round(val_acc,6)],
        ):
            history[k].append(v)

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Tr {train_loss:.4f}/{train_acc:.4f}  "
              f"Va {val_loss:.4f}/{val_acc:.4f}  "
              f"lr_head={cur_lr_head:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  => Early stopping.")
                break

    model.load_state_dict(best_state)

    # -- training curves ---------------------------------------------------
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history["epoch"], history["train_loss"], "b-",  lw=2.2, label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"],   "c--", lw=2.2, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], "r-",  lw=2.2, alpha=0.85, label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"],   "m--", lw=2.2, alpha=0.85, label="Val Acc")
    ax2.set_ylabel("Accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r"); ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", prop={"weight": "bold"})
    ax1.set_title(f"{model_name}  —  Seed {seed}"); ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=160, bbox_inches="tight")
    plt.close()

    # -- test inference + GradCAM + DLFD/FSA visualisation ------------------
    cam_base = paths["gradcam_base"]
    fd_base  = paths["fd_base"]
    for cls in classes:
        (cam_base / cls).mkdir(parents=True, exist_ok=True)
        (fd_base  / cls).mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model, model.target_layer)
    model.eval()
    y_true_l, y_pred_l, y_prob_l = [], [], []
    is_fcda = isinstance(model, FCDAModel)

    for i, (x, y) in enumerate(test_loader):
        try:
            x_dev = x.to(DEVICE); y_dev = y.to(DEVICE)
            with torch.set_grad_enabled(True):
                out      = model(x_dev)
                logits_e = out[0]
                prob     = F.softmax(logits_e.detach(), dim=1)
                pred     = logits_e.argmax(1).item()
                cam      = cam_engine.generate(x_dev, cls=y_dev.item())

            orig     = denorm(x[0])
            cls_name = classes[y.item()]
            tag      = "correct" if pred == y.item() else "wrong"

            # GradCAM panel
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(orig);       axes[0].axis("off"); axes[0].set_title("Original")
            axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[1].axis("off");        axes[1].set_title("GradCAM")
            axes[2].imshow(orig)
            axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
            axes[2].set_title(f"True: {cls_name}  |  Pred: {classes[pred]}  ({tag})")
            axes[2].axis("off")
            fig.suptitle(f"{model_name}  —  Seed {seed}", y=1.01)
            plt.tight_layout()
            plt.savefig(cam_base / cls_name / f"img_{i:05d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close()

            # DLFD / FSA visualisation (only every 20th test image to save space)
            if is_fcda and i % 20 == 0 and out[1] is not None:
                # CHANGE 2 applied here: fixed-range D map + percentile-stretched
                # attention map, instead of the old per-image min/max normalisation
                # that could collapse a (near-)constant map to solid black.
                d_map = to_map(out[1])
                a_map = to_map(out[2], percentile_stretch=True)
                fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
                axes[0].imshow(orig); axes[0].axis("off"); axes[0].set_title("Original")
                axes[1].imshow(d_map, cmap="magma", vmin=0, vmax=1); axes[1].axis("off")
                axes[1].set_title("Local FD map (DLFD)")
                axes[2].imshow(a_map, cmap="viridis", vmin=0, vmax=1); axes[2].axis("off")
                axes[2].set_title("FSA attention")
                axes[3].imshow(orig)
                axes[3].imshow(plt.cm.magma(d_map)[:, :, :3], alpha=0.45)
                axes[3].axis("off"); axes[3].set_title("FD overlay")
                fig.suptitle(f"{model_name}  —  Seed {seed}  —  {cls_name} ({tag})", y=1.02)
                plt.tight_layout()
                plt.savefig(fd_base / cls_name / f"fd_{i:05d}.png",
                            dpi=110, bbox_inches="tight")
                plt.close()

            y_true_l.append(y.item())
            y_pred_l.append(pred)
            y_prob_l.append(prob.cpu().numpy()[0])
        except Exception as e:
            print(f"    ! test img {i} skipped: {e}")
            continue

    cam_engine.release()
    y_true = np.array(y_true_l)
    y_pred = np.array(y_pred_l)
    y_prob = np.array(y_prob_l)

    # -- test metrics ------------------------------------------------------
    acc, mf1, mprec, mrec, mauc = save_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        out_dir=dirs["metrics"],
    )

    # -- confusion matrix --------------------------------------------------
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                linewidths=0.6, linecolor="gray", ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {model_name}  Seed {seed}")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=150, bbox_inches="tight")
    plt.close()

    # -- ROC curves --------------------------------------------------------
    roc_rows = []
    fig, ax  = plt.subplots(figsize=(9, 6))
    for j, cls in enumerate(classes):
        fpr, tpr, thr = roc_curve((y_true==j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{cls} (AUC={roc_auc:.3f})")
        for f, t, th in zip(fpr, tpr, thr):
            roc_rows.append({"model": model_name, "seed": seed, "class": cls,
                             "fpr": round(float(f),6), "tpr": round(float(t),6),
                             "threshold": round(float(th),6), "auc": round(roc_auc,6)})
    ax.plot([0,1],[0,1],"k--",lw=1.2); ax.legend(prop={"weight":"bold"})
    ax.grid(True, alpha=0.3); ax.set_title(f"ROC — {model_name}  Seed {seed}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout(); plt.savefig(paths["roc_png"], dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    result = {"seed": seed, "model": model_name,
               "acc":             round(acc,   4),
               "f1_macro":        round(mf1,   4),
               "precision_macro": round(mprec, 4),
               "recall_macro":    round(mrec,  4),
               "auc_macro":       round(mauc,  4)}
    print(f"  [ok] Acc={acc:.4f}  F1={mf1:.4f}  AUC={mauc:.4f}")
    return result


# ==========================================================================
# Statistical helpers
# ==========================================================================
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2)
                       / (na+nb-2+EPS))
    return float((a.mean()-b.mean()) / (pooled+EPS))


def effect_label(d):
    a = abs(d)
    if a >= 0.8: return "large"
    if a >= 0.5: return "medium"
    if a >= 0.2: return "small"
    return "negligible"


def load_cm(model_name, seed):
    p  = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    df = pd.read_csv(p, index_col=0)
    cls = list(df.index); cm = df.values.astype(int)
    yt, yp = [], []
    for ti in range(len(cls)):
        for pi in range(len(cls)):
            n = cm[ti, pi]
            yt.extend([ti]*n); yp.extend([pi]*n)
    return np.array(yt), np.array(yp), cls


def mcnemar_test_pair(y_true, pa, pb):
    ca = (pa==y_true); cb = (pb==y_true)
    b  = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    if b+c == 0:
        return np.nan, np.nan, b, c
    chi2_v = (abs(b-c)-1.0)**2 / (b+c)
    p = 1.0 - chi2.cdf(chi2_v, df=1)
    return chi2_v, p, b, c


def run_wilcoxon(df, model_set, tag):
    print(f"\n=== Wilcoxon Signed-Rank Test [{tag}] ===")
    rows = []
    for metric in METRICS:
        for m1, m2 in itertools.combinations(model_set, 2):
            s1 = df[df["model"]==m1][metric].values
            s2 = df[df["model"]==m2][metric].values
            if len(s1) < 2 or len(s2) < 2: continue
            try:
                w, p = stats.wilcoxon(s1, s2, zero_method="wilcox", correction=False)
            except ValueError:
                w, p = np.nan, np.nan
            d     = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            sig   = (not np.isnan(p)) and (p < ALPHA_STAT)
            rows.append({
                "metric": metric, "model_A": m1, "model_B": m2,
                "mean_A": round(s1.mean(),4), "mean_B": round(s2.mean(),4),
                "delta_A_minus_B": round(delta,6),
                "wilcoxon_W": round(w,2) if not np.isnan(w) else np.nan,
                "p_value": round(p,6) if not np.isnan(p) else np.nan,
                "cohens_d": round(d,4), "effect_size": effect_label(abs(d)),
                "significant_005": sig, "n_seeds": len(s1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR/f"wilcoxon_{tag}.csv", index=False)
    print(f"  [saved] wilcoxon_{tag}.csv"); return out


def run_mcnemar(all_seeds, model_set, tag):
    print(f"\n=== McNemar Pooled [{tag}] ===")
    pair_b, pair_c = {}, {}
    for seed in all_seeds:
        preds, y_ref = {}, None
        for m in model_set:
            try:
                yt, yp, _ = load_cm(m, seed)
                preds[m]  = yp
                if y_ref is None: y_ref = yt
            except Exception as e:
                print(f"  ! {e}"); continue
        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            _, _, b, c = mcnemar_test_pair(y_ref, preds[m1], preds[m2])
            key = (m1, m2)
            pair_b[key] = pair_b.get(key, 0) + b
            pair_c[key] = pair_c.get(key, 0) + c
    rows = []
    for key in pair_b:
        m1, m2 = key; b_t = pair_b[key]; c_t = pair_c[key]
        if b_t+c_t == 0:
            chi2_v, p = np.nan, np.nan
        else:
            chi2_v = (abs(b_t-c_t)-1.0)**2 / (b_t+c_t)
            p = 1.0 - chi2.cdf(chi2_v, df=1)
        sig    = (not np.isnan(p)) and (p < ALPHA_STAT)
        winner = m1 if b_t > c_t else (m2 if c_t > b_t else "tie")
        rows.append({
            "model_A": m1, "model_B": m2,
            "pooled_b": b_t, "pooled_c": c_t,
            "chi2": round(chi2_v,4) if not np.isnan(chi2_v) else np.nan,
            "p_value": round(p,6) if not np.isnan(p) else np.nan,
            "significant_005": sig, "winner": winner,
            "n_seeds_pooled": len(all_seeds),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR/f"mcnemar_{tag}.csv", index=False)
    print(f"  [saved] mcnemar_{tag}.csv"); return out


def build_superiority_table(df, wilc_df, mc_df, model_set, tag):
    rows = []
    for metric in METRICS:
        s_p = df[df["model"]==PROPOSED][metric].values
        for m in model_set:
            if m == PROPOSED: continue
            s_m   = df[df["model"]==m][metric].values
            if len(s_m) == 0: continue
            d     = cohens_d(s_p, s_m)
            delta = s_p.mean() - s_m.mean()
            wrow  = wilc_df[(wilc_df["metric"]==metric) & (
                ((wilc_df["model_A"]==PROPOSED) & (wilc_df["model_B"]==m)) |
                ((wilc_df["model_A"]==m) & (wilc_df["model_B"]==PROPOSED))
            )]
            p_w   = wrow.iloc[0]["p_value"] if not wrow.empty else np.nan
            mcrow = mc_df[
                ((mc_df["model_A"]==PROPOSED) & (mc_df["model_B"]==m)) |
                ((mc_df["model_A"]==m) & (mc_df["model_B"]==PROPOSED))
            ]
            p_mc  = mcrow.iloc[0]["p_value"] if not mcrow.empty else np.nan
            sig   = (delta > 0) and (
                (not np.isnan(p_mc) and p_mc < ALPHA_STAT) or
                (not np.isnan(p_w)  and p_w  < ALPHA_STAT)
            )
            rows.append({
                "Metric":            METRIC_LABELS[metric],
                "Proposed":          PROPOSED,
                "Compared_to":       m,
                "Proposed_mean±std": f"{s_p.mean():.4f}±{s_p.std(ddof=1):.4f}",
                "Other_mean±std":    f"{s_m.mean():.4f}±{s_m.std(ddof=1):.4f}",
                "Delta":             f"{delta:+.4f}",
                "Cohens_d":          f"{d:.4f}",
                "Effect":            effect_label(abs(d)),
                "Wilcoxon_p":        f"{p_w:.4f}"  if not np.isnan(p_w)  else "n/a",
                "McNemar_p":         f"{p_mc:.4f}" if not np.isnan(p_mc) else "n/a",
                "Proposed_wins":     "YES *" if sig else ("numerically" if delta > 0 else "NO"),
            })
    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR/f"superiority_table_{tag}.csv", index=False)
    print(f"\n=== Superiority Table [{tag}] ===")
    print(sup_df.to_string(index=False))
    return sup_df


# ==========================================================================
# Plots
# ==========================================================================
def save_plots(df, agg, all_seeds, model_set, tag):
    palette = sns.color_palette("tab10", len(model_set))
    shorts  = [m.replace("BASE_", "B-").replace("FCDA_", "F-") for m in model_set]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(32, 6.5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(x="model", y=metric, data=df, order=model_set,
                    palette=palette, width=0.55, ax=ax, linewidth=1.8)
        sns.stripplot(x="model", y=metric, data=df, order=model_set,
                      color="k", size=5.5, jitter=0.18, ax=ax)
        ax.set_title(METRIC_LABELS[metric]); ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticklabels(shorts, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"FCDA-Net — {tag} (n={len(all_seeds)} seeds)")
    plt.tight_layout()
    plt.savefig(SAVE_DIR/f"boxplot_{tag}.png", dpi=180, bbox_inches="tight")
    plt.close()

    for metric in METRICS:
        cm_col = f"{metric}_mean"; cs_col = f"{metric}_std"
        if cm_col not in agg.columns: continue
        valid = [m for m in model_set if m in agg.index]
        means = agg.loc[valid, cm_col].values
        stds  = agg.loc[valid, cs_col].values
        valid_shorts = [m.replace("BASE_", "B-").replace("FCDA_", "F-") for m in valid]
        x = np.arange(len(means))
        fig, ax = plt.subplots(figsize=(16, 6.5))
        bars = ax.bar(x, means, yerr=stds, capsize=7,
                      color=palette[:len(means)], edgecolor="k", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_shorts, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} ± Std")
        ax.set_title(f"{METRIC_LABELS[metric]} — {tag} (n={len(all_seeds)} seeds)")
        ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.006,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(SAVE_DIR/f"barplot_{tag}_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()

    print(f"  [ok] Plots saved for {tag}.")


def save_main_gain_plot(agg, all_seeds):
    """FCDA gain over BASE, per backbone × metric (main comparison only)."""
    gain_rows = []
    for bname in BACKBONE_NAMES:
        base_n = f"BASE_{bname}"; fcda_n = f"FCDA_{bname}"
        for metric in METRICS:
            mc = f"{metric}_mean"
            if mc not in agg.columns: continue
            bm = agg.loc[base_n, mc] if base_n in agg.index else np.nan
            fm = agg.loc[fcda_n, mc] if fcda_n in agg.index else np.nan
            gain_rows.append({"backbone": bname, "metric": METRIC_LABELS[metric],
                              "gain": round(float(fm-bm), 4)})
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(SAVE_DIR/"fcda_gain.csv", index=False)

    gp = gain_df.pivot(index="backbone", columns="metric", values="gain")
    gp = gp[[METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in gp.columns]]
    x_pos = np.arange(len(gp)); w = 0.14
    pal = sns.color_palette("Set2", len(gp.columns))
    fig, ax = plt.subplots(figsize=(14, 6))
    for ci, col in enumerate(gp.columns):
        offset = (ci - len(gp.columns)/2 + 0.5) * w
        vals   = gp[col].values
        bars   = ax.bar(x_pos+offset, vals, w, label=col,
                        color=pal[ci], edgecolor="k", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(0.001 if v>=0 else -0.004),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v>=0 else "top", fontsize=7.5, fontweight="bold")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xticks(x_pos); ax.set_xticklabels(gp.index, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("FCDA Gain (FCDA_X - BASE_X)")
    ax.set_title(f"FCDA Gain per Backbone × Metric (n={len(all_seeds)} seeds)")
    ax.legend(loc="upper right", prop={"weight":"bold"}); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR/"fcda_gain_barplot.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_pvalue_heatmap(mc_csv, model_set, tag, title):
    try:
        mc_df  = pd.read_csv(mc_csv)
        p_mat  = pd.DataFrame(np.ones((len(model_set),len(model_set))),
                              index=model_set, columns=model_set)
        for _, row in mc_df.iterrows():
            pv = row["p_value"]
            if pd.isna(pv): continue
            if row["model_A"] in p_mat.index and row["model_B"] in p_mat.columns:
                p_mat.loc[row["model_A"], row["model_B"]] = pv
                p_mat.loc[row["model_B"], row["model_A"]] = pv
        sm = {m: m.replace("BASE_", "B-").replace("FCDA_", "F-") for m in model_set}
        p_plot = p_mat.rename(index=sm, columns=sm)
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(p_plot.astype(float), annot=True, fmt=".4f",
                    cmap="RdYlGn_r", vmin=0, vmax=0.1,
                    mask=np.eye(len(model_set), dtype=bool), ax=ax,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size":8,"weight":"bold"},
                    cbar_kws={"label": "McNemar p (pooled)"})
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontweight="bold", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(SAVE_DIR/f"mcnemar_heatmap_{tag}.png", dpi=160, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"  ! mcnemar heatmap [{tag}] skipped: {e}")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":

    print("\n" + "="*72)
    print("  FCDA-Net : Fractal-Conditioned Dual-Gate Attention Network")
    print(f"  Backbones       : {BACKBONE_NAMES}")
    print(f"  Proposed        : {PROPOSED}")
    print(f"  Main comparison : {len(MAIN_MODELS)} models × {len(SEEDS)} seeds")
    print(f"  LOO ablation    : {len(LOO_MODELS)} variants ({ABLATION_BACKBONE} only)")
    print(f"  Device          : {DEVICE}")
    print("="*72)

    all_results = []

    # 1 -- train MAIN comparison + LOO ablation (union, de-duplicated)
    for model_name in ALL_MODELS:
        for seed in SEEDS:
            gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
            res = run_one_seed(seed, model_name)
            all_results.append(res)
            (pd.DataFrame(all_results)
               .drop_duplicates(subset=["model","seed"])
               .reset_index(drop=True)
               .to_csv(SAVE_DIR/"all_results.csv", index=False))

    df = (pd.read_csv(SAVE_DIR/"all_results.csv")
            .drop_duplicates(subset=["model","seed"])
            .reset_index(drop=True))

    agg_spec = {}
    for m in METRICS:
        agg_spec[m] = ["mean","std","min","max"] if m=="acc" else ["mean","std"]
    agg = df.groupby("model").agg(agg_spec).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex([m for m in ALL_MODELS if m in agg.index])
    agg.to_csv(SAVE_DIR/"summary_aggregated.csv")

    print("\n\n=== Aggregated Summary (all models) ===")
    print(agg.to_string())

    # 2 -- MAIN comparison: BASE vs FCDA across backbones ------------------
    wilc_main = run_wilcoxon(df, MAIN_MODELS, tag="main")
    mc_main   = run_mcnemar(SEEDS, MAIN_MODELS, tag="main")
    build_superiority_table(df, wilc_main, mc_main, MAIN_MODELS, tag="main")
    save_plots(df, agg, SEEDS, MAIN_MODELS, tag="main")
    save_main_gain_plot(agg, SEEDS)
    save_pvalue_heatmap(SAVE_DIR/"mcnemar_main.csv", MAIN_MODELS, "main",
                        "Pairwise McNemar p — BASE vs FCDA, all backbones")

    # 3 -- LOO ablation: FCDA_<ABLATION_BACKBONE> variants -------------------
    wilc_loo = run_wilcoxon(df, LOO_MODELS, tag="loo")
    mc_loo   = run_mcnemar(SEEDS, LOO_MODELS, tag="loo")
    build_superiority_table(df, wilc_loo, mc_loo, LOO_MODELS, tag="loo")
    save_plots(df, agg, SEEDS, LOO_MODELS, tag="loo")
    save_pvalue_heatmap(SAVE_DIR/"mcnemar_loo.csv", LOO_MODELS, "loo",
                        f"Pairwise McNemar p — FCDA_{ABLATION_BACKBONE} leave-one-out ablation")

    print(f"\n{'='*72}")
    print(f"  Section 4 outputs saved to: {SAVE_DIR.resolve()}")
    print("  NOTE: 'Proposed_wins' in the superiority tables reflects the")
    print("  actual computed statistics — report them as-is, including any")
    print("  metric/backbone where FCDA does not win; that is expected and")
    print("  strengthens rather than weakens the paper's credibility.")
    print("="*72)
