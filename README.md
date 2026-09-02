# FCDA-Net: Fractal-Conditioned Dual-Gate Attention Network

## Brain Tumour Classification (4 Classes)
**Classes:** glioma | meningioma | notumor | pituitary

---

## Overview

FCDA-Net is a novel deep learning framework that integrates differentiable fractal dimension estimation into a CNN architecture through a dual-gate attention mechanism. The core innovation makes local fractal dimension estimation learnable and differentiable, folding it directly into the feature extraction pipeline.

---

## Key Innovation

Radiological literature has long used fractal dimension (FD) of tumour margins/internal texture as a hand-crafted biomarker of lesion irregularity. FCDA-Net makes this **differentiable and learnable**, enabling the network to learn optimal fractal representations directly from data.

---

## Architecture Components

### Differentiable Local Fractal Dimension (DLFD) Extractor
- Soft edge detection from backbone features
- Learnable sigmoid threshold/temperature (replaces hard box-counting)
- Per-pixel slope estimation via weighted least squares on log N(eps) vs log(1/eps)
- Fully backpropagatable

### Fractal Channel Gate (FCG)
- Channel recalibration from FD statistics (mean, std, max)
- Applies learnable gating to feature channels

### Fractal Spatial Attention (FSA)
- Spatial attention mask derived from local FD map
- Enhances regions with meaningful fractal patterns

### Residual Bypass
- Gradient highway matching TCDA v2 design
- Ensures stable gradient flow

### Fractal-Consistency Regulariser
- Variance-floor penalty to prevent FD map collapse
- Maintains meaningful fractal representations

---

## Backbones Evaluated

| Backbone | Model Name |
|----------|------------|
| DenseNet121 | `densenet121.ra_in1k` |
| MobileNetV3-Large | `mobilenetv3_large_100.ra_in1k` |
| ConvNeXt-Tiny | `convnext_tiny.in12k_ft_in1k` |

### Model Variants

**Main Comparison (3 Backbones × 2 Variants)**
- `BASE_{Backbone}` - Backbone only (baseline)
- `FCDA_{Backbone}` - Full FCDA module

**Leave-One-Out Ablation (ConvNeXt-Tiny only)**

| Variant | Description |
|---------|-------------|
| `FCDA_ConvNeXtTiny` | Full model (reference) |
| `FCDA_ConvNeXtTiny_noFCG` | Channel gate removed |
| `FCDA_ConvNeXtTiny_noFSA` | Spatial attention removed |
| `FCDA_ConvNeXtTiny_classicalFBC` | Fixed, non-differentiable box-counting |
| `BASE_ConvNeXtTiny` | Backbone only |

---

## Experimental Setup

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Learning Rate | 3e-4 (backbone: 0.1×) |
| Epochs | 25 |
| Warmup Epochs | 3 |
| Patience | 15 |
| Label Smoothing | 0.08 |
| CutMix Alpha | 1.0 |
| MixUp Alpha | 0.4 |
| Fractal Loss Weight | 0.05 |
| Fractal Margin | 0.15 |
| Stochastic Depth | 0.10 |
| Seeds for Robustness | [42, 43, 44, 45, 46] |

### Statistical Tests
- **Wilcoxon Signed-Rank Test** (paired across seeds)
- **Pooled McNemar Test** (per-pixel classification agreement)

---

## Dataset Structure

### Data Sources
- **Training:** Kaggle training set  
  [MRI Brain Tumor Dataset 4 Class - 7023 Images](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)
- **Testing:** Mendeley test set  
  [Brain Tumor Dataset](https://data.mendeley.com/datasets/zwr4ntf94j/1)

### Directory Structure

Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
