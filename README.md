# FCDA-Net: Fractal-Conditioned Dual-Gate Attention Network

## Brain Tumour Classification (4 Classes)
**Classes:** glioma | meningioma | notumor | pituitary
## Before performing the experiment install the required libraries

pip install torch torchvision timm scikit-learn scipy pandas numpy matplotlib seaborn pillow
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

# FCDA-Net: Fractal-Conditioned Dual-Gate Attention Network

## Dataset Directory Structure

The dataset must be organized exactly as follows:

```text
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each class directory must contain the corresponding brain MRI images.

For example:

```text
Dataset/
├── Training/
│   ├── glioma/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   │
│   ├── meningioma/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   │
│   ├── notumor/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   │
│   └── pituitary/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
│
└── Test/
    ├── glioma/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    │
    ├── meningioma/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    │
    ├── notumor/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    │
    └── pituitary/
        ├── image001.jpg
        ├── image002.jpg
        └── ...
```

### Important

Do **not** use the following structure:

```text
Dataset/
└── Training/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── notumor/
```

This is incorrect because the class directories are supposed to be **direct children of both `Training/` and `Test/`**.

## Required Paths in `FCDA_main.py`

The Python script currently expects:

```python
TRAIN_PATH = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH  = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"
```

Therefore, the actual filesystem should correspond to:

```text
/nfsshare/users/raghavan/Brainz/Brain tumor dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The script uses `torchvision.datasets.ImageFolder`, so each class must be represented by a separate directory directly under `Training/` and `Test/`.

## Expected Classes

The experiment contains four classification classes:

1. `glioma`
2. `meningioma`
3. `notumor`
4. `pituitary`

The directory names should remain consistent between the training and test sets.

## Dataset Loading

The dataset is loaded using:

```python
full_train = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
test_ds = datasets.ImageFolder(TEST_PATH, transform=val_tf)
```

`ImageFolder` automatically interprets each immediate subdirectory as a class.

Thus:

```text
Training/glioma/
Training/meningioma/
Training/notumor/
Training/pituitary/
```

become the four training classes, and:

```text
Test/glioma/
Test/meningioma/
Test/notumor/
Test/pituitary/
```

become the corresponding test classes.

## Experimental Design

FCDA-Net is evaluated using three pretrained CNN backbones:

* DenseNet121
* MobileNetV3-Large
* ConvNeXt-Tiny

The main comparison consists of:

```text
BASE_DenseNet121
FCDA_DenseNet121

BASE_MobileNetV3L
FCDA_MobileNetV3L

BASE_ConvNeXtTiny
FCDA_ConvNeXtTiny
```

The leave-one-out ablation study is performed only on ConvNeXt-Tiny:

```text
FCDA_ConvNeXtTiny
FCDA_ConvNeXtTiny_noFCG
FCDA_ConvNeXtTiny_noFSA
FCDA_ConvNeXtTiny_classicalFBC
BASE_ConvNeXtTiny
```

Five random seeds are used:

```text
42, 43, 44, 45, 46
```

## FCDA Components

The proposed FCDA module consists of:

1. **Differentiable Local Fractal Dimension (DLFD)**
2. **Fractal Channel Gate (FCG)**
3. **Fractal Spatial Attention (FSA)**
4. **Residual bypass**
5. **Fractal-consistency regularization**

The classical box-counting variant is included as a control condition. It uses a fixed, non-differentiable fractal descriptor and is used to investigate whether the learnable differentiable formulation provides an advantage.

## Training Configuration

Important training settings include:

```text
Image size          : 224 × 224
Batch size          : 16
Initial head LR     : 3 × 10⁻⁴
Backbone LR         : 3 × 10⁻⁵
Maximum epochs      : 25
Warm-up epochs      : 3
Early stopping      : patience = 15
Label smoothing     : 0.08
CutMix              : enabled
MixUp               : enabled
Optimizer           : AdamW
Fractal loss weight : 0.05
FD scales           : [1, 2, 4, 8]
```

## Data Augmentation

Training images are processed using:

* Random resized crop
* Random horizontal flip
* Random vertical flip
* Random rotation
* Color jitter
* Random grayscale
* Random affine shear
* ImageNet normalization

Validation and test images are resized to `224 × 224` and normalized using ImageNet mean and standard deviation.

## Output Directory

All experimental outputs are stored under:

```text
FCDA_2026/
```

The directory is organized approximately as:

```text
FCDA_2026/
├── BASE_DenseNet121/
├── FCDA_DenseNet121/
├── BASE_MobileNetV3L/
├── FCDA_MobileNetV3L/
├── BASE_ConvNeXtTiny/
├── FCDA_ConvNeXtTiny/
├── FCDA_ConvNeXtTiny_noFCG/
├── FCDA_ConvNeXtTiny_noFSA/
├── FCDA_ConvNeXtTiny_classicalFBC/
│
├── all_results.csv
├── summary_aggregated.csv
├── wilcoxon_main.csv
├── wilcoxon_loo.csv
├── mcnemar_main.csv
├── mcnemar_loo.csv
├── superiority_table_main.csv
├── superiority_table_loo.csv
├── fcda_gain.csv
├── fcda_gain_barplot.png
├── boxplot_main.png
├── boxplot_loo.png
└── ...
```

Each model directory contains metrics, curves, Grad-CAM visualizations, and fractal/FSA visualizations.

## Statistical Analysis

The experiment reports:

* Accuracy
* Macro F1
* Macro Precision
* Macro Recall
* Macro AUC

Statistical comparisons use:

* Wilcoxon signed-rank test across seeds
* Pooled McNemar test across test predictions
* Cohen's *d* effect size

The statistical significance threshold is:

```text
α = 0.05
```

Results should be reported exactly as computed. The superiority table must not be interpreted as assuming that FCDA wins every metric or every backbone.

## Visualization

For each trained model and seed, the pipeline generates:

* Training/validation loss curves
* Training/validation accuracy curves
* Confusion matrices
* ROC curves
* Grad-CAM visualizations

The DLFD visualization uses the fixed theoretical range `[0, 3]` rather than per-image min-max normalization. This prevents nearly constant but non-zero fractal maps from incorrectly appearing completely black.

## Reproducibility

p
The experiment is repeated using five seeds:

```text
42
43
44
45
46
```

The random seeds are applied to Python, NumPy, and PyTorch.

For reproducible experiments, keep the following unchanged unless there is a documented reason to modify them:

* Dataset split
* Random seeds
* Backbone versions
* Image size
* Batch size
* Learning rates
* Augmentation configuration
* Number of epochs
* Model configuration
* Statistical procedures



Before running the script, verify that the dataset has the required structure:

```text
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Then verify the paths in `FCDA_main.py`:

```python
TRAIN_PATH = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH  = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"
```

Run:

```bash
python FCDA_main.py
```

The script trains the configured models across all five seeds and subsequently generates the aggregated statistical analysis and plots.

## Important Implementation Note

The code expects the directory structure to be compatible with `ImageFolder`. There must not be an additional class-directory level between `Training/` or `Test/` and the four class folders.

Correct:

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Incorrect:

```text
Training/
└── dataset/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Also incorrect:

```text
Training/
└── notumor/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The same rule applies to the `Test/` directory.
