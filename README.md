FCDA-Net: Fractal-Conditioned Dual-Gate Attention Network
Brain Tumour Classification (4 Classes)
Classes: glioma | meningioma | notumor | pituitary

Overview
FCDA-Net is a novel deep learning framework that integrates differentiable fractal dimension estimation into a CNN architecture through a dual-gate attention mechanism. The core innovation makes local fractal dimension estimation learnable and differentiable, folding it directly into the feature extraction pipeline.

Key Innovation
Radiological literature has long used fractal dimension (FD) of tumour margins/internal texture as a hand-crafted biomarker of lesion irregularity. FCDA-Net makes this differentiable and learnable, enabling the network to learn optimal fractal representations directly from data.

Architecture Components
1. Differentiable Local Fractal Dimension (DLFD) Extractor
Soft edge detection from backbone features

Learnable sigmoid threshold/temperature (replaces hard box-counting)

Per-pixel slope estimation via weighted least squares on log N(eps) vs log(1/eps)

Fully backpropagatable

2. Fractal Channel Gate (FCG)
Channel recalibration from FD statistics (mean, std, max)

Applies learnable gating to feature channels

3. Fractal Spatial Attention (FSA)
Spatial attention mask derived from local FD map

Enhances regions with meaningful fractal patterns

4. Residual Bypass
Gradient highway matching TCDA v2 design

Ensures stable gradient flow

5. Fractal-Consistency Regulariser
Variance-floor penalty to prevent FD map collapse

Maintains meaningful fractal representations

Backbones Evaluated
Backbone	Model Name
DenseNet121	densenet121.ra_in1k
MobileNetV3-Large	mobilenetv3_large_100.ra_in1k
ConvNeXt-Tiny	convnext_tiny.in12k_ft_in1k
Model Variants
Main Comparison (3 Backbones × 2 Variants)
BASE_{Backbone} - Backbone only (baseline)

FCDA_{Backbone} - Full FCDA module

Leave-One-Out Ablation (ConvNeXt-Tiny only)
Variant	Description
FCDA_ConvNeXtTiny	Full model (reference)
FCDA_ConvNeXtTiny_noFCG	Channel gate removed
FCDA_ConvNeXtTiny_noFSA	Spatial attention removed
FCDA_ConvNeXtTiny_classicalFBC	Fixed, non-differentiable box-counting
BASE_ConvNeXtTiny	Backbone only
Experimental Setup
Hyperparameters
Parameter	Value
Image Size	224 × 224
Batch Size	16
Learning Rate	3e-4 (backbone: 0.1×)
Epochs	25
Warmup Epochs	3
Patience	15
Label Smoothing	0.08
CutMix Alpha	1.0
MixUp Alpha	0.4
Fractal Loss Weight	0.05
Fractal Margin	0.15
Stochastic Depth	0.10
Seeds for Robustness
[42, 43, 44, 45, 46]

Statistical Tests
Wilcoxon Signed-Rank Test (paired across seeds)

Pooled McNemar Test (per-pixel classification agreement)

Dataset Structure
text
Training- Kaggle training set (https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)
Testing- Mendeley test set (https://data.mendeley.com/datasets/zwr4ntf94j/1)
/nfsshare/users/raghavan/Brainz/Brain tumor dataset/


Dataset Structure:
Dataset/

Training/
    glioma/
    meningioma/
    pituitary/
    notumor/

Test/
    glioma/
    meningioma/
    pituitary/
    notumor/


    
Visualisations Generated
For each model × seed:

Training/Validation Curves (loss & accuracy)

Confusion Matrices (CSV + PNG)

ROC Curves (with AUC per class)

GradCAM Heatmaps (per test image)

Output Directory Structure
text
FCDA_2026/
{model_name}/
│    metrics/
│    test_metrics_seed{seed}.csv
│   │── cm_seed{seed}.{csv,png}
│   │── roc_data_seed{seed}.csv
│   │   ├── roc_seed{seed}.png
│   │   └── history_seed{seed}.csv
│   ├── curves/
│   │   └── curves_seed{seed}.png
│   ├── gradcam/
│   │   └── seed{seed}/{class}/img_*.png
│   |
│   └── checkpoint_seed{seed}.pt
├── all_results.csv
├── summary_aggregated.csv
├── wilcoxon_main.csv
├── mcnemar_main.csv
├── superiority_table_main.csv
├── wilcoxon_loo.csv
├── mcnemar_loo.csv
├── superiority_table_loo.csv
├── boxplot_{main,loo}.png
├── barplot_{main,loo}_{metric}.png
├── fcda_gain.csv
├── fcda_gain_barplot.png
└── mcnemar_heatmap_{main,loo}.png

Running the Code
bash
python FCDA_main.py
Requirements
bash
pip install torch torchvision timm scikit-learn scipy pandas numpy matplotlib seaborn pillow
GPU Configuration
Set gpu_id = 1 (or your preferred GPU index) in the configuration section.

Results Interpretation
Superiority Tables
The generated superiority_table_{main,loo}.csv reports:

Mean ± std for each metric

Cohen's d effect size

Wilcoxon p-value

McNemar p-value

"Proposed_wins" indicator:

YES * = statistically significant win

numerically = positive but not statistically significant

NO = proposed model underperforms

Note: Any metric/backbone where FCDA does not win is expected and strengthens the paper's credibility. Report results as-is.

Metrics Tracked
Metric	Description
acc	Accuracy
f1_macro	Macro-averaged F1
precision_macro	Macro-averaged Precision
recall_macro	Macro-averaged Recall
auc_macro	Macro-averaged AUC
Ablation Study Insights
The leave-one-out ablation on ConvNeXt-Tiny isolates the contribution of:

Learnable FD estimation (vs. classical fixed box-counting)

Channel gating (FCG)

Spatial attention (FSA)

This design cleanly demonstrates the methodological claim that making fractal dimension learnable provides meaningful benefit over fixed hand-crafted features.

Citation
If you use this code, please cite:

bibtex
@article{FCDA-Net,
  title={FCDA-Net: Fractal-Conditioned Dual-Gate Attention Network for Brain Tumour Classification},
  author={Raghavan et al.},
  year={2026}
}
