# OAM Dataset and SE-SAM3 Recognition

Physics-based orbital angular momentum (OAM) optical-field simulation, degraded-dataset construction, and deep-learning-based OAM mode recognition.

This repository contains the core source code, configuration files, dataset manifests, and selected experimental results for two closely related studies on OAM mode recognition under physically degraded free-space optical channels.

The work is organized as a continuous research pipeline:

1. Physics-based OAM degraded optical-field dataset construction
2. Baseline validation using conventional deep-learning models
3. SAM3 encoder adaptation for OAM mode classification
4. SE-based channel recalibration
5. Condition-wise robustness evaluation under turbulence and receiver noise

---

## Overview

Orbital angular momentum beams provide an additional spatial degree of freedom for free-space optical communication.

In practical propagation environments, however, the received optical field may be degraded by:

- atmospheric turbulence
- diffraction during long-distance propagation
- receiver noise
- mode distortion and energy redistribution

This project constructs a controlled physics-based OAM degradation pipeline and evaluates deep-learning-based mode-recognition methods under different physical channel conditions.

The repository currently covers two stages of the research.

---

# Stage 1 — Physics-based OAM Degraded Dataset Construction

A physically parameterized simulation pipeline is used to generate degraded OAM intensity images under different propagation and receiver conditions.

## Dataset Configuration

The dataset covers:

- **11 OAM topological-charge classes**
- **7 atmospheric turbulence strengths**
- **4 propagation distances**
- **5 signal-to-noise ratio (SNR) levels**
- **30 independent realizations per physical condition**
- **46,200 grayscale intensity images in total**
- **Image resolution: 224 × 224**

The full-factorial configuration is:

```text
11 OAM classes
×
7 turbulence strengths
×
4 propagation distances
×
5 SNR levels
×
30 independent realizations
=
46,200 images
```

## Physical Simulation Pipeline

The degraded optical-field generation process includes:

1. Laguerre-Gaussian OAM field generation
2. Angular-spectrum propagation
3. Atmospheric turbulence modeling
4. Receiver-noise degradation
5. Intensity-image generation
6. Dataset partitioning
7. Baseline model evaluation

The conceptual pipeline is:

```text
Laguerre-Gaussian OAM field
            ↓
Angular-spectrum propagation
            ↓
Atmospheric turbulence
            ↓
Receiver noise
            ↓
Degraded OAM intensity image
            ↓
Deep-learning recognition
```

A **ResNet-50** classifier is used as the primary baseline in the dataset-validation stage.

---

# Public Dataset

The corresponding OAM degraded optical-field dataset has been publicly released on **ScienceDB**.

## Multi-factor Orbital Angular Momentum Degraded Optical Field Dataset

**DOI:** `10.57760/sciencedb.42738`

**CSTR:** `31253.11.sciencedb.42738`

The public dataset contains:

- degraded OAM intensity images
- category labels
- training / validation / test split information
- simulation-parameter metadata
- physically controlled degradation conditions

The complete dataset is hosted externally rather than inside this GitHub repository because the raw image collection is substantially larger than the source-code package.

---

# Stage 2 — SE-SAM3 OAM Mode Recognition

Based on the same physics-structured dataset, multiple image-recognition architectures are trained and evaluated using consistent data partitions and evaluation protocols.

The compared models include:

- **ResNet-50**
- **Swin Transformer (Swin-T)**
- **SAM3 baseline**
- **SE-SAM3-PPI**

The proposed SE-SAM3-PPI framework adapts a SAM3 image encoder for OAM mode classification.

## SE-SAM3-PPI Architecture

The main processing pipeline is:

```text
Degraded OAM intensity image
            ↓
SAM3 image encoder
            ↓
SE channel recalibration
            ↓
Global average pooling
            ↓
11-class classification head
            ↓
Predicted OAM mode
```

The original segmentation-oriented decoder is not required for the OAM classification task.

Instead, the encoded feature representation is processed by:

1. **SAM3 image encoder**
2. **Squeeze-and-Excitation (SE) channel recalibration**
3. **Global average pooling**
4. **11-class classification head**

The SE module performs channel-wise feature recalibration using global feature statistics before classification.

---

# Key Results

On the held-out test set, **SE-SAM3-PPI achieved an overall classification accuracy of 81.10%**.

Compared with the matched baseline models, the reported improvement was:

- **+4.35 percentage points** over ResNet-50
- **+6.21 percentage points** over Swin-T
- **+17.40 percentage points** over the SAM3 baseline

Condition-wise evaluation was further conducted across:

- OAM order
- atmospheric turbulence strength
- propagation condition
- receiver SNR

The experiments show that severe atmospheric turbulence remains the dominant limitation for intensity-only OAM recognition.

---

# Evaluation

The evaluation pipeline includes:

- overall classification accuracy
- class-wise performance
- confusion-matrix analysis
- OAM-order-wise accuracy
- turbulence-strength-wise accuracy
- SNR-wise accuracy
- model comparison
- grouped condition analysis
- paper-figure generation
- paper-table generation

Selected outputs are retained in the repository to provide compact and reproducible experimental evidence without including the complete raw dataset or large model checkpoints.

---

# Repository Structure

```text
.
├── configs/
│   ├── simulation.yaml
│   └── training.yaml
│
├── manifests/
│   ├── all.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── outputs/
│   ├── evaluation/
│   ├── group_analysis/
│   ├── metrics/
│   ├── paper_figures/
│   └── paper_tables/
│
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_split_dataset.py
│   ├── 03_train_resnet50.py
│   ├── 03_train_sam3_baseline.py
│   ├── 03_train_sesam3.py
│   ├── 03_train_swin.py
│   ├── 04_evaluate_all_models.py
│   ├── 05_group_analysis.py
│   ├── 06_plot_paper_figures.py
│   └── 07_make_paper_tables.py
│
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── models_se_sam3.py
│   └── oam_dataset.py
│
├── check.py
├── README.md
└── requirements.txt
```

---

# Typical Workflow

The numbered scripts under `scripts/` are organized according to the experimental workflow.

```text
OAM field generation
        ↓
Physics-based degradation simulation
        ↓
Dataset generation
        ↓
Dataset split
        ↓
Model training
        ↓
Model evaluation
        ↓
Condition-wise analysis
        ↓
Paper figures and tables
```

A typical workflow is therefore:

```text
01_generate_dataset.py
        ↓
02_split_dataset.py
        ↓
03_train_*.py
        ↓
04_evaluate_all_models.py
        ↓
05_group_analysis.py
        ↓
06_plot_paper_figures.py
        ↓
07_make_paper_tables.py
```

---

# Main Source Files

## `src/oam_dataset.py`

Handles dataset loading and preprocessing for OAM intensity-image classification.

## `src/models.py`

Contains baseline model definitions and model-loading utilities used in the comparison experiments.

## `src/models_se_sam3.py`

Contains the SE-SAM3 model implementation used for OAM classification.

## `configs/simulation.yaml`

Stores physics-related simulation parameters.

## `configs/training.yaml`

Stores model-training parameters and experimental configuration.

---

# Dataset Manifests

The `manifests/` directory contains the dataset partition information used during the experiments.

```text
all.csv
train.csv
val.csv
test.csv
```

These files provide reproducible sample assignments for model training and evaluation.

Large raw image data are intentionally excluded from GitHub.

---

# Selected Experimental Outputs

The repository contains selected experiment outputs rather than all intermediate training artifacts.

Typical retained results include:

- classification reports
- confusion matrices
- per-condition accuracy tables
- OAM-order performance curves
- turbulence-strength performance curves
- SNR performance curves
- model-comparison tables

Representative analysis categories include:

```text
outputs/
├── evaluation/
├── group_analysis/
├── metrics/
├── paper_figures/
└── paper_tables/
```

---

# Environment

The experiments were conducted primarily using the following software stack:

- **Python 3.11**
- **PyTorch 2.7**
- **NumPy**
- **Pandas**
- **SciPy**
- **scikit-learn**
- **OpenCV**
- **Matplotlib**
- **timm**
- **CUDA-enabled PyTorch**

The primary experimental workstation was configured with:

- **CPU:** AMD Ryzen 9 9900X
- **Memory:** 128 GB RAM
- **GPU:** NVIDIA GeForce RTX 5090 D v2
- **GPU memory:** approximately 24 GB
- **Operating system:** Windows 11 Pro
- **Python:** 3.11
- **PyTorch:** 2.7.0
- **PyTorch CUDA runtime:** CUDA 12.8

Different experiments were maintained in isolated Conda environments.

---

# Installation

A basic environment can be created using Conda.

```bash
conda create -n oam python=3.11
conda activate oam
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

GPU-enabled PyTorch should be installed according to the appropriate NVIDIA / PyTorch CUDA configuration of the local system.

---

# Requirements

The main Python dependencies include:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
opencv-python
Pillow
tqdm
PyYAML
torch
torchvision
timm
```

Exact package versions may depend on the selected CUDA and PyTorch environment.

---

# Reproducibility Notes

This repository focuses on:

- reproducible source-code organization
- explicit simulation configuration
- explicit training configuration
- reproducible train / validation / test manifests
- selected evaluation outputs
- representative experimental figures
- representative result tables

Large raw datasets and model checkpoints are intentionally excluded.

The complete public dataset is available through ScienceDB:

**DOI:** `10.57760/sciencedb.42738`

---

# Files Not Included

To keep the repository lightweight, the following large or temporary artifacts are not included:

- complete raw image datasets
- large `.pt` / `.pth` model checkpoints
- temporary cache files
- local IDE configuration
- Python cache directories
- compressed experimental archives
- intermediate training outputs that are not required for result inspection

---

# Manuscripts

The following related manuscripts are currently **under review**.

1. **Multi-factor OAM Degraded Optical-field Dataset Construction and Baseline Validation**

2. **Physics-structured SE-SAM3 Framework for Robust Orbital Angular Momentum Mode Recognition under Atmospheric Turbulence**

The first study focuses primarily on physics-based degraded-dataset construction and baseline validation.

The second study extends the same experimental benchmark with SAM3-based representation learning, SE channel recalibration, and multi-model robustness analysis.

---

# Research Scope

The current research focuses on the intersection of:

- Orbital Angular Momentum (OAM)
- Structured light
- Free-space optical communication
- Atmospheric turbulence
- Optical-field degradation
- Deep learning
- Computer vision
- Physics-structured machine learning
- Intelligent optical-signal recognition

---

# Author

**Wei Zeng**

M.Eng. Candidate in Communication Engineering  
College of Information and Mechanical Engineering  
Shanghai Normal University

Research interests:

- Orbital Angular Momentum (OAM)
- Free-space Optical Communication
- Deep Learning
- Computer Vision
- Intelligent Optical Signal Recognition

Email: `1000569129@smail.shnu.edu.cn`

---

# License

This project is released under the MIT License unless otherwise noted.

Please refer to the repository-level `LICENSE` file for details.