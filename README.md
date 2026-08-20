# OAM Deep Learning Research

Physics-based orbital angular momentum (OAM) simulation, degraded optical-field dataset construction, deep-learning recognition, and interpretable harmonic signal analysis.

This repository collects my research work on intelligent recognition of OAM optical fields under realistic free-space optical degradation conditions.

The current research focuses on the intersection of:

- Orbital Angular Momentum (OAM)
- Free-space Optical Communication
- Structured Light
- Atmospheric Turbulence
- Optical-field Degradation
- Deep Learning
- Computer Vision
- Harmonic Signal Analysis
- Physics-structured Machine Learning

---

# Research Overview

The repository currently contains two major research pipelines.

## Project 1 — OAM Degraded Dataset and SE-SAM3 Recognition

This project focuses on physics-based degraded OAM dataset construction and deep-learning-based mode recognition.

Main components include:

- Laguerre-Gaussian OAM field generation
- Angular-spectrum propagation
- Atmospheric turbulence simulation
- Receiver-noise degradation
- Multi-factor OAM dataset construction
- ResNet-50 baseline validation
- Swin Transformer comparison
- SAM3 encoder adaptation
- SE channel recalibration
- Condition-wise robustness analysis

Repository:

[`oam_dataset_and_se_sam3/`](./oam_dataset_and_se_sam3/)

---

## Project 2 — MC-VWLS Harmonic Recognition

This project investigates a training-free and interpretable recognition method for discrete OAM superposition states under joint turbulence, partial occlusion, and receiver noise.

Main components include:

- OAM superposition-state simulation
- Atmospheric turbulence
- Partial aperture occlusion
- Receiver-noise modeling
- Polar-coordinate sampling
- Harmonic feature extraction
- Valid-region masking
- Mask-constrained normalization
- Weighted least-squares fitting
- Ablation experiments
- Statistical comparison
- Runtime benchmarking

Repository:

[`mc_vwls_harmonic_recognition/`](./mc_vwls_harmonic_recognition/)

---

# Research Pipeline

The overall research can be summarized as:

```text
Physical OAM field generation
            ↓
Free-space propagation
            ↓
Atmospheric turbulence
            ↓
Receiver degradation
            ↓
Structured optical observation
            ↓
 ┌───────────────────────────────┐
 │                               │
 ↓                               ↓
Deep-learning recognition   Harmonic signal analysis
 │                               │
 ↓                               ↓
SE-SAM3 / CNN / ViT        MC-VWLS / ULS / DAF
 │                               │
 └───────────────┬───────────────┘
                 ↓
        Robust OAM recognition
```

---

# Public Dataset

A multi-factor degraded OAM optical-field dataset has been publicly released on **ScienceDB**.

## Multi-factor Orbital Angular Momentum Degraded Optical Field Dataset

**DOI:** `10.57760/sciencedb.42738`

**CSTR:** `31253.11.sciencedb.42738`

The dataset contains:

- **11 OAM topological-charge classes**
- **7 atmospheric turbulence strengths**
- **4 propagation distances**
- **5 SNR levels**
- **30 independent realizations per physical condition**
- **46,200 grayscale optical-field intensity images**
- **224 × 224 image resolution**

Full-factorial configuration:

```text
11 OAM classes
× 7 turbulence strengths
× 4 propagation distances
× 5 SNR levels
× 30 realizations
=
46,200 images
```

The dataset includes:

- optical-field images
- class labels
- train / validation / test split files
- simulation parameters
- physical degradation metadata

The full dataset is hosted on ScienceDB rather than directly inside GitHub.

---

# Project 1 — OAM Dataset and SE-SAM3

Directory:

```text
oam_dataset_and_se_sam3/
```

The project contains:

```text
oam_dataset_and_se_sam3/
├── configs/
├── manifests/
├── outputs/
├── scripts/
├── src/
├── README.md
└── requirements.txt
```

## Models

The main comparison models include:

- ResNet-50
- Swin Transformer (Swin-T)
- SAM3 baseline
- SE-SAM3-PPI

The SE-SAM3-PPI processing pipeline is:

```text
Degraded OAM image
        ↓
SAM3 image encoder
        ↓
SE channel recalibration
        ↓
Global average pooling
        ↓
11-class classification head
        ↓
OAM mode prediction
```

## Selected Result

On the held-out test set, the reported SE-SAM3-PPI accuracy is:

**81.10%**

Reported improvement relative to matched baselines:

- **+4.35 percentage points** over ResNet-50
- **+6.21 percentage points** over Swin-T
- **+17.40 percentage points** over the SAM3 baseline

Additional evaluation includes:

- OAM-order-wise performance
- turbulence-strength-wise performance
- SNR-wise performance
- confusion-matrix analysis

For detailed implementation and experimental outputs, see:

[`oam_dataset_and_se_sam3/README.md`](./oam_dataset_and_se_sam3/README.md)

---

# Project 2 — MC-VWLS Harmonic Recognition

Directory:

```text
mc_vwls_harmonic_recognition/
```

The project contains:

```text
mc_vwls_harmonic_recognition/
├── config/
├── results/
├── src/
├── README.md
└── requirements.txt
```

The source code is organized into:

```text
src/
├── algorithms/
├── dataset/
├── evaluation/
├── models/
├── physics/
├── validation/
└── visualization/
```

## MC-VWLS Pipeline

```text
Received OAM intensity image
            ↓
Valid-region mask
            ↓
Polar-coordinate sampling
            ↓
Angular harmonic representation
            ↓
Mask-constrained normalization
            ↓
Weighted least-squares fitting
            ↓
State parameter estimation
            ↓
Discrete OAM-state recognition
```

## Experimental Scale

The study includes:

- **32 discrete OAM superposition-state classes**
- **44,800 base degraded samples**
- **33,600 noisy test observations**

## Selected Results

Reported MC-VWLS performance:

- **Joint recognition accuracy:** 84.8125%
- **Order recognition accuracy:** 89.3452%
- **Phase recognition accuracy:** 85.7917%
- **End-to-end runtime:** approximately 2.318 ms per observation

The project also contains:

- ablation studies
- pairwise statistical comparisons
- runtime benchmarks
- failure-case analysis
- turbulence / distance / occlusion / SNR condition analysis
- experiment-consistency checks

For detailed implementation and experimental outputs, see:

[`mc_vwls_harmonic_recognition/README.md`](./mc_vwls_harmonic_recognition/README.md)

---

# Selected Technical Contributions

The current repository demonstrates several research and engineering components.

## Physics-based Simulation

- Laguerre-Gaussian OAM field generation
- angular-spectrum propagation
- atmospheric phase-screen simulation
- turbulence degradation
- receiver-noise modeling
- partial occlusion modeling

## Deep Learning

- PyTorch-based model training
- ResNet-50
- Swin Transformer
- SAM3 encoder adaptation
- Squeeze-and-Excitation modules
- classification-head modification
- GPU training and evaluation

## Signal Processing

- polar-coordinate sampling
- harmonic feature extraction
- mask-supported normalization
- weighted least-squares fitting
- robust state estimation

## Experimental Engineering

- YAML-based configuration
- train / validation / test manifests
- condition-wise evaluation
- confusion matrices
- statistical significance testing
- ablation analysis
- runtime benchmarking
- reproducibility checks
- automated paper figure and table generation

---

# Repository Structure

```text
OAM-DeepLearning-Research/
│
├── oam_dataset_and_se_sam3/
│   ├── configs/
│   ├── manifests/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── mc_vwls_harmonic_recognition/
│   ├── config/
│   ├── results/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── .gitignore
├── LICENSE
└── README.md
```

---

# Experimental Environment

The main experimental workstation used in this research is configured with:

- **CPU:** AMD Ryzen 9 9900X
- **Memory:** 128 GB RAM
- **GPU:** NVIDIA GeForce RTX 5090 D v2
- **GPU memory:** approximately 24 GB
- **Operating system:** Windows 11 Pro
- **Python:** 3.11
- **PyTorch:** 2.7.0
- **PyTorch CUDA runtime:** CUDA 12.8
- **Environment management:** Conda

Different projects are maintained in separate Conda environments.

---

# Main Software Stack

The research workflow mainly uses:

```text
Python
PyTorch
NumPy
SciPy
Pandas
OpenCV
Matplotlib
scikit-learn
timm
h5py
PyYAML
Git
Conda
CUDA
```

---

# Reproducibility

The repository retains selected materials required to inspect or reproduce the main research workflow, including:

- source code
- simulation configurations
- training configurations
- dataset manifests
- experiment evaluation scripts
- selected experimental outputs
- figures
- result tables
- statistical analysis
- validation reports

Large generated datasets, caches, and model checkpoints are intentionally excluded where appropriate.

---

# Research Outputs

Current research outputs include:

## Public Dataset

**Multi-factor Orbital Angular Momentum Degraded Optical Field Dataset**

ScienceDB  
DOI: `10.57760/sciencedb.42738`

## Manuscripts Under Review

1. **Multi-factor OAM Degraded Optical-field Dataset Construction and Baseline Validation**

2. **Physics-structured SE-SAM3 Framework for Robust Orbital Angular Momentum Mode Recognition under Atmospheric Turbulence**

3. **Mask-constrained Harmonic Recognition of Discrete Orbital Angular Momentum Superposition States under Joint Turbulence and Occlusion Degradation**

All three manuscripts are currently under review.

---

# Current Research Interests

- Orbital Angular Momentum
- Structured Light
- Free-space Optical Communication
- Atmospheric Turbulence
- Optical Signal Processing
- Deep Learning
- Computer Vision
- Physics-guided Machine Learning
- Intelligent Communication Systems

---

# Author

**Wei Zeng**

M.Eng. Candidate in Communication Engineering  
College of Information and Mechanical Engineering  
Shanghai Normal University

Research areas:

- OAM optical communication
- intelligent optical-signal recognition
- deep learning
- computer vision
- signal processing

Email: `1000569129@smail.shnu.edu.cn`

---

# License

This repository is released under the MIT License unless otherwise noted.

See:

[`LICENSE`](./LICENSE)

for details.