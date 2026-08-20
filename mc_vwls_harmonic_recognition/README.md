# MC-VWLS Harmonic Recognition for OAM Superposition States

Physics-based simulation and mask-constrained harmonic recognition of discrete orbital angular momentum (OAM) superposition states under atmospheric turbulence, partial occlusion, and receiver noise.

This repository contains the core source code, physical simulation modules, evaluation pipeline, validation scripts, and selected experimental results for a training-free OAM recognition study based on mask-constrained weighted harmonic fitting.

---

# Overview

Orbital angular momentum (OAM) superposition states can encode information through structured spatial intensity patterns.

In practical free-space optical propagation, however, the received optical field may be degraded by multiple physical factors, including:

- atmospheric turbulence
- propagation-induced distortion
- partial aperture occlusion
- receiver noise
- energy redistribution
- structural deformation of the intensity pattern

Instead of relying exclusively on end-to-end deep-learning classification, this project investigates a physics-guided and interpretable recognition pipeline based on:

- polar-coordinate sampling
- harmonic feature extraction
- valid-region masking
- weighted least-squares fitting
- physically constrained decision rules

The core method is referred to as **MC-VWLS**.

---

# Research Pipeline

The overall workflow is:

```text
OAM superposition-state generation
                ↓
Free-space propagation
                ↓
Atmospheric turbulence
                ↓
Partial occlusion
                ↓
Receiver noise
                ↓
Valid-region mask construction
                ↓
Polar sampling
                ↓
Harmonic feature extraction
                ↓
Mask-constrained weighted fitting
                ↓
OAM state recognition
```

The project integrates physical simulation, signal processing, algorithmic recognition, ablation experiments, statistical testing, and runtime evaluation in a single reproducible workflow.

---

# Physical Simulation

The simulation modules are organized under:

```text
src/physics/
```

and include:

- Laguerre-Gaussian field generation
- angular-spectrum propagation
- atmospheric phase-screen simulation
- turbulence degradation
- receiver-noise modeling

Key modules include:

```text
src/physics/
├── angular_spectrum.py
├── lg_mode.py
├── phase_screen.py
├── receiver_noise.py
└── turbulence.py
```

These modules are used to construct physically controlled OAM observations under different propagation conditions.

---

# Occlusion Modeling

Partial aperture occlusion is explicitly included in the experimental pipeline.

The occlusion-generation procedure is used to construct degraded observations with different valid-region geometries.

The recognition algorithms therefore operate not only on the observed intensity distribution, but also on the spatial support that remains valid after occlusion.

This allows the method to distinguish between:

```text
observed region
        ↓
valid physical support
        ↓
harmonic information extraction
```

rather than treating missing observations as ordinary zero-valued measurements.

---

# Dataset and Experimental Scale

The experimental dataset contains multiple OAM superposition states under combinations of:

- atmospheric turbulence
- propagation distance
- occlusion level
- receiver-noise condition

The study includes:

- **32 discrete OAM superposition-state classes**
- **44,800 base degraded samples**
- **33,600 noisy test observations**

The dataset generation and partitioning code is located under:

```text
src/dataset/
```

with the main modules:

```text
src/dataset/
├── create_group_split_v1.py
├── dataloader.py
├── extract_turbulence_base.py
├── generate_corrected_occlusion.py
├── generate_dataset.py
└── split_dataset.py
```

Dataset split information is provided under:

```text
data/manifest/
├── group_split_v1.csv
├── sample_split_v1.csv
└── sample_split_v1.npz
```

Large generated optical-field datasets are not included in this GitHub repository.

---

# Recognition Algorithms

The main signal-processing and recognition algorithms are implemented under:

```text
src/algorithms/
```

The directory contains:

```text
src/algorithms/
├── __init__.py
├── daf.py
├── harmonic_fit.py
├── mc_vwls.py
├── polar_sampling.py
├── rvnv.py
└── uls.py
```

The code includes the proposed method and several comparison or ablation variants.

---

# MC-VWLS

The core recognition method is **MC-VWLS**.

Its conceptual pipeline is:

```text
Received intensity image
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
Discrete OAM-state decision
```

The method combines spatial-support information with harmonic fitting in order to improve robustness when the received field contains missing or severely degraded regions.

---

# Polar Sampling

The received intensity field is transformed from Cartesian coordinates into a polar representation.

Conceptually:

```text
I(x, y)
   ↓
I(r, θ)
```

This representation makes angular periodicity and harmonic structure easier to analyze.

The implementation is located in:

```text
src/algorithms/polar_sampling.py
```

Dedicated validation code is also included to verify the sampling procedure.

---

# Harmonic Fitting

The angular intensity distribution is represented through harmonic components.

The fitting procedure is implemented in:

```text
src/algorithms/harmonic_fit.py
```

The harmonic representation is used to extract state-dependent periodic structure from the received intensity pattern.

This provides an interpretable alternative to treating the entire degraded image as an unconstrained feature vector.

---

# Mask-Constrained Processing

A central component of the method is the use of a valid-region mask.

The mask identifies which spatial observations should contribute to feature normalization and fitting.

Conceptually:

```text
Full observation
      ↓
Occlusion / invalid regions
      ↓
Valid-region mask
      ↓
Mask-supported normalization
      ↓
Harmonic estimation
```

This avoids allowing invalid or occluded regions to distort the estimated harmonic structure.

---

# Comparison Methods

The experimental code contains several recognition methods or variants used for comparison and ablation.

These include modules corresponding to:

- DAF
- ULS
- RVNV
- MC-VWLS

Additional baseline and ablation evaluations are implemented under:

```text
src/evaluation/
```

---

# Evaluation Pipeline

The evaluation modules include:

```text
src/evaluation/
├── benchmark_runtime_complexity.py
├── bias_free_harmonic.py
├── evaluate_mc_vwls_ablation.py
├── evaluate_mc_vwls_subset.py
├── evaluate_mc_vwls_tuned_full_test.py
├── evaluate_traditional_methods.py
├── metrics.py
├── plot_confusion_matrix.py
├── plot_failure_cases.py
├── plot_main_results.py
├── runtime.py
├── statistical_comparison.py
├── statistics.py
├── summarize_by_turbulence_distance.py
├── summarize_confusion_patterns.py
├── summarize_error_types_by_condition.py
├── update_experiment_results_inventory.py
└── validate_final_experiment_consistency.py
```

The evaluation includes:

- overall recognition accuracy
- component-wise recognition accuracy
- confusion-matrix analysis
- turbulence-conditioned evaluation
- propagation-distance-conditioned evaluation
- SNR-conditioned evaluation
- occlusion-conditioned evaluation
- ablation experiments
- pairwise statistical comparison
- failure-case analysis
- runtime benchmarking
- experiment-consistency validation

---

# Key Results

For the reported joint-recognition task, MC-VWLS achieved:

- **Joint recognition accuracy: 84.8125%**
- **Order recognition accuracy: 89.3452%**
- **Phase recognition accuracy: 85.7917%**

The experiments further investigate recognition performance under combinations of:

- atmospheric turbulence
- propagation distance
- occlusion
- receiver noise

The reported end-to-end processing time of MC-VWLS is approximately:

- **2.318 ms per observation**

under the experimental configuration used in the study.

---

# Ablation Analysis

The project includes explicit ablation experiments designed to separate the contributions of different components.

A key observation is that much of the robustness improvement is associated with **mask-supported processing and normalization**.

The experiments indicate that the difference between Mask-ULS and MC-VWLS can be relatively small under some evaluation settings.

This suggests that:

```text
valid-region masking
        +
mask-supported normalization
```

provides a substantial portion of the improvement, while weighting introduces a more limited incremental contribution under some conditions.

This result is retained rather than hidden because the purpose of the ablation study is to identify which components actually contribute to performance.

---

# Statistical Analysis

Statistical comparison is included to evaluate whether observed differences between methods are consistent across the test set.

The repository contains:

- confidence-interval analysis
- pairwise statistical comparison
- condition-wise summaries
- consistency checks

Selected outputs are stored under:

```text
results/csv/
results/tables/
results/validation/
```

---

# Runtime and Complexity

Runtime benchmarking is included to evaluate the computational cost of the proposed method.

Relevant modules include:

```text
src/evaluation/benchmark_runtime_complexity.py
src/evaluation/runtime.py
```

Selected runtime results are provided in:

```text
results/csv/
```

and visualized under:

```text
results/figures/
```

---

# Validation

A dedicated validation layer is included under:

```text
src/validation/
```

The project contains checks for:

- physical consistency of generated HDF5 data
- occlusion generation
- experiment-manifest integrity
- comparison baselines
- harmonic fitting
- polar sampling
- receiver-noise generation
- paper-output consistency
- final experiment consistency

This validation layer was used during experiment development to identify implementation or data-processing errors before final result generation.

---

# Repository Structure

```text
.
├── config/
│   └── frozen_v1.yaml
│
├── data/
│   └── manifest/
│       ├── group_split_v1.csv
│       ├── sample_split_v1.csv
│       └── sample_split_v1.npz
│
├── results/
│   ├── csv/
│   ├── figures/
│   ├── tables/
│   └── validation/
│
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── daf.py
│   │   ├── harmonic_fit.py
│   │   ├── mc_vwls.py
│   │   ├── polar_sampling.py
│   │   ├── rvnv.py
│   │   └── uls.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── create_group_split_v1.py
│   │   ├── dataloader.py
│   │   ├── extract_turbulence_base.py
│   │   ├── generate_corrected_occlusion.py
│   │   ├── generate_dataset.py
│   │   └── split_dataset.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark_runtime_complexity.py
│   │   ├── bias_free_harmonic.py
│   │   ├── evaluate_mc_vwls_ablation.py
│   │   ├── evaluate_mc_vwls_subset.py
│   │   ├── evaluate_mc_vwls_tuned_full_test.py
│   │   ├── evaluate_traditional_methods.py
│   │   ├── metrics.py
│   │   ├── plot_confusion_matrix.py
│   │   ├── plot_failure_cases.py
│   │   ├── plot_main_results.py
│   │   ├── runtime.py
│   │   ├── statistical_comparison.py
│   │   ├── statistics.py
│   │   ├── summarize_by_turbulence_distance.py
│   │   ├── summarize_confusion_patterns.py
│   │   ├── summarize_error_types_by_condition.py
│   │   └── validate_final_experiment_consistency.py
│   │
│   ├── models/
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── angular_spectrum.py
│   │   ├── lg_mode.py
│   │   ├── phase_screen.py
│   │   ├── receiver_noise.py
│   │   └── turbulence.py
│   │
│   ├── validation/
│   │
│   ├── visualization/
│   │
│   ├── __init__.py
│   └── main.py
│
├── README.md
└── requirements.txt
```

---

# Selected Result Files

The `results/` directory contains selected experimental outputs.

## CSV Results

Representative outputs include:

- main recognition results
- ablation summaries
- confidence intervals
- confusion matrices
- runtime benchmarks
- condition-wise summaries
- pairwise statistical comparisons

## Figures

Representative figures include:

- overall method comparison
- recognition accuracy versus occlusion
- recognition accuracy versus SNR
- turbulence-distance heatmap
- confusion matrix
- failure-case analysis
- runtime comparison
- receiver-noise quality control
- occlusion quality control

## Tables

The repository includes structured result tables such as:

```text
Table1_dataset_statistics.csv
Table2_occlusion_validation.csv
Table3_noise_validation.csv
Table4_main_results.csv
Table5_ablation_results.csv
Table6_cn2_distance_results.csv
Table7_statistical_significance.csv
```

---

# Configuration

The primary frozen experiment configuration is stored in:

```text
config/frozen_v1.yaml
```

The purpose of the frozen configuration is to keep key experimental parameters fixed once the final evaluation protocol is established.

This reduces accidental parameter changes between experimental runs.

---

# Environment

The project was developed primarily in a Python scientific-computing environment.

The main software stack includes:

- Python 3.11
- NumPy
- SciPy
- Pandas
- Matplotlib
- scikit-learn
- h5py
- PyYAML
- OpenCV
- PyTorch
- torchvision

The primary workstation used in the broader OAM research workflow includes:

- AMD Ryzen 9 9900X
- 128 GB RAM
- NVIDIA GeForce RTX 5090 D v2
- Windows 11 Pro
- Python 3.11
- Conda-based environment isolation

Although MC-VWLS itself is primarily an analytical / signal-processing recognition pipeline, GPU-capable infrastructure is also available for comparison-model and related OAM experiments.

---

# Installation

A basic Conda environment can be created using:

```bash
conda create -n mc_vwls python=3.11
conda activate mc_vwls
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Typical Workflow

A representative workflow is:

```text
Physical OAM field generation
          ↓
Turbulence simulation
          ↓
Occlusion generation
          ↓
Receiver-noise generation
          ↓
Dataset split
          ↓
Polar sampling
          ↓
Harmonic fitting
          ↓
MC-VWLS recognition
          ↓
Baseline comparison
          ↓
Ablation analysis
          ↓
Statistical analysis
          ↓
Runtime benchmark
          ↓
Figures and tables
```

---

# Reproducibility Notes

The repository retains:

- physical simulation code
- algorithm implementation
- frozen configuration
- data-split manifests
- evaluation scripts
- statistical-analysis scripts
- selected result tables
- representative figures
- validation outputs

Large generated data files are intentionally excluded from GitHub.

These include:

- raw optical-field datasets
- turbulence caches
- large HDF5 files
- intermediate generated observations
- large binary experimental artifacts

---

# Files Not Included

To keep the public repository lightweight, the following are intentionally excluded:

```text
data/cache/
data/generated/
data/raw/
```

as well as:

- large `.h5` files
- temporary experiment archives
- IDE metadata
- local caches
- compressed research packages
- unpublished manuscript-editing files

---

# Manuscript

The related manuscript is currently **under review**.

The study focuses on mask-constrained harmonic recognition of discrete OAM superposition states under joint turbulence, occlusion, and receiver-noise degradation.

The manuscript evaluates:

- recognition accuracy
- robustness under multiple degradation conditions
- ablation behavior
- statistical significance
- runtime efficiency
- failure patterns

---

# Research Scope

The project lies at the intersection of:

- Orbital Angular Momentum (OAM)
- structured light
- free-space optical communication
- atmospheric turbulence
- physical signal modeling
- harmonic analysis
- weighted least squares
- robust signal recognition
- interpretable optical-signal processing

---

# Author

**Wei Zeng**

M.Eng. Candidate in Communication Engineering  
College of Information and Mechanical Engineering  
Shanghai Normal University

Research interests:

- Orbital Angular Momentum
- Free-space Optical Communication
- Signal Processing
- Deep Learning
- Computer Vision
- Intelligent Optical Signal Recognition

Email: `1000569129@smail.shnu.edu.cn`

---

# License

This project follows the repository-level MIT License unless otherwise noted.

Please refer to the root-level `LICENSE` file for details.