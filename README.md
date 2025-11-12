# ResNet Perturbation Analysis: Optimizer Robustness Study

A comprehensive study analyzing the robustness of ResNet models trained with different optimizers when subjected to input perturbations. This project is part of the Mahoney Group research at RISELab, investigating how optimization algorithms affect model resilience to noisy inputs.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Research Motivation](#research-motivation)
- [ML Methodology](#ml-methodology)
  - [Architecture](#architecture)
  - [Training Pipeline](#training-pipeline)
  - [Perturbation Analysis](#perturbation-analysis)
  - [Low-Rank + Sparse Decomposition](#low-rank--sparse-decomposition)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running Perturbation Analysis](#running-perturbation-analysis)
- [Experimental Setup](#experimental-setup)
- [Results Interpretation](#results-interpretation)
- [Project Structure](#project-structure)
- [References](#references)

---

## Quick Start

**Want to understand this project in 5 minutes?**

### 🚀 NEW: Interactive Web Application

**Full-featured web app** (deployable to Vercel, Netlify, etc.):
```bash
cd webapp
python3 -m http.server 8000
# Open http://localhost:8000
```

The webapp includes:
- 🎨 Real-time perturbation visualization with adjustable noise
- 📊 Interactive robustness curves for all optimizers
- 🏗️ Animated ResNet architecture explorer
- ⚙️ Optimizer comparison dashboard
- 📱 Mobile-friendly, modern dark theme
- ⚡ No installation needed - pure HTML/CSS/JS

**Deploy to Vercel in seconds:** See [`webapp/README.md`](webapp/README.md)

### 📄 Static Browser Demo

🎯 **Lightweight HTML demo** (no server required):
```bash
# Just open in your browser!
open interactive_demo.html
```

This launches a beautiful interactive dashboard with:
- 📊 Live robustness curves you can explore
- 🎨 Interactive perturbation visualization (adjust noise levels in real-time!)
- 🏗️ ResNet architecture walkthrough
- ⚙️ Optimizer comparison with detailed explanations
- 💡 Key insights and methodology

### 📊 Generate Static Visualizations

**Publication-quality PNG images:**
```bash
pip install numpy matplotlib seaborn
python3 visualization_demo.py
# Creates 5 publication-quality PNG visualizations
```

---

## Project Overview

This project investigates a fundamental question in deep learning: **Do different optimization algorithms produce models with varying robustness to input noise?**

We train ResNet models on MNIST and CIFAR-10 using multiple optimizers (SGD, Adam, Adadelta, Adahessian, Frank-Wolfe), then systematically test their performance when inputs are corrupted with:
- **Gaussian white noise**: Random pixel perturbations
- **Salt & Pepper noise**: Random black/white pixel corruption

### Key Findings

Different optimizers produce models with significantly different robustness profiles under perturbation, suggesting that the optimization trajectory affects not just final accuracy but also the learned feature representations' stability.

---

## Research Motivation

### Why Study Perturbation Robustness?

Real-world machine learning systems encounter noisy inputs from:
- Sensor noise (cameras, medical imaging)
- Transmission errors (network data)
- Adversarial attacks (security applications)
- Natural variations (lighting, weather)

Understanding which training methods produce inherently robust models can:
1. **Improve deployment reliability** without additional robustness training
2. **Guide optimizer selection** for safety-critical applications
3. **Reveal insights** about loss landscape geometry and generalization

---

## ML Methodology

### Architecture

We use **ResNet (Residual Networks)** - a deep convolutional architecture with skip connections that enables training very deep networks.

#### ResNet Architecture Overview

```
Input Image (MNIST: 1x224x224, CIFAR-10: 3x32x32)
      |
      v
┌─────────────────┐
│ Conv Layer (3x3)│  Initial convolution
│   + BatchNorm   │
│   + ReLU        │
└────────┬────────┘
         |
         v
┌─────────────────┐
│  Residual Block │  ┌──────────────────────┐
│                 │  │   Identity Shortcut  │
│  ┌──────────┐   │  │         (skip)       │
│  │ Conv 3x3 │   │  │          ↓           │
│  │ BatchNorm│   │◄─┤      ┌───────┐      │
│  │   ReLU   │   │  │      │   +   │      │
│  └────┬─────┘   │  │      └───┬───┘      │
│       |         │  │          ↓           │
│  ┌────▼─────┐   │  │       Output         │
│  │ Conv 3x3 │   │  └──────────────────────┘
│  │ BatchNorm│   │
│  └────┬─────┘   │
│       |         │
│     [Add]       │
│       |         │
│     ReLU        │
└───────┬─────────┘
        |
   [Repeat n times with increasing channels: 16→32→64]
        |
        v
┌───────────────┐
│  Global Avg   │
│    Pooling    │
└───────┬───────┘
        |
        v
┌───────────────┐
│ Fully Connected│  10 classes (MNIST/CIFAR-10)
└───────────────┘
```

#### Key Components

**Residual Block**: Solves vanishing gradient problem
```
F(x) = H(x) - x  where H(x) is the desired mapping
Output = F(x) + x = H(x)
```

**ResNet Variants** (for CIFAR-10):
- ResNet-20: [3, 3, 3] blocks → 3×2×3 + 2 = 20 layers
- ResNet-32: [5, 5, 5] blocks → 5×2×3 + 2 = 32 layers
- ResNet-56: [9, 9, 9] blocks → 9×2×3 + 2 = 56 layers
- ResNet-110: [18, 18, 18] blocks

### Training Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                              │
└──────────────────────────────────────────────────────────────────┘

Step 1: Data Preparation
├─ MNIST: 28x28 → Resize to 224x224
├─ CIFAR-10: 32x32x3 (RGB)
├─ Normalization: (pixel - mean) / std
└─ Batching: 128 samples per batch

Step 2: Model Initialization
├─ ResNet-20 (CIFAR-10) or Custom ResNet (MNIST)
├─ Kaiming Normal weight initialization
└─ Batch normalization layers

Step 3: Optimizer Selection
┌─────────────────────────────────────────────────────────┐
│ SGD          │ lr=0.01,  momentum=0.9                   │
│ Adam         │ lr=0.001, adaptive moments               │
│ Adadelta     │ lr=1.0,   adaptive learning rate         │
│ Adahessian   │ 2nd-order (Hessian) information          │
│ Frank-Wolfe  │ Constrained optimization (projection-free)│
└─────────────────────────────────────────────────────────┘

Step 4: Training Loop (110 epochs)
┌─────────────────────────────────────────────────────────┐
│ For each epoch:                                         │
│   1. Learning rate decay at epochs [30, 60, 90]         │
│      lr_new = lr_old × 0.1                              │
│                                                         │
│   2. Forward pass                                       │
│      ├─ output = model(input_batch)                     │
│      └─ loss = CrossEntropy(output, target) + L2_reg    │
│                                                         │
│   3. Backward pass                                      │
│      ├─ loss.backward()                                 │
│      └─ optimizer.step()                                │
│                                                         │
│   4. Validation                                         │
│      ├─ Compute accuracy on test set                    │
│      └─ Save best model (highest validation accuracy)   │
└─────────────────────────────────────────────────────────┘

Step 5: Model Checkpointing
└─ Save best model: {dataset}_resnet_{optimizer}_best.pkl
```

#### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 110 | Total training iterations |
| Batch Size | 128 | Training batch size |
| Weight Decay | 5e-4 | L2 regularization |
| L2 Lambda | 3e-4 | Additional L2 penalty |
| LR Decay | 0.1× | Multiplicative decay factor |
| Decay Steps | [30, 60, 90] | Epochs for LR reduction |

### Perturbation Analysis

After training, we evaluate model robustness by systematically degrading test inputs.

#### Perturbation Types

**1. Gaussian White Noise**
```python
noisy_image = original_image + noise * noise_level
where noise ~ N(0, 1)  # Standard normal distribution
```

Noise levels tested: [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]

**2. Salt & Pepper Noise**
```python
# Randomly set fraction of pixels to:
# - Maximum value (1.0) - "salt"
# - Minimum value (0.0) - "pepper"
# 50-50 split between salt and pepper
```

Noise levels tested: [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]

#### Perturbation Analysis Workflow

```
┌──────────────────────────────────────────────────────────────┐
│              PERTURBATION ANALYSIS PIPELINE                   │
└──────────────────────────────────────────────────────────────┘

Input: Trained models from different optimizers
       [SGD_model, Adam_model, Adadelta_model, ...]

For each noise level λ ∈ [0.0, 0.01, 0.03, ..., 0.25]:

  ┌──────────────────────────────────────────────────┐
  │ 1. Load Test Dataset                             │
  │    └─ MNIST or CIFAR-10 test split               │
  └──────────────┬───────────────────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────────────────┐
  │ 2. Apply Perturbation                            │
  │    ├─ White Noise: x' = x + ε*λ, ε~N(0,1)        │
  │    └─ Salt & Pepper: Randomly flip λ% of pixels  │
  │                                                  │
  │    Clamping: x' = clip(x', 0, 1)                 │
  └──────────────┬───────────────────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────────────────┐
  │ 3. Evaluate Each Model                           │
  │    For each optimizer's model:                   │
  │      ├─ predictions = model(perturbed_data)      │
  │      ├─ accuracy = correct / total               │
  │      └─ Store accuracy for this noise level      │
  └──────────────┬───────────────────────────────────┘
                 |
                 v
  ┌──────────────────────────────────────────────────┐
  │ 4. Aggregate Results                             │
  │    accuracy_curve[optimizer][noise_level] = acc  │
  └──────────────────────────────────────────────────┘

Final Output:
┌──────────────────────────────────────────────────────┐
│  Robustness Curves: Accuracy vs Noise Level          │
│                                                      │
│  Acc                                                 │
│   ^                                                  │
│   │ ●●●●●                                            │
│   │      ●●●●  ← Optimizer A (more robust)          │
│   │           ●●●                                    │
│   │              ◆◆                                  │
│   │                ◆◆◆  ← Optimizer B (less robust) │
│   │                   ◆◆◆                            │
│   └────────────────────────────────→ Noise Level    │
│                                                      │
│  Key Metrics:                                        │
│  • Area Under Curve (AUC): Overall robustness        │
│  • Critical Noise Level: Where accuracy < threshold  │
│  • Degradation Rate: Slope of accuracy decline       │
└──────────────────────────────────────────────────────┘
```

#### Evaluation Metrics

1. **Accuracy at each noise level**: Primary metric
2. **Robustness curve shape**: How gracefully accuracy degrades
3. **Critical threshold**: Noise level where accuracy drops below 50%
4. **Comparative ranking**: Which optimizer maintains highest accuracy

### Low-Rank + Sparse Decomposition

Advanced analysis using Frank-Wolfe optimization to decompose weight matrices.

#### Mathematical Framework

Every weight matrix W can be decomposed as:
```
W = L + S

where:
  L = Low-rank component (captures primary patterns)
  S = Sparse component (captures fine-grained adjustments)
```

**Optimization Objective**:
```
minimize: Loss(L + S)
subject to:
  ||L||_* ≤ τ_nuclear     (nuclear norm constraint - promotes low rank)
  ||S||_1 ≤ τ_l1          (L1 norm constraint - promotes sparsity)
```

#### Frank-Wolfe Algorithm

```
┌─────────────────────────────────────────────────────┐
│         FRANK-WOLFE OPTIMIZATION                     │
└─────────────────────────────────────────────────────┘

Initialize: L₀, S₀

For t = 1, 2, ..., max_iterations:

  1. Compute Gradient
     ∇_L = ∂Loss/∂L,  ∇_S = ∂Loss/∂S

  2. Linear Minimization Oracle (LMO)
     For low-rank: Find direction minimizing <∇_L, D_L>
                   subject to nuclear norm constraint

     For sparse: Find direction minimizing <∇_S, D_S>
                 subject to L1 norm constraint

  3. Update Parameters
     L_{t+1} = L_t + γ_t * D_L
     S_{t+1} = S_t + γ_t * D_S

     where γ_t is step size (line search or fixed)

  4. Retraction (project back to feasible set)
     Ensure constraints are satisfied
```

#### Why Low-Rank + Sparse?

**Benefits**:
1. **Compression**: Fewer parameters needed to represent weights
2. **Interpretability**: Separates global patterns (L) from local adjustments (S)
3. **Robustness**: Different decompositions may show different noise sensitivity

**Analysis Pipeline**:
```
Trained Model Weights
        |
        v
┌───────────────┐
│ SVD Analysis  │  Extract singular values of L
└───────┬───────┘
        |
        v
┌───────────────┐
│ Sparsity      │  Measure ||S||₀ (number of non-zeros)
│ Analysis      │  and distribution of sparse values
└───────┬───────┘
        |
        v
┌───────────────┐
│ Perturbation  │  Test if L+S decomposition affects
│ Testing       │  robustness differently than W alone
└───────────────┘
```

---

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/resnet-perturbations-riselab.git
cd resnet-perturbations-riselab

# Install dependencies
pip install torch torchvision
pip install numpy matplotlib scikit-learn tqdm
pip install adahessian  # For Adahessian optimizer
pip install chop-pytorch  # For Frank-Wolfe optimization
pip install seaborn pandas easydict
```

---

## Usage

### Training Models

Train ResNet models with different optimizers on MNIST or CIFAR-10.

#### Basic Training Commands

**MNIST with SGD**:
```bash
cd training-scripts
python3 resnet_training.py --optimizer sgd --dataset mnist --epochs 110
```

**CIFAR-10 with Adam**:
```bash
python3 resnet_training.py --optimizer adam --dataset cifar10 --epochs 110
```

#### All Optimizer Options

| Optimizer | Command | Default LR |
|-----------|---------|-----------|
| SGD | `--optimizer sgd` | 0.01 |
| Adam | `--optimizer adam` | 0.001 |
| Adadelta | `--optimizer adadelta` | 1.0 |
| Adahessian | `--optimizer adahessian` | Auto |

#### Additional Parameters

```bash
python3 resnet_training.py \
  --optimizer sgd \
  --dataset cifar10 \
  --epochs 110 \
  --learning_rate 0.01 \
  --weight_decay 5e-4 \
  --l2_lambda 3e-4
```

**Parameters**:
- `--weight_decay`: L2 regularization coefficient (default: 5e-4)
- `--learning_rate`: Initial learning rate (optimizer-dependent)
- `--dataset`: 'mnist' or 'cifar10'
- `--epochs`: Number of training epochs (default: 110)
- `--optimizer`: Optimization algorithm
- `--l2_lambda`: Additional L2 penalty term (default: 3e-4)

**Output**:
- Saved model: `{dataset}_resnet_{optimizer}_best.pkl`
- Location: Current working directory

### Running Perturbation Analysis

After training multiple models with different optimizers:

#### CPU-based Analysis

```bash
python3 perturbation_analysis.py
```

or

```bash
python3 CPU_perturbation_analysis.py
```

#### Jupyter Notebooks

Interactive analysis with visualizations:

```bash
jupyter notebook perturbation_analysis.ipynb
```

**Available Notebooks**:
- `perturbation_analysis.ipynb`: General perturbation analysis
- `cifar10_perturbation_analysis.ipynb`: CIFAR-10 specific analysis
- `pytorch_resnet_mnist_{optimizer}.ipynb`: Individual optimizer notebooks

#### Visualization Demo

To quickly understand the project methodology without running full experiments, use the visualization demo:

```bash
python3 visualization_demo.py
```

This generates 5 comprehensive visualizations:

1. **demo_perturbations.png**: Shows how different noise types affect images
2. **demo_robustness_curves.png**: Simulated optimizer comparison curves
3. **demo_architecture.png**: ResNet architecture diagram
4. **demo_complete_workflow.png**: End-to-end pipeline visualization
5. **demo_methodology_summary.png**: Key concepts and metrics

These visualizations are perfect for:
- Understanding the project methodology
- Presentations and reports
- Teaching and learning about robustness analysis
- Quick reference without running full experiments

**Requirements for visualization**:
```bash
pip install numpy matplotlib seaborn
```

#### Customizing Perturbation Analysis

Edit the script to modify:

```python
# Noise type
ntype = 'white'  # or 'sp' for salt & pepper

# Noise levels to test
noise_level = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]

# Models to compare (update paths)
adadelta = torch.load("./mnist_resnet_adadelta_best.pkl")
adam = torch.load("./mnist_resnet_adam_best.pkl")
sgd = torch.load("./mnist_resnet_sgd_best.pkl")
```

---

## Experimental Setup

### Complete Workflow Example

```bash
# 1. Train models with different optimizers (MNIST)
cd training-scripts

python3 resnet_training.py --optimizer sgd --dataset mnist --epochs 110
python3 resnet_training.py --optimizer adam --dataset mnist --epochs 110
python3 resnet_training.py --optimizer adadelta --dataset mnist --epochs 110

# 2. Move trained models to analysis directory
cd ..

# 3. Run perturbation analysis
python3 perturbation_analysis.py

# 4. View results
# Opens plot: plot_perturb.pdf
```

### Reproducing CIFAR-10 Experiments

```bash
cd training-scripts

# Train multiple optimizers
for opt in sgd adam adadelta; do
    python3 resnet_training.py --optimizer $opt --dataset cifar10 --epochs 110
done

# For Frank-Wolfe with low-rank + sparse decomposition
python3 low_rank_plus_sparse_CIFAR10_training.py
```

---

## Results Interpretation

### Understanding Perturbation Curves

```
Typical Results Pattern:

Accuracy
   ^
100%│●●●●●●●●●
    │         ●●●●
 90%│             ●●●  ← Robust optimizer
    │                ●
 80%│                 ●
    │◆◆◆◆◆◆
 70%│      ◆◆◆         ← Less robust optimizer
    │         ◆◆
 60%│           ◆◆
    │             ◆
 50%│              ◆
    └──────────────────────→ Noise Level
    0.0  0.05  0.1  0.15  0.2
```

**Key Observations**:
1. **Initial accuracy** (noise = 0): Measures standard model performance
2. **Curve steepness**: Faster drop = less robust to noise
3. **Plateau point**: Where accuracy stabilizes at low level
4. **Relative ordering**: Which optimizer maintains lead under noise

### Comparing Optimizers

**Typical Robustness Ranking** (from literature):
1. **SGD with momentum**: Often most robust, flatter minima
2. **Adadelta**: Adaptive but maintains some robustness
3. **Adam**: Fast training but sometimes less robust
4. **Adahessian**: Second-order information may help or hurt

**Why differences exist**:
- **Loss landscape geometry**: Different optimizers find different local minima
- **Implicit regularization**: SGD's noise acts as regularization
- **Sharpness of minima**: Flatter minima → better generalization → more robust

### Statistical Significance

When comparing results:
- Run multiple seeds (random initialization)
- Compute mean ± standard deviation across runs
- Use statistical tests (t-test) for comparing robustness

---

## Project Structure

```
resnet-perturbations-riselab/
│
├── README.md                          # This file - comprehensive documentation
├── interactive_demo.html              # 🌟 Browser-based interactive demo (open in browser!)
├── visualization_demo.py              # ⭐ Static visualization generator (requires matplotlib)
├── getData.py                         # Data loading utilities
│
├── training-scripts/                  # Model training
│   ├── resnet_training.py            # Main training script
│   ├── dataset_helpers.py            # Data preprocessing
│   ├── resnet.py                     # ResNet architecture
│   ├── tools.py                      # Training utilities
│   ├── low_rank_plus_sparse_CIFAR10_training.py  # Frank-Wolfe training
│   ├── load_checkpoint.py            # Model loading
│   └── train_models.sh               # Batch training script
│
├── analysis/                          # Advanced analysis tools
│   └── tools.py                      # Low-rank/sparse decomposition
│
├── perturbation_analysis.py          # Main perturbation testing (GPU)
├── CPU_perturbation_analysis.py      # CPU version
│
└── notebooks/                         # Interactive analysis
    ├── perturbation_analysis.ipynb
    ├── cifar10_perturbation_analysis.ipynb
    ├── pytorch_resnet_mnist_SGD.ipynb
    ├── pytorch_resnet_mnist_adam.ipynb
    ├── pytorch_resnet_mnist_adadelta.ipynb
    ├── pytorch_resnet_mnist_adahessian.ipynb
    ├── pytorch_resnet_mnist_frank_wolfe.ipynb
    └── gpu_pytorch_resnet_cifar10.ipynb
```

---

## Key Concepts Explained

### 1. Why ResNet?

ResNets enable very deep networks through skip connections:
- **Gradient flow**: Skip connections allow gradients to flow directly through network
- **Identity mapping**: Network can learn to "do nothing" if that's optimal
- **Performance**: State-of-the-art on ImageNet, CIFAR-10, etc.

### 2. Why Multiple Optimizers?

Different optimizers have different properties:
- **SGD**: Simple, converges to flat minima, good generalization
- **Adam**: Adaptive, faster convergence, may find sharp minima
- **Adadelta**: Adaptive learning rate, no manual LR tuning
- **Adahessian**: Uses curvature information (Hessian)
- **Frank-Wolfe**: Projection-free, works with constraints

### 3. What is Perturbation Robustness?

A model is robust if:
```
Small change in input → Small change in output
```

Mathematically:
```
||f(x + δ) - f(x)|| ≤ ε  for small ||δ||
```

This is related to but different from:
- **Adversarial robustness**: Targeted worst-case perturbations
- **Generalization**: Performance on unseen data
- **Out-of-distribution detection**: Recognizing unfamiliar inputs

### 4. Loss Landscape and Robustness

Research suggests:
```
Flat minima → Better generalization → More robust
Sharp minima → Overfit to training data → Less robust
```

Different optimizers find different minima in the loss landscape:
```
SGD with momentum → Flat minima (noise helps escape sharp minima)
Adam → Potentially sharp minima (fast convergence, less exploration)
```

---

## References

### Key Papers

1. **ResNet Architecture**:
   - He et al. (2016). "Deep Residual Learning for Image Recognition"
   - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. **Optimizer Comparison**:
   - Keskar et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
   - Wilson et al. (2017). "The Marginal Value of Adaptive Gradient Methods in Machine Learning"

3. **Perturbation Robustness**:
   - Goodfellow et al. (2015). "Explaining and Harnessing Adversarial Examples"
   - Madry et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks"

4. **Frank-Wolfe Optimization**:
   - Jaggi (2013). "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization"
   - Pedregosa et al. (2020). "Constrained Optimization for Machine Learning"

5. **Adahessian**:
   - Yao et al. (2020). "ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning"

### Datasets

- **MNIST**: LeCun et al. (1998). Handwritten digits, 60k train, 10k test
- **CIFAR-10**: Krizhevsky (2009). 10 classes, 50k train, 10k test

---

## Contributing

This is a research project. For questions or collaborations:
- Open an issue on GitHub
- Contact the Mahoney Group at RISELab

---

## License

[Specify your license here]

---

## Acknowledgments

- RISELab at UC Berkeley
- Mahoney Group for research guidance
- PyTorch team for the deep learning framework

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{resnet_perturbations,
  title={ResNet Perturbation Analysis: Optimizer Robustness Study},
  author={Mahoney Group, RISELab},
  year={2024},
  url={https://github.com/yourusername/resnet-perturbations-riselab}
}
```
