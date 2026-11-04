# Anchor-Scarce GNN Localisation (IEEE SPL)
## Master Architectural, Methodological & Theoretical Guide

> **Purpose of this Document:**  
> This file is the single, authoritative reference for the anchor-scarce network localisation codebase. Whenever starting a new session or handing files to an AI assistant / collaborator, provide this document alongside the Colab notebook or code files. It defines **what** the system does, **why** every mathematical and algorithmic choice was made, and the strict rules governing the project for **IEEE Signal Processing Letters (SPL)** submission.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Physical Problem
Given an ad-hoc sensor network in 2D Euclidean space ($\mathbb{R}^2$) with $N$ nodes (default $N=50$, generalising to $N \in [30, 200]$) in $[-10, 10]^2$:
- **Anchor Scarcity:** Only $K = 3$ anchor nodes have known coordinates—the absolute mathematical minimum required to fix translation, rotation, and reflection in $\mathbb{R}^2$ ($d + 1$ anchors).
- **Sparse Connectivity:** $10\% \text{ to } 90\%$ of pairwise distance measurements are missing ($p_{\text{missing}} \in [0.1, 0.9]$), with the primary performance bottleneck at $60\% - 90\%$.
- **Measurement Impairments:**
  $$\tilde{d}_{ij} = d_{ij} \cdot (1 + \sigma \cdot \epsilon_{ij}) + b_{ij} \cdot \eta_{ij}$$
  where $\epsilon_{ij} \sim \mathcal{N}(0, 1)$ is multiplicative Gaussian noise ($\sigma = 0.05$), $b_{ij} \sim \text{Bernoulli}(p_{\text{nlos}})$ indicates Non-Line-of-Sight (NLOS) conditions ($p_{\text{nlos}} = 0.2$), and $\eta_{ij} \sim \mathcal{U}[0, \text{nlos\_max}]$ is a strictly positive range bias ($\text{nlos\_max} = 4.0$).
- **Goal:** Reconstruct the coordinates of all $N - K$ unknown sensor nodes $X_{\text{unknown}} \in \mathbb{R}^{(N-K) \times 2}$ given only the sparse, corrupted range measurements and the anchor positions.

### 1.2 Why Existing Approaches Fail Under Minimal Anchors
1. **Classical Multidimensional Scaling (MDS-MAP):** Uses all-pairs shortest paths (Dijkstra) to approximate missing Euclidean distances. At high missing rates ($>60\%$), shortest path hop-distances drastically overestimate true Euclidean distances, creating severe geometric distortion.
2. **Semidefinite Programming (SDP / Biswas-Ye):** Relaxes the non-convex distance equality constraints to a convex semidefinite cone. However:
   - Computational complexity is $\mathcal{O}(N^3) \text{ to } \mathcal{O}(N^{3.5})$, making it impractical for large networks.
   - Positive NLOS biases severely stretch the SDP relaxation, pulling interior nodes toward the convex hull of the anchors.
3. **Pure Black-Box GNNs (e.g., Yan et al., GraphSAGE, MPNN-Loc):**
   - Directly regressing node coordinates lacks physical distance constraints.
   - When coordinates undergo global rotations, black-box networks fail unless trained on massive data augmentations.
   - They were formulated primarily for anchor-rich networks and frequently fail to converge on sparse, irregularly deformed graphs with only $K=3$ anchors.

---

## 2. Proposed Solution: AnchorGNN + Unrolled Learned SMACOF

Our method combines the representation learning power of Graph Neural Networks with the physical inductive bias of Majorization-Minimization (SMACOF / distance-weighted MDS).

```
Sparse Graph + Anchors
        │
        ▼
[ Canonical Geometric Anchor Frame ] ──> Guarantees exact E(2)-equivariance
        │
        ▼
[ AnchorGNN Encoder (5 MPNN Layers) ] ──> Injects observed distances into every message
        │
        ▼
   Coarse x_A (Stage-A Prediction)
        │
        ▼
[ Unrolled Learned-SMACOF (T Steps) ] ──> Majorization-Minimization deep unrolling
   ├── Learned Huber Gate (w_ij = w_Huber * Gate, exact 1.0 at init)
   ├── Learned Damped Step Size (alpha in [0.1, 0.7])
   └── Numerical Divergence Guards
        │
        ▼
   Refined x_T
        │
        ▼
[ Huber Stress Post-Polish (L-BFGS) ] ──> Final fairness polish (shared across all methods)
```

---

## 3. Deep Dive: Key Methodological Choices & "Why"

### 3.1 Canonical Geometric Anchor Frame ($E(2)$-Equivariance)
- **What it does:** Transforms arbitrary raw coordinates into an anchor-centered canonical coordinate system:
  1. **Translation:** Origin is set to the anchor triangle centroid $c = \frac{1}{3}\sum_{a=1}^3 x_a$.
  2. **Scale:** Unit distance is normalized by the mean pairwise anchor distance $s = \frac{1}{3}\sum_{i<j} \|x_a^{(i)} - x_a^{(j)}\|$.
  3. **Orientation:** Aligns the baseline vector with the positive x-axis and reflects so the 3rd anchor lies in the upper half-plane ($y \ge 0$).
- **The Critical Design Choice (Triangle Geometry Ordering):**  
  *Why:* Ordering anchors by polar angle ($\text{arctan2}$) fails because measuring angle from the global x-axis breaks under global rotation and reflection. Ordering by raw node index breaks under node relabeling.  
  *Solution:* We order reference anchors by intrinsic triangle geometry:
  $$\text{Vertex opposite the longest side first} \longrightarrow \text{Vertex opposite the next-longest side second}$$
  This is proven invariant under all 2D Euclidean transformations ($E(2) = O(2) \ltimes \mathbb{R}^2$) and arbitrary node indexing.

### 3.2 Handling $K \ge 3$ Anchors (Zero-Shot Generalization)
- The model is trained strictly with $K=3$ anchors ($N_{\text{ANCHORS}}=3$).
- The node feature matrix $X \in \mathbb{R}^{N \times 19}$ explicitly takes geodesic distances and hop counts to the 3 reference triangle anchors defining the canonical frame.
- When evaluating on $K = 4..8$ anchors in the extended sensitivity sweep:
  - The 3 best-spread anchors (largest triangle area) define the canonical $E(2)$ feature frame.
  - All $K$ anchors are fully exploited in the closed-form least-squares multilateration prior $x_0 \in \mathbb{R}^{N \times 2}$, the anchor indicator flag `is_a`, and the anchor coordinates `own`.
  - In `LearnedSMACOFStep`, all $K$ anchors are strictly pinned after every iteration.
- *Scientific framing:* This represents zero-shot transfer across anchor density—the network maintains a constant 19-D input while naturally ingesting additional anchor constraints.

### 3.3 AnchorGNN Encoder (Edge-Aware Message Passing)
- **What it does:** Computes initial node embeddings $h_i$ and edge embeddings $e_{ij}$ through 5 MPNN layers without requiring external libraries like PyG:
  $$m_{ij} = \text{MLP}_m([h_i, h_j, e_{ij}]), \quad e_{ij} \leftarrow \text{LayerNorm}(e_{ij} + m_{ij}), \quad h_i \leftarrow \text{LayerNorm}(h_i + \text{MLP}_u([h_i, \bar{m}_i]))$$
- **Why edge-aware:** Unlike standard GCN or GraphSAGE (which only pass node states), the measured physical distance $d_{ij}$ and distance inconsistency $|d_{ij} - \|x_{0,i} - x_{0,j}\||$ enter **every** message vector, immediately alerting the network to edge stretch.

### 3.4 Unrolled Learned-SMACOF (dwMDS) Refiner
- **Majorization-Minimization Formulation:**  
  Minimizing weighted stress $\sigma(X) = \sum_{i<j} w_{ij} (\|x_i - x_j\| - d_{ij})^2$ via the Guttman transform yields the stationary point iteration:
  $$x_i^{(t+1)} = \frac{\sum_{j} w_{ij} \left( x_j^{(t)} + d_{ij} \frac{x_i^{(t)} - x_j^{(t)}}{\|x_i^{(t)} - x_j^{(t)}\|} \right)}{\sum_j w_{ij}}$$
- **Learned Neural Enhancements:**
  1. **Exact Huber Initialization:**
     $$w_{ij} = \text{clamp}\left(w_{ij}^{\text{Huber}} \cdot \text{Gate}_{\theta}(h_i, h_j, e_{ij}, d_{ij}, \hat{d}_{ij}, |r_{ij}|), \text{min}=10^{-3}, \text{max}=1.0\right)$$
     The linear readout layer `w_out` has its weights and bias zero-initialized (`nn.init.zeros_`). Therefore, at epoch 0, $z=0 \implies \text{Gate} \equiv 1.0$ everywhere. The model starts **identically as classical robust Huber-IRLS** and learns residual gating on top of it.
  2. **Learned Damped Step Size $\alpha$:**
     $$x_i^{(t+1)} = x_i^{(t)} + \alpha \cdot (\bar{x}_i^{(t+1)} - x_i^{(t)}), \quad \alpha = \text{clamp}(\text{sigmoid}(\alpha_{\text{logit}}), 0.1, 0.7)$$
     Prevents oscillatory overshoot in sparse graphs.
  3. **Numerical Divergence Guards:**
     - Denominator flooring: $\hat{d}_{ij} \ge 10^{-4}$ (prevents division by zero).
     - Step displacement clamping: $\Delta x \in [-5.0, 5.0]$ (prevents gradient explosion).
     - Weight lower bound: $w_{ij} \ge 10^{-3}$ (prevents disconnected components).
  4. **Strict Anchor Pinning:**
     $$x_{\text{anchors}}^{(t)} \equiv x_{\text{anchors}}^{\text{true}}$$
- *Note on Theory:* This unrolled architecture is a deep unrolled optimization heuristic. Because edge weights are dynamically modulated by neural states and updates are damped and clamped, it does not possess monotonic descent convergence guarantees.

### 3.5 Rigidity-Certified Loss Masking
- In sparse networks, uncertified nodes suffer from unidentifiable flexes or flip ambiguities.
- We test combinatorial trilateration and generic global rigidity (Gortler-Healy-Thurston randomized rank test on the stress matrix $\Omega$).
- **The Training Loss Rule:**
  $$\mathcal{L} = \text{HuberLoss}\left( X_{\text{pred}}[\text{certified}], X_{\text{true}}[\text{certified}] \right) + 0.3 \cdot \mathcal{L}_{\text{aux}}(X_{\text{stageA}}[\text{certified}])$$
  Restricting training loss strictly to certified nodes stabilizes training and eliminates contradictory gradients from topologically ambiguous subgraphs.

---

## 4. Experimental Rigor for IEEE Signal Processing Letters

### 4.1 Theoretical Cramér-Rao Lower Bound (CRLB) Benchmark
- Evaluated strictly on Gaussian noise scenarios (`gauss`).
- Under multiplicative range noise $\tilde{d}_{ij} \sim \mathcal{N}(d_{ij}, \sigma^2 d_{ij}^2)$, the variance depends on position. By the Slepian-Bangs formula (differentiating both mean and covariance), the exact Fisher Information Matrix (FIM) weight is:
  $$w_{ij} = \frac{1 + 2\sigma^2}{\sigma^2 d_{ij}^2}$$
  At $\sigma = 0.05$, $2\sigma^2 = 0.005$ ($\approx 0.25\%$ adjustment in RMSE bound).
- The fundamental lower bound over $N-K$ unknown nodes is:
  $$\text{CRLB} = \sqrt{\frac{1}{N-K} \text{Tr}(J^{-1})}$$
- *Reviewer disclosure:* CRLB bounds unbiased estimators. Supervised neural models with spatial priors can achieve lower RMSE than the unbiased bound under severe noise or sparsity.

### 4.2 Non-Parametric Paired Wilcoxon Signed-Rank Hypothesis Tests
- Implemented natively in [`step3_evaluate.py`](file:///c:/Users/priya/OneDrive/Desktop/Paper%20ka%20new%20code/gnn_loc/step3_evaluate.py) (`run_paired_wilcoxon_tests`).
- Evaluates per-graph paired performance on globally rigid graphs at $60\% - 85\%$ missing distance constraints.
- Reports two-sided $p$-values, win/loss/tie counts, mean RMSE, and failure rate ($\text{RMSE} > 1.0$) into `results/paired_tests.csv`.

### 4.3 NLOS Edge Detection ROC-AUC
- Evaluates whether learned weights $w_{ij}$ actively identify corrupted NLOS edges ($b_{ij} > 0$).
- Uses tie-safe average ranks (`scipy.stats.rankdata`) so uniform weights properly return the neutral baseline $\text{AUC} = 0.5$.

### 4.4 Reviewer Fairness Controls
1. **Multi-Seed Protocol:** 3 independent seeds ($0, 1, 2$) generate separate training runs, but evaluate on **shared, identical test sets**.
2. **Equal Polish:** Huber stress L-BFGS polish is applied equally to our model and baselines (`SDP + Huber polish`, `Random + Huber polish`, `DV-distance + Huber polish`).
3. **Multi-Start Objective Fairness:** The winning initialization among random starts is selected using the exact robust Huber stress objective being minimized.
4. **SDP Accounting & Timing:** 
   - SDP fallbacks (CVXPY/SCS failure $\rightarrow$ MDS-MAP) are counted and reported.
   - The SDP solve duration is added to `SDP + Huber polish` runtime so it accurately reflects total execution time.
5. **Fair Single-Threaded CPU Runtime Benchmark:** 
   Measures end-to-end CPU time including feature building (Dijkstra shortest paths + multilateration initialization) + neural inference + polish vs classical baseline solvers.

---

## 5. Repository Structure & File Map

```
c:\Users\priya\OneDrive\Desktop\Paper ka new code\
├── PROJECT_OVERVIEW.md         <-- This comprehensive guide
├── gnn_loc/
│   ├── PROJECT_OVERVIEW.md     <-- Synchronized copy
│   ├── config.py               <-- Central source of truth for all hyperparameters
│   ├── common.py               <-- Geometry, E(2) frame, graph generators, rigidity, Slepian-Bangs CRLB
│   ├── model.py                <-- AnchorGNN encoder, unrolled SMACOF step, zero-init Huber gate
│   ├── baselines.py            <-- Classical & learned baselines (SDP, MDS-MAP, etc.)
│   ├── step1_data.py           <-- Builds train, val, test, transfer, and sweep datasets
│   ├── step2_train.py          <-- Trains proposed model & ablations (.done marker guards)
│   ├── step3_evaluate.py       <-- Multi-seed evaluation, Wilcoxon tests, caching, publication figures
│   ├── colab_all_in_one.ipynb  <-- Self-contained base Colab notebook
│   └── colab_all_in_one_extended.ipynb <-- Extended notebook (anchor K & NLOS sweeps)
```

---

## 6. Evaluated Methods & Baseline Taxonomy

| Method Name | Category | Description |
|---|---|---|
| **Ours (Proposed)** | Deep Unrolled | AnchorGNN + 50 learned-SMACOF steps |
| **Ours + Huber polish** | Proposed + Polish | Proposed model refined by local Huber stress L-BFGS |
| **Ablation: uniform weights** | Unrolled Ablation | SMACOF with $w_{ij}=1$ (classical dwMDS unrolled) |
| **Ablation: Huber weights** | Unrolled Ablation | Classical iteratively reweighted least squares (IRLS) |
| **Ablation: Huber + learned $\alpha$**| Unrolled Ablation | Huber weights with learned step size $\alpha$ |
| **Ablation: encoder only** | GNN Ablation | Stage-A AnchorGNN predictions without SMACOF ($T=0$) |
| **SDP (Biswas-Ye)** | Convex Relaxation | Semidefinite programming relaxation via CVXPY/SCS |
| **SDP + Huber polish** | Convex + Polish | SDP solution refined by local Huber stress |
| **MDS-MAP** | Classical Algebraic | Geodesic completion + classical MDS + Procrustes |
| **MDS-MAP + Huber stress** | Classical Optimization | MDS-MAP refined by robust Huber stress |
| **DV-distance + Huber polish**| Heuristic + Polish | Multilateration from shortest paths + Huber polish |
| **Yan et al. (ICASSP 2021)**| Supervised GNN | Per-graph GCN using 3 anchors as supervisory labels |
| **MPNN-Loc (Xu & Lin 2025)** | Semi-Supervised GNN | Anchor MSE loss + Laplacian correlation refinement |
| **GraphSAGE (per-graph)** | Unsupervised GNN | GraphSAGE trained on stress minimization per graph |
| **OptSpace Spectral EDM** | Matrix Completion | Low-rank EDM completion (EDM Book Alg 3) without polish |
| **s-stress L-BFGS (Alg 4)** | Matrix Optimization| Alternating descent on s-stress (EDM Book Alg 4) without polish |
| **Random (multi-start) + polish**| Local Search | 5 random coordinate initializations polished by Huber |

---

## 7. Golden Invariants: Rules for Future Modifications

When editing or extending this codebase, adhere strictly to these rules:

1. **Zero Ground-Truth Leakage:**  
   Ground-truth coordinates `g["true_xy"]` must **never** be accessed by any baseline, feature builder, or model forward pass. It is strictly reserved for:
   - Supervised loss during training on synthetic graphs.
   - Final evaluation scoring (`coord_rmse`, `rel_edm_error`).
2. **Anchors Must Remain Pinned:**  
   Anchor coordinates are fixed and known. In all refiners, baselines, and neural layers, anchor positions must be pinned to their true values after every iteration.
3. **Multi-Seed Isolation with Shared Evaluation:**  
   Models are trained per seed under `checkpoints/seed{s}/` and evaluated under `results/seed{s}/`. However, `data/test.pkl` must remain **shared and identical** across all seeds to preserve paired statistical validity.
4. **Preserve the `.done` Marker Pattern:**  
   Never check for `.pt` checkpoint existence when skipping completed models. Always check for `.done` files to prevent evaluating half-trained models resulting from Colab disconnections.
5. **Frame Invariance Integrity:**  
   Never alter anchor ordering to use coordinate-axis-dependent angles. Keep the intrinsic triangle geometry ordering (longest side $\rightarrow$ next-longest side).
6. **Tie-Safe Statistical Ranking:**  
   Always use `scipy.stats.rankdata` with average ties for rank-based metrics (AUC).

---
*Document Version: 2.0 (IEEE SPL Verified & Mathematically Audited)*  
*Maintained for pair-programming and seamless transfer to future development sessions.*
