# Formal Pre-Registration Protocol: Cooperative Localization via AnchorGNN and Unrolled SMACOF

**Target Venue:** IEEE Signal Processing Letters (SPL)  
**Authors:** Priyanshu et al.  
**Protocol Status:** Locked Prior to Fresh-Seed Master Execution  
**Git Base Commit (`gnn_loc`):** `42addf2a945a5467bd8f4a5464af6935765bee95`  
**Git Pre-Registration Lock Commit:** `d693157209c34c707eaebdb359815e1070aa1862`  
**Calibration Rule:** All hyperparameter adjustments (LR reduced to `5e-4`, grad clip tightened to `0.5`, step clamp tightened to `1.0`) were chosen strictly on the validation set using pilot diagnostic runs (Seeds 0–2) prior to this lock, and are permanently frozen across all fresh evaluation runs.

---

## 1. Pinned Hyperparameters & Configuration Table
All hyperparameters are permanently frozen as specified below. Any exploratory diagnostic tuning was conducted solely on the validation set using pilot seeds (Seeds 0–2):

| Parameter | Pinned Value | Description / Scope |
| :--- | :--- | :--- |
| **`N_NODES`** | `50` | Number of sensors per test graph (main benchmark setting) |
| **`N_ANCHORS`** | `3` | Theoretical minimum anchors in $\mathbb{R}^2$ ($K = d + 1$) |
| **`COORD_RANGE`** | `10.0` | 2-D deployment field: $[-10, 10]^2\,\text{m}^2$ (area $400\,\text{m}^2$) |
| **`EPOCHS`** | `100` | Full convergence training epochs |
| **`BATCH_GRAPHS`** | `32` | Batch size (graphs per batch) |
| **`LR`** | `5e-4` | AdamW learning rate (stabilized post-warmup) |
| **`WEIGHT_DECAY`** | `1e-5` | AdamW weight decay |
| **`GRAD_CLIP`** | `0.5` | Maximum gradient norm clipping |
| **`STEP_CLAMP`** | `1.0` | Step update clamp in normalized coordinate units $[-1.0, 1.0]$ |
| **`REFINE_STEPS`** | `10` | Unrolled SMACOF majorization steps during training |
| **`TEST_REFINE_STEPS`**| `50` | Unrolled SMACOF majorization steps during inference |
| **`TEST_PER_LEVEL`** | `20` | Test graphs per parametric cell (total $64 \times 20 = 1,280$ graphs) |
| **`HUBER_DELTA`** | `0.05` | Huber threshold in normalized units (anchor scale = 1.0) |
| **`AUX_WEIGHT`** | `0.3` | Deep auxiliary supervision loss weight on Stage-A GNN output |
| **`TEST_FILE`** | `data/test_fresh.pkl` | **Completely held-out fresh test set** |
| **`TEST_SEED`** | `8888` | **Independent random generation seed for fresh test data** |
| **`RUN_CPU_TIMING`**| `True` | Generates single-threaded `fair_runtime_benchmark.csv` |

---

## 2. Fresh Benchmark Data & Quarantined Seeds
- **Fresh Held-Out Test Set:** Evaluated exclusively on `data/test_fresh.pkl`, generated with independent random seed `TEST_SEED = 8888`. All baselines (SDP, MDS-MAP, Huber polish, etc.) will be evaluated directly on this held-out set.
- **Strict Quarantine of Pilot Seeds:** Seeds 0, 1, 2 are permanently quarantined as exploratory/diagnostic seeds. Their test scores will **not** be used to support the primary paper claims.
- **Fresh Evaluation Seeds:** Model training will be executed strictly on **Seeds 3, 4, 5, 6, 7** (5 independent random seeds).
- **Reporting Commitment:** Every seed will be reported in the manuscript tables (Mean $\pm$ Std across all 5 seeds). No seed will be omitted or cherry-picked.

---

## 3. Evaluation Metrics & Slicing
1. **Certified Coordinate RMSE ($m$):** Computed over nodes certified as locally unique via trilateral 3-hop expansion on generically globally rigid topologies ($\operatorname{rank}(\mathbf{\Omega}) = N-3$, 3-connected, redundantly rigid).
2. **All-Node Coordinate RMSE ($m$):** Computed over all $N-K$ non-anchor sensors regardless of certificate.
3. **Primary Slicing:** The generically globally rigid subset of Random Geometric Graphs (RGG) under NLOS noise ($p_{\text{nlos}}=0.2, b_{\max}=4.0\,\text{m}$) across **10% to 70% missing edges** (where ~96%–100% of graphs are globally rigid).
4. **Overall Slicing:** Reported across all 10% to 90% missing edges and all topologies (RGG, Random Spanning, Barabási-Albert).

---

## 4. Formal Decision Rule: Option B vs. Option A

### Criteria for Option B (Learned Edge Gate as Headline):
The learned edge-gated architecture (`variant="learned"`) is declared the winning headline method if and only if **all four** of the following pre-registered conditions are satisfied on the fresh test set (`data/test_fresh.pkl`):
1. **Absolute Performance Threshold:** At least **4 of the 5 fresh seeds (Seeds 3, 4, 5, 6, 7)** achieve a mean certified RMSE of **$\le 0.35\,\text{m}$** on the globally rigid subset of RGG NLOS (10%–70% missing).
2. **Testing the Core Hypothesis (Gate vs. Parameter-Free Huber):** On the matched ablation seeds (Seeds 3, 4, 5), the learned gate statistically outperforms the parameter-free Huber refiner (`Ablation: encoder + Huber-weight steps`) in paired Wilcoxon signed-rank tests ($p < 0.05$, win rate $> 50\%$).
3. **Per-Seed Statistical Superiority over Classical Baselines:** In individual per-seed paired Wilcoxon signed-rank tests, at least **4 of the 5 seeds** statistically beat **both**:
   - `MDS-MAP + Huber Polish`: $p < 0.05$, win rate $> 50\%$, negative median error difference.
   - `SDP + Huber Polish`: $p < 0.05$, win rate $> 50\%$, negative median error difference.
4. **Minimum Practical Effect Size:** The 5-seed mean certified RMSE achieves at least a **$10\%$ reduction in error** compared to `MDS-MAP + Huber Polish` on the globally rigid subset of RGG NLOS (10%–70% missing).

### Fallback to Option A (Learned Initializer for Robust SMACOF):
If any of the above four conditions fails, Option B is **rejected**. We will immediately and cleanly adopt **Option A**:
- **Formal Hypothesis:** An $\mathrm{SE}(2)$-canonical message-passing encoder (`AnchorGNN`) provides a fast, millisecond coordinate initialization that overcomes the local-minima vulnerability of classical iterative majorization, enabling parameter-free Huber-SMACOF to achieve localization accuracy comparable to or better than classical heuristics and convex relaxations at substantially lower runtime.
- **Control Comparison:** `AnchorGNN + Huber polish` vs. `MDS-MAP + Huber polish` (evaluating the impact of the initial coordinate embedding under the identical Huber polish refiner).
- **Runtime Positioning:** Evaluated against `MDS-MAP + Huber polish` and `SDP + Huber polish` using the empirical measurements from `fair_runtime_benchmark.csv`.
- **Ablation Transparency:** The learned gate's failure to consistently improve over parameter-free Huber weights will be documented openly in the ablation section as a valuable negative result regarding unconstrained neural weighting in unrolled Euclidean distance geometry.

---

## 5. Fair Multi-Seed Ablations
To ensure fair comparison, ablation variants are trained with identical hyperparameters (`LR = 5e-4`, `GRAD_CLIP = 0.5`, `EPOCHS = 100`, `BATCH_GRAPHS = 32`) across **3 seeds (Seeds 3, 4, 5)**:
- `uniform`: Unweighted Jacobi SMACOF ($w_{ij}=1$).
- `huber`: Analytical Huber M-estimator weighting.
- `encoder_only`: Stage-A GNN coordinates ($T=0$ refiner steps).

---

## 6. Extended Sweeps & Benchmarks
1. **Single-Threaded CPU Timing:** Measured using `RUN_CPU_TIMING = True` (`fair_runtime_benchmark.csv`), enforcing strict 1-thread execution (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`) with untimed warm-up passes.
2. **Anchor Count Sweep ($K=3..8$):** Evaluated on `data/anchor_sweep_test.pkl`.
3. **NLOS Excess Delay Sweep ($b_{\max} \in [1.0, 6.0]\,\text{m}$):** Evaluated on `data/nlos_sweep_test.pkl`.

---

## 7. Related Adjacent Literature Grounding
The manuscript will explicitly cite and distinguish from adjacent localization and unfolding works:
1. **Schwartz & Talmon (IEEE TSP 2019, arXiv:1806.00556):** Intrinsic isometric manifold learning using diffusion maps and MDS. Distinguished by our focus on graph-neural $\mathrm{SE}(2)$-equivariant initialization, minimal $K=3$ anchor gauge resolution, and iterative algorithm unrolling under positive NLOS ranging delays.
2. **Nithya & Prasanthi (Iran. J. Comput. Sci. 2025):** Gates-controlled deep unfolding network (LGDU-BBO) with LSTM-like gate modules and metaheuristic tuning. Distinguished by our exact physical unrolling of the Cauchy-Schwarz Guttman barycenter transform for non-convex stress majorization and 2-D graph rigidity certification ($\operatorname{rank}(\mathbf{\Omega})=N-3$).
