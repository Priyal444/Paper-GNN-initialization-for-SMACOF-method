"""
run_local_gpu.py -- Master 1-Click Runner for Local GPU Workstations.
Executes the full IEEE Signal Processing Letters benchmark pipeline in one go:
  1. System & GPU check
  2. Step 1: Dataset generation (10,000 train, 1,000 val, 1,280 test graphs)
  3. Step 2: Multi-seed training (Seeds 0, 1, 2) with 100 epochs for full gate convergence
  4. Step 3: Multi-seed evaluation on identical test sets
  5. Multi-seed statistical aggregation & paired Wilcoxon hypothesis tests
"""
import os
import sys
import time
import glob
import subprocess
import pandas as pd
import numpy as np

# Ensure working directory is gnn_loc
script_dir = os.path.dirname(os.path.abspath(__file__))
gnn_loc_dir = os.path.join(script_dir, "gnn_loc") if os.path.exists(os.path.join(script_dir, "gnn_loc")) else script_dir
os.chdir(gnn_loc_dir)
print(f"[*] Working Directory: {os.getcwd()}", flush=True)

# ----------------- CONFIGURATION SETTINGS -----------------
EPOCHS_TO_TRAIN  = 100        # 100 epochs gives full convergence for the learned gate (as in Run 8)
SEEDS            = [0, 1, 2]  # Strict IEEE SPL multi-seed statistical rigor
VARIANTS         = ["learned", "huber", "huber_alpha", "uniform", "encoder_only"]
TEST_GRAPHS      = 20         # 20 per cell * 64 cells = 1,280 test graphs
N_TRAIN          = 10000      # 10,000 synthetic training graphs
N_VAL            = 1000       # 1,000 validation graphs
# -----------------------------------------------------------

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["N_TRAIN"] = str(N_TRAIN)
os.environ["N_VAL"] = str(N_VAL)
os.environ["EPOCHS"] = str(EPOCHS_TO_TRAIN)
os.environ["TEST_PER_LEVEL"] = str(TEST_GRAPHS)
os.environ["BA_PER_M"] = str(TEST_GRAPHS)
os.environ["RUN_SDP"] = "1"
os.environ["RUN_TRANSFER"] = "0"

def log_header(title):
    print("\n" + "=" * 70, flush=True)
    print(f"   {title}", flush=True)
    print("=" * 70, flush=True)

def sh(cmd, **env):
    print(f"\n$ {cmd}", flush=True)
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {p.returncode}): {cmd}")

# ==================== 1. HARDWARE & GPU AUDIT ====================
log_header("PHASE 1/5: GPU & ENVIRONMENT AUDIT")
import torch
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA Enabled: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"Detected {n_gpu} CUDA GPU(s):")
    for i in range(n_gpu):
        prop = torch.cuda.get_device_properties(i)
        vram = prop.total_memory / (1024 ** 3)
        print(f"  [GPU {i}] {prop.name} | VRAM: {vram:.1f} GB | Compute: {prop.major}.{prop.minor}")
else:
    print("[!] Warning: No CUDA GPU detected. Code will run on CPU (slower).")

# ==================== 2. DATASET GENERATION ====================
log_header("PHASE 2/5: STEP 1 - GENERATING DATASETS")
os.makedirs("data", exist_ok=True)
need_data = not (os.path.exists("data/train.pkl") and os.path.exists("data/test.pkl"))
if need_data:
    sh("python -u step1_data.py")
else:
    print("[*] Datasets (train.pkl, test.pkl) already exist. Skipping Step 1.", flush=True)

# ==================== 3. MULTI-SEED TRAINING ====================
log_header("PHASE 3/5: STEP 2 - MULTI-SEED MODEL TRAINING")
for s in SEEDS:
    tag_dir = f"seed{s}"
    print(f"\n>>> Starting Training Loop for {tag_dir} (Epochs: {EPOCHS_TO_TRAIN}) <<<", flush=True)
    sh(f"python -u step2_train.py --seed {s}", RUN_TAG=tag_dir)

# ==================== 4. MULTI-SEED EVALUATION ====================
log_header("PHASE 4/5: STEP 3 - MULTI-SEED BENCHMARK EVALUATION")
for s in SEEDS:
    tag_dir = f"seed{s}"
    print(f"\n>>> Starting Benchmark Evaluation on Identical Test Set: {tag_dir} <<<", flush=True)
    sh("python -u step3_evaluate.py", RUN_TAG=tag_dir)

# ==================== 5. STATISTICAL AGGREGATION & TESTS ====================
log_header("PHASE 5/5: MULTI-SEED AGGREGATION & WILCOXON HYPOTHESIS TESTS")
seed_csvs = sorted(glob.glob("results/seed*/all_results.csv"))
if seed_csvs:
    df_list = []
    for f in seed_csvs:
        s_tag = os.path.basename(os.path.dirname(f))
        df_list.append(pd.read_csv(f).assign(seed=s_tag))
    df_all = pd.concat(df_list, ignore_index=True)
    
    cols = ["rmse", "rmse_certified", "fail_10", "fail_15"]
    avail_cols = [c for c in cols if c in df_all.columns]
    cell = df_all.groupby(["seed", "scenario", "missing", "method"])[avail_cols].mean().reset_index()
    agg = cell.groupby(["scenario", "missing", "method"])[avail_cols].agg(["mean", "std"])
    os.makedirs("results", exist_ok=True)
    agg.to_csv("results/multi_seed_aggregated_metrics.csv")
    print("[*] Saved multi-seed aggregated metrics -> results/multi_seed_aggregated_metrics.csv", flush=True)

    # Paired Wilcoxon Tests
    from scipy.stats import wilcoxon
    test_models = [
        "Ours: AnchorGNN + learned SMACOF",
        "Ablation: encoder + Huber-weight steps",
        "Ours: AnchorGNN + learned SMACOF + Huber polish"
    ]
    rows = []
    for m_ours in test_models:
        for f in seed_csvs:
            s_name = os.path.basename(os.path.dirname(f))
            d = pd.read_csv(f)
            if "globally_rigid" in d.columns:
                d = d[d["globally_rigid"] == True]
            for comp in ["SDP + Huber polish", "MDS-MAP + Huber stress", "DV-distance + Huber polish"]:
                for nlos_flag in [True, False]:
                    sub = d[d["scenario"].str.contains("nlos")] if nlos_flag else d
                    piv = sub.pivot_table(index=["scenario", "missing", "graph_id"], columns="method", values="rmse")
                    if m_ours in piv.columns and comp in piv.columns:
                        pair = piv[[m_ours, comp]].dropna()
                        x, y = pair[m_ours], pair[comp]
                        diff = x - y
                        try:
                            pv = 1.0 if (diff.abs() <= 1e-4).all() else wilcoxon(diff[diff.abs() > 1e-4]).pvalue
                        except Exception:
                            pv = np.nan
                        rows.append(dict(
                            seed=s_name,
                            A=m_ours,
                            B=comp,
                            nlos_only=nlos_flag,
                            n=len(pair),
                            win=(diff < -1e-4).mean(),
                            loss=(diff > 1e-4).mean(),
                            mean_A=x.mean(),
                            mean_B=y.mean(),
                            fail_A=(x > 1.0).mean(),
                            fail_B=(y > 1.0).mean(),
                            p=pv
                        ))
    if rows:
        pt = pd.DataFrame(rows)
        pt.to_csv("results/paired_tests.csv", index=False)
        print("[*] Saved paired Wilcoxon test results -> results/paired_tests.csv", flush=True)
        print("\nSummary of Paired Tests on NLOS Rigid Graphs vs SDP+Huber:")
        summary_pt = pt[(pt["B"] == "SDP + Huber polish") & (pt["nlos_only"] == True)]
        print(summary_pt[["seed", "A", "win", "mean_A", "mean_B", "p"]].to_string())

log_header("SUCCESS! ALL PIPELINE PHASES COMPLETE")
print("All model weights are stored in ./checkpoints/")
print("All final publication CSVs and figures are in ./results/")
