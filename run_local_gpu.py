"""
run_local_gpu.py -- Master 1-Click Runner for Local GPU Workstations.
Executes the pre-registered IEEE Signal Processing Letters benchmark pipeline:
  1. System & GPU audit
  2. Dataset verification
  3. Pre-registered multi-seed model training on FRESH seeds (Seeds 3, 4, 5, 6, 7)
  4. Fair multi-seed ablation training (Seeds 3, 4, 5) for uniform, huber, encoder_only
  5. Multi-seed evaluation on identical test sets with fair CPU timing & sweeps
  6. Decision rule evaluation & Wilcoxon hypothesis tests
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
EPOCHS_TO_TRAIN   = 100
SEEDS             = [3, 4, 5, 6, 7]             # Strictly fresh evaluation seeds (0, 1, 2 quarantined)
ABLATION_SEEDS    = [3, 4, 5]                   # 3 seeds for fair ablation comparison
ABLATION_VARIANTS = ["uniform", "huber", "encoder_only"]
TEST_GRAPHS       = 20                          # 20 per cell * 64 cells = 1,280 test graphs
N_TRAIN           = 10000                       # 10,000 synthetic training graphs
N_VAL             = 1000                        # 1,000 validation graphs
# -----------------------------------------------------------

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["N_TRAIN"] = str(N_TRAIN)
os.environ["N_VAL"] = str(N_VAL)
os.environ["EPOCHS"] = str(EPOCHS_TO_TRAIN)
os.environ["TEST_PER_LEVEL"] = str(TEST_GRAPHS)
os.environ["BA_PER_M"] = str(TEST_GRAPHS)
os.environ["RUN_SDP"] = "1"
os.environ["RUN_CPU_TIMING"] = "1"              # Fair single-threaded CPU runtime benchmark
os.environ["RUN_EXTENDED_EXPERIMENTS"] = "1"     # Anchor sweeps (K=3..8) and NLOS sweeps
os.environ["RUN_TRANSFER"] = "0"

def log_header(title):
    print("\n" + "=" * 75, flush=True)
    print(f"   {title}", flush=True)
    print("=" * 75, flush=True)

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
    print("[!] Warning: No CUDA GPU detected. Code will run on CPU.")

# ==================== 2. DATASET VERIFICATION ====================
log_header("PHASE 2/5: STEP 1 - DATASET VERIFICATION")
os.makedirs("data", exist_ok=True)
need_data = not (os.path.exists("data/train.pkl") and os.path.exists("data/test_fresh.pkl"))
if need_data:
    sh("python3 -u step1_data.py")
else:
    print("[*] Datasets (train.pkl, test_fresh.pkl) already exist. Skipping Step 1.", flush=True)

# ==================== 3. MULTI-SEED MODEL & ABLATION TRAINING ====================
log_header("PHASE 3/5: STEP 2 - MULTI-SEED TRAINING (FRESH SEEDS)")
for s in SEEDS:
    tag_dir = f"seed{s}"
    print(f"\n>>> Training Proposed Learned Model for {tag_dir} (Epochs: {EPOCHS_TO_TRAIN}) <<<", flush=True)
    sh(f"python3 -u step2_train.py --seed {s} --variant learned", RUN_TAG=tag_dir)
    
    if s in ABLATION_SEEDS:
        for var in ABLATION_VARIANTS:
            print(f"\n>>> Training Ablation '{var}' for {tag_dir} (Epochs: {EPOCHS_TO_TRAIN}) <<<", flush=True)
            sh(f"python3 -u step2_train.py --seed {s} --variant {var}", RUN_TAG=tag_dir)

# ==================== 4. MULTI-SEED BENCHMARK EVALUATION ====================
log_header("PHASE 4/5: STEP 3 - MULTI-SEED BENCHMARK EVALUATION")
for s in SEEDS:
    tag_dir = f"seed{s}"
    print(f"\n>>> Starting Benchmark Evaluation on Identical Test Set: {tag_dir} <<<", flush=True)
    sh("python3 -u step3_evaluate.py", RUN_TAG=tag_dir)

# ==================== 5. STATISTICAL AGGREGATION & DECISION RULE EVALUATION ====================
log_header("PHASE 5/5: STATISTICAL AGGREGATION & DECISION RULE EVALUATION")
seed_csvs = [f"results/seed{s}/all_results.csv" for s in SEEDS if os.path.exists(f"results/seed{s}/all_results.csv")]
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
            comp_list = ["SDP + Huber polish", "MDS-MAP + Huber stress", "DV-distance + Huber polish"]
            if m_ours == "Ours: AnchorGNN + learned SMACOF":
                comp_list.append("Ablation: encoder + Huber-weight steps")
            for comp in comp_list:
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

    # Pre-Registered Decision Rule Verification (Primary Slice: Globally Rigid RGG NLOS, missing <= 70%)
    print("\n" + "=" * 75)
    print("   PRE-REGISTERED DECISION RULE EVALUATION (FRESH SEEDS 3, 4, 5, 6, 7)")
    print("=" * 75)
    
    # Primary slice definition
    d_rgg = df_all[(df_all["scenario"] == "rgg/nlos") & (df_all["missing"] <= 0.70)]
    if "globally_rigid" in d_rgg.columns:
        d_rgg = d_rgg[d_rgg["globally_rigid"] == True]

    m_ours = "Ours: AnchorGNN + learned SMACOF"
    m_huber = "Ablation: encoder + Huber-weight steps"
    m_mds = "MDS-MAP + Huber stress"
    m_sdp = "SDP + Huber polish"

    d_ours = d_rgg[d_rgg["method"] == m_ours]
    seed_means = d_ours.groupby("seed")["rmse_certified"].mean()

    # Condition 1: Absolute Performance Threshold (<= 0.35m on >= 4 of 5 seeds)
    n_passed_c1 = (seed_means <= 0.35).sum()
    c1_pass = (n_passed_c1 >= 4)
    print(f"\n[Condition 1] Absolute Performance (<= 0.35m certified RMSE on RGG NLOS rigid <=70%):")
    for s_name, val in seed_means.items():
        print(f"  {s_name}: {val:.4f} m ({'PASS' if val <= 0.35 else 'FAIL'})")
    print(f"  -> Result: {'PASS' if c1_pass else 'FAIL'} ({n_passed_c1}/{len(seed_means)} seeds passed)")

    # Condition 2: Core Hypothesis (Learned Gate > Huber refiner on ablation seeds 3, 4, 5)
    ablation_seeds_eval = [f"seed{s}" for s in ABLATION_SEEDS if f"seed{s}" in d_rgg["seed"].unique()]
    c2_passes = 0
    print(f"\n[Condition 2] Core Hypothesis (Learned Gate vs Parameter-Free Huber on Seeds 3, 4, 5):")
    for s_tag in ablation_seeds_eval:
        d_sub = d_rgg[d_rgg["seed"] == s_tag]
        piv = d_sub.pivot_table(index="graph_id", columns="method", values="rmse_certified").dropna()
        if m_ours in piv.columns and m_huber in piv.columns:
            diff = piv[m_ours] - piv[m_huber]
            win_rate = (diff < -1e-4).mean()
            med_diff = diff.median()
            try:
                pv = 1.0 if (diff.abs() <= 1e-4).all() else wilcoxon(diff[diff.abs() > 1e-4]).pvalue
            except Exception:
                pv = np.nan
            seed_p = (pv < 0.05) and (win_rate > 0.50) and (med_diff < 0)
            if seed_p:
                c2_passes += 1
            print(f"  {s_tag}: win_rate={win_rate*100:.1f}%, med_diff={med_diff:.4f}m, p={pv:.4e} -> {'PASS' if seed_p else 'FAIL'}")
        else:
            print(f"  {s_tag}: ablation data missing for {m_huber}")
    c2_pass = (c2_passes >= max(1, len(ablation_seeds_eval) - 1)) if ablation_seeds_eval else False
    print(f"  -> Result: {'PASS' if c2_pass else 'FAIL'} ({c2_passes}/{len(ablation_seeds_eval)} ablation seeds passed)")

    # Condition 3: Per-Seed Superiority over Classical Baselines (MDS-MAP+Huber and SDP+Huber on >= 4 seeds)
    c3_passes = 0
    print(f"\n[Condition 3] Per-Seed Statistical Superiority vs Classical Baselines (MDS-MAP & SDP):")
    for s_tag in seed_means.index:
        d_sub = d_rgg[d_rgg["seed"] == s_tag]
        piv = d_sub.pivot_table(index="graph_id", columns="method", values="rmse_certified").dropna()
        pass_mds, pass_sdp = False, False
        if m_ours in piv.columns and m_mds in piv.columns:
            diff = piv[m_ours] - piv[m_mds]
            pv = 1.0 if (diff.abs() <= 1e-4).all() else wilcoxon(diff[diff.abs() > 1e-4]).pvalue
            pass_mds = (pv < 0.05) and ((diff < -1e-4).mean() > 0.50) and (diff.median() < 0)
        if m_ours in piv.columns and m_sdp in piv.columns:
            diff = piv[m_ours] - piv[m_sdp]
            pv = 1.0 if (diff.abs() <= 1e-4).all() else wilcoxon(diff[diff.abs() > 1e-4]).pvalue
            pass_sdp = (pv < 0.05) and ((diff < -1e-4).mean() > 0.50) and (diff.median() < 0)
        seed_p = pass_mds and pass_sdp
        if seed_p:
            c3_passes += 1
        print(f"  {s_tag}: vs MDS-MAP={'PASS' if pass_mds else 'FAIL'}, vs SDP={'PASS' if pass_sdp else 'FAIL'} -> {'PASS' if seed_p else 'FAIL'}")
    c3_pass = (c3_passes >= 4)
    print(f"  -> Result: {'PASS' if c3_pass else 'FAIL'} ({c3_passes}/{len(seed_means)} seeds passed)")

    # Condition 4: Minimum 10% Practical Effect Size Reduction over MDS-MAP+Huber
    mean_ours = d_ours["rmse_certified"].mean()
    d_mds = d_rgg[d_rgg["method"] == m_mds]
    mean_mds = d_mds["rmse_certified"].mean() if len(d_mds) > 0 else 0.0
    pct_reduction = (mean_mds - mean_ours) / mean_mds * 100.0 if mean_mds > 0 else 0.0
    c4_pass = (pct_reduction >= 10.0)
    print(f"\n[Condition 4] Minimum 10% Effect Size Reduction vs MDS-MAP+Huber:")
    print(f"  Ours Mean: {mean_ours:.4f} m | MDS-MAP Mean: {mean_mds:.4f} m | Error Reduction: {pct_reduction:.2f}%")
    print(f"  -> Result: {'PASS' if c4_pass else 'FAIL'}")

    # Overall Formal Pre-Registered Decision
    all_passed = c1_pass and c2_pass and c3_pass and c4_pass
    print("\n" + "-" * 75)
    if all_passed:
        decision_txt = "DECISION: OPTION B CONFIRMED (All 4 pre-registered conditions satisfied)."
        print(f"[+] {decision_txt}")
        print("    -> Proceed to write paper based on AnchorGNN + Learned Gated SMACOF as headline!")
    else:
        failed_conds = []
        if not c1_pass: failed_conds.append("Condition 1 (Threshold <= 0.35m)")
        if not c2_pass: failed_conds.append("Condition 2 (Gate > Huber Ablation)")
        if not c3_pass: failed_conds.append("Condition 3 (Per-Seed vs Baselines)")
        if not c4_pass: failed_conds.append("Condition 4 (>=10% Effect Size)")
        decision_txt = f"DECISION: FALLBACK TO OPTION A ({', '.join(failed_conds)} failed)."
        print(f"[!] {decision_txt}")
        print("    -> Proceed to write paper based on AnchorGNN as Fast Initializer for Robust SMACOF!")
    print("-" * 75)

    with open("results/decision_rule_evaluation.txt", "w") as f:
        f.write(decision_txt + "\n\n")
        f.write(f"Condition 1 (<=0.35m on >=4/5 seeds): {'PASS' if c1_pass else 'FAIL'} ({n_passed_c1}/{len(seed_means)})\n")
        f.write(f"Condition 2 (Gate > Huber on seeds 3-5): {'PASS' if c2_pass else 'FAIL'} ({c2_passes}/{len(ablation_seeds_eval)})\n")
        f.write(f"Condition 3 (vs MDS-MAP & SDP on >=4/5 seeds): {'PASS' if c3_pass else 'FAIL'} ({c3_passes}/{len(seed_means)})\n")
        f.write(f"Condition 4 (>=10% reduction vs MDS-MAP): {'PASS' if c4_pass else 'FAIL'} ({pct_reduction:.2f}% reduction)\n\n")
        f.write("Certified RMSE on RGG NLOS (rigid <=70%) per seed:\n")
        f.write(seed_means.to_string() + "\n")

log_header("SUCCESS! ALL PIPELINE PHASES COMPLETE")
print("All model weights are stored in ./checkpoints/")
print("All final publication CSVs and figures are in ./results/")
