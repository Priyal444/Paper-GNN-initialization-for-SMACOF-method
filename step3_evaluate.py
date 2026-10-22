"""
STEP 3 -- compare the proposed GNN with existing methods on the SAME graphs.

For every test graph, every method gets exactly the same inputs
(edges, noisy distances, 3 anchors) and is scored against the same truth.

Outputs (results/):
    all_results.csv              one row per (graph, method)
    summary_table.csv            mean / std / median per method x scenario x missing
    fig_main_rmse.png            average coordinate RMSE vs missing %  (the main figure)
    fig_certified_rmse.png       same, only on nodes whose position is provably unique
    fig_ablation.png             learned vs uniform vs huber vs encoder-only
    fig_runtime.png              seconds per graph
    fig_generalisation.png       trained on n = 30..80, tested on n = 100 / 200
    fig_ba_rigidity.png          Barabasi-Albert m = 1..5 (rigidity threshold)
    fig_examples.png             true vs predicted positions for a few graphs

    transfer_results.csv         (--transfer) each train-on-some-types model on all test graphs
    transfer_matrix.csv          (--transfer) mean RMSE, rows = training set / method, cols = test type
    fig_transfer_matrix.png      (--transfer) the same as a heatmap

Run:  python step3_evaluate.py
      python step3_evaluate.py --plot-only     # only redraw the figures from results/all_results.csv
      python step3_evaluate.py --transfer      # after the main run and step2 --train-on ...
"""
import glob, os, pickle, time, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from common import coord_rmse, rel_edm_error
from model import AnchorGNNLocalizer, collate, to_original_units
import baselines as BL

warnings.filterwarnings("ignore")
# Our model runs on the GPU if there is one, the existing methods always on CPU.
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
if NUM_GPUS >= 2:
    DEVICES = [f"cuda:{i}" for i in range(NUM_GPUS)]
    DEVICE = "cuda:0"
elif NUM_GPUS == 1:
    DEVICES = ["cuda:0"]
    DEVICE = "cuda:0"
else:
    DEVICES = ["cpu"]
    DEVICE = "cpu"

OURS = "Ours: AnchorGNN + learned SMACOF"
VARIANT_NAMES = {
    "learned": OURS,
    "uniform": "Ablation: encoder + uniform-weight steps",
    "huber": "Ablation: encoder + Huber-weight steps",
    "encoder_only": "Ablation: encoder only (no refiner)",
    "learned_selfsup": "Ours, self-supervised (no ground truth)",
}


def load(name):
    p = os.path.join(C.DATA_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


TRANSFER = "Ours, trained on "          # + "rgg", "rgg+ba", "all", ...


def load_models(transfer=False):
    """Main comparison: every checkpoint except the transfer ones.
    transfer=True: only the learned-variant transfer checkpoints (step2 --train-on)."""
    models = {}
    for path in sorted(glob.glob(os.path.join(C.CKPT_DIR, "*.pt"))):
        tag = os.path.basename(path)[:-3]
        if ("__train-" in tag) != transfer:
            continue
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if transfer and ck["variant"] != "learned":
            continue
        m = AnchorGNNLocalizer(ck["in_dim"], ck["edge_dim"], ck["H"], ck["enc_layers"],
                               ck["refine_steps"], ck["variant"], C.HUBER_DELTA).to(DEVICE)
        m.load_state_dict(ck["state"]); m.eval()
        name = TRANSFER + ck["train_on"] if transfer else VARIANT_NAMES.get(tag, tag)
        models[name] = m
    return models


def run_gnn(model, graphs, bs=64):
    """One forward pass per batch; parallel across available GPUs."""
    import copy, concurrent.futures
    if len(DEVICES) >= 2 and len(graphs) >= 2:
        models_per_dev = [model if dev == DEVICE else copy.deepcopy(model).to(dev).eval()
                          for dev in DEVICES]
        split_indices = np.array_split(np.arange(len(graphs)), len(DEVICES))
        
        def run_dev(m, chk, dev):
            out, t = [], 0.0
            with torch.no_grad():
                for i in range(0, len(chk), bs):
                    b_chk = chk[i:i + bs]
                    B = collate(b_chk, dev)
                    t0 = time.perf_counter(); x, _ = m(B, T=None if m.T == 0 else C.TEST_REFINE_STEPS)
                    if "cuda" in dev: torch.cuda.synchronize(dev)
                    t += time.perf_counter() - t0
                    out += to_original_units(B, x)
            return out, t

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(DEVICES)) as pool:
            futures = [pool.submit(run_dev, models_per_dev[idx], [graphs[k] for k in split_indices[idx]], DEVICES[idx])
                       for idx in range(len(DEVICES))]
            all_out = []
            total_t = 0.0
            for f in futures:
                o, t_dev = f.result()
                all_out.extend(o)
                total_t += t_dev
        return all_out, total_t / len(graphs)
    else:
        out, t = [], 0.0
        with torch.no_grad():
            for i in range(0, len(graphs), bs):
                chunk = graphs[i:i + bs]
                B = collate(chunk, DEVICE)
                t0 = time.perf_counter(); x, _ = model(B, T=None if model.T == 0 else C.TEST_REFINE_STEPS)
                if DEVICE.startswith("cuda"): torch.cuda.synchronize()
                t += time.perf_counter() - t0
                out += to_original_units(B, x)
        return out, t / len(graphs)


def score_rows(g, method, X, secs):
    return dict(scenario=g["scenario"], missing=int(round(100 * g["missing_nominal"])), n=g["n"],
                graph_id=g["graph_id"], method=method, rmse=coord_rmse(g, X),
                rmse_certified=coord_rmse(g, X, g["certified"]), rel_edm=rel_edm_error(g, X),
                seconds=secs, globally_rigid=bool(g["globally_rigid"]),
                frac_certified=float(g["certified"].mean()), ba_m=g.get("ba_m"))


def warmup_cuda(devices):
    for dev in devices:
        if "cuda" in dev:
            try:
                dummy = torch.eye(4, device=dev)
                _ = torch.linalg.eigh(dummy)
                _ = torch.linalg.svd(dummy)
            except Exception:
                pass


def evaluate_set(graphs, models, baselines, polish=True, label=""):
    import concurrent.futures, threading
    rows, preds = [], {}
    n_graphs = len(graphs)
    
    # Warm up cuSOLVER sequentially on all GPUs to prevent lazy wrapper race condition
    warmup_cuda(DEVICES)

    # Pre-build baselines per device for round-robin GPU dispatch
    base_per_dev = {dev: BL.get_baselines(C, device=dev) for dev in DEVICES}
    gpu_methods = {
        "MDS-MAP", "MDS-MAP + stress (L-BFGS)", "MDS-MAP + Huber stress",
        "DV-distance + dwMDS (no learning)", "Classical MDS (direct)",
        "Rank Alternation (EDM book Alg 2)", "OptSpace (EDM book Alg 3)",
        "Alternating Descent (EDM book Alg 4)", "Yan et al. GCN (3 anchor labels)",
        "Per-graph GraphSAGE (previous)"
    }

    # 1. Models
    for name, m in models.items():
        print(f"\n{label} [Model] Running '{name}' on {n_graphs} test graphs ({'Dual-GPU' if len(DEVICES) >= 2 else DEVICE})...", flush=True)
        P, sec = run_gnn(m, graphs)
        preds[name] = P
        rows += [score_rows(g, name, X, sec) for g, X in zip(graphs, P)]
        m_rmse = np.nanmean([coord_rmse(g, X) for g, X in zip(graphs, P)])
        print(f"{label} [Model] '{name}' done -> Mean RMSE: {m_rmse:.4f} ({sec*1000:.1f} ms/graph)", flush=True)

        if polish and (name == OURS or name.startswith(TRANSFER)):
            use_gpu_pol = len(DEVICES) >= 1 and DEVICES[0] != "cpu"
            worker_count = len(DEVICES) if len(DEVICES) >= 2 else (1 if use_gpu_pol else min(os.cpu_count() or 4, 8))
            mode_str = f"Dual-GPU parallel ({', '.join(DEVICES)})" if len(DEVICES) >= 2 else (DEVICE if use_gpu_pol else f"Multi-core CPU ({worker_count} workers)")
            print(f"{label} [Polish] Running Huber L-BFGS polish on {n_graphs} graphs [{mode_str}]...", flush=True)
            t_pol_start = time.time()
            log_step = max(1, n_graphs // 5)
            
            def polish_one(item):
                idx, g, X = item
                dev = DEVICES[idx % len(DEVICES)] if use_gpu_pol else "cpu"
                t0 = time.perf_counter()
                Y = BL.stress_lbfgs(g, X, robust=True, device=dev)
                dur = time.perf_counter() - t0
                return idx, Y, dur

            P2 = [None] * n_graphs
            polish_rows = [None] * n_graphs
            completed = 0
            print_lock = threading.Lock()

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(polish_one, (idx, g, X)) for idx, (g, X) in enumerate(zip(graphs, P))]
                for f in concurrent.futures.as_completed(futures):
                    idx, Y, dur = f.result()
                    P2[idx] = Y
                    polish_rows[idx] = score_rows(graphs[idx], name + " + Huber polish", Y, sec + dur)
                    with print_lock:
                        completed += 1
                        if completed % log_step == 0 or completed == n_graphs:
                            elapsed = time.time() - t_pol_start
                            eta = (elapsed / completed) * (n_graphs - completed)
                            print(f"    progress: {completed}/{n_graphs} polished ({completed*100//n_graphs}%) | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

            rows.extend(polish_rows)
            preds[name + " + Huber polish"] = P2
            pol_rmse = np.nanmean([coord_rmse(g, Y) for g, Y in zip(graphs, P2)])
            print(f"{label} [Polish] Finished in {time.time() - t_pol_start:.1f}s -> Mean RMSE: {pol_rmse:.4f}", flush=True)

    # 2. Baselines
    n_base = len(baselines)
    for b_idx, (name, fn) in enumerate(baselines.items(), 1):
        is_gpu = (name in gpu_methods) and (DEVICES[0] != "cpu")
        if is_gpu and len(DEVICES) >= 2:
            worker_count = len(DEVICES)
            mode_str = f"Dual-GPU parallel ({', '.join(DEVICES)})"
        elif is_gpu:
            worker_count = 1
            mode_str = DEVICE
        else:
            worker_count = min(os.cpu_count() or 4, 8)
            mode_str = f"Multi-core CPU ({worker_count} workers)"

        print(f"\n{label} [{b_idx}/{n_base}] Baseline: '{name}' on {n_graphs} graphs [{mode_str}]...", flush=True)
        t0 = time.time()
        P = [None] * n_graphs
        base_rows = [None] * n_graphs
        log_step = max(1, n_graphs // 5)
        completed = 0
        print_lock = threading.Lock()

        def run_one_baseline(item):
            k, g = item
            dev = DEVICES[k % len(DEVICES)] if is_gpu else DEVICE
            target_fn = base_per_dev[dev].get(name, fn)
            X, sec = BL.timed(target_fn, g)
            return k, X, sec

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(run_one_baseline, (k, g)) for k, g in enumerate(graphs)]
            for f in concurrent.futures.as_completed(futures):
                k, X, sec = f.result()
                P[k] = X
                base_rows[k] = score_rows(graphs[k], name, X, sec)
                with print_lock:
                    completed += 1
                    if completed % log_step == 0 or completed == n_graphs:
                        elapsed = time.time() - t0
                        eta = (elapsed / completed) * (n_graphs - completed)
                        print(f"    progress: {completed}/{n_graphs} graphs ({completed*100//n_graphs}%) | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

        preds[name] = P
        rows.extend(base_rows)
        b_rmse = np.nanmean([coord_rmse(g, X) for g, X in zip(graphs, P)])
        print(f"{label} [{b_idx}/{n_base}] '{name}' completed in {time.time() - t0:.1f}s -> Mean RMSE: {b_rmse:.4f}", flush=True)

    return pd.DataFrame(rows), preds




# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
STYLE = {OURS: dict(color="crimson", lw=2.6, marker="o"),
         OURS + " + Huber polish": dict(color="darkred", lw=2.0, marker="s", ls="--")}


PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#17becf", "#bcbd22", "#393b79"]
FLOOR = 1e-6     # errors below this are numerically exact; drawn on the floor


def style_for(mth, methods):
    if mth in STYLE:
        return STYLE[mth]
    others = [m for m in methods if m not in STYLE]
    return dict(color=PALETTE[others.index(mth) % len(PALETTE)], lw=1.3, marker=".")


def plot_vs_missing(df, metric, methods, fname, title, ylabel):
    """Line = MEAN over the test graphs at each missing level.
    Shaded band = 25th-75th percentile (a +/- std band is meaningless on a log axis)."""
    scen = list(dict.fromkeys(df.scenario))
    cols = 2; rows = int(np.ceil(len(scen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.6 * rows), squeeze=False)
    for ax, sc in zip(axes.ravel(), scen):
        d = df[df.scenario == sc]
        for mth in methods:
            g = d[d.method == mth].groupby("missing")[metric]
            if g.ngroups == 0: continue
            mean, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
            st = style_for(mth, methods)
            ax.plot(mean.index, np.maximum(mean, FLOOR), label=mth, **st)
            ax.fill_between(mean.index, np.maximum(q1, FLOOR), np.maximum(q3, FLOOR),
                            color=st.get("color"), alpha=0.10, lw=0)
        ax.set_yscale("log"); ax.set_ylim(bottom=FLOOR * 0.7); ax.set_title(sc)
        ax.set_xlim(df.missing.min() - 3, df.missing.max() + 3)   # same x range in every panel (BA starts at 50 %)
        ax.set_xlabel("missing distances (%)")
        ax.set_ylabel(f"{ylabel}  (floor = exact)"); ax.grid(alpha=.3, which="both")
    for ax in axes.ravel()[len(scen):]: ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title + "\n(line = mean over test graphs, band = 25-75th percentile)"); plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.savefig(os.path.join(C.RESULTS_DIR, fname), dpi=150); plt.close()


def plot_examples(graphs, preds, methods, fname, levels=(30, 70, 90)):
    pick = []
    for lv in levels:
        for i, g in enumerate(graphs):
            if int(round(100 * g["missing_nominal"])) == lv:
                pick.append(i); break
    if not pick: return
    fig, axes = plt.subplots(len(methods), len(pick), figsize=(4.2 * len(pick), 4 * len(methods)), squeeze=False)
    for c, i in enumerate(pick):
        g = graphs[i]; T = g["true_xy"]
        for r, mth in enumerate(methods):
            ax = axes[r, c]; X = preds[mth][i]
            for a, b in g["edges"]:
                ax.plot(T[[a, b], 0], T[[a, b], 1], c="0.85", lw=0.4, zorder=0)
            ax.scatter(T[:, 0], T[:, 1], s=40, facecolors="none", edgecolors="tab:blue", label="true")
            ax.scatter(X[:, 0], X[:, 1], s=12, c="tab:red", label="predicted")
            for k in range(g["n"]):
                ax.plot([T[k, 0], X[k, 0]], [T[k, 1], X[k, 1]], c="gray", lw=0.5)
            ax.scatter(*g["anchor_xy"].T, marker="*", s=180, c="green", zorder=5, label="anchors")
            ax.set_title(f"{mth[:34]}\n{g['scenario']} {int(100 * g['missing_nominal'])}% missing, "
                         f"RMSE {coord_rmse(g, X):.2f}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, fname), dpi=130); plt.close()


def main_plots(df):
    """All figures that only need all_results.csv (re-drawable with --plot-only)."""
    ablations = [m for m in df.method.unique() if m.startswith("Ablation") or m.startswith("Ours, self")]
    main_methods = [m for m in df.method.unique() if m not in ablations]
    plot_vs_missing(df, "rmse", main_methods, "fig_main_rmse.png",
                    "Average coordinate RMSE on identical test graphs", "coordinate RMSE")
    plot_vs_missing(df, "rmse_certified", main_methods, "fig_certified_rmse.png",
                    "RMSE on nodes whose position is provably unique", "certified-node RMSE")
    if ablations:
        plot_vs_missing(df, "rmse", [OURS] + ablations + ["DV-distance + dwMDS (no learning)"],
                        "fig_ablation.png", "Ablation: what does each part contribute?", "coordinate RMSE")
    rt = df.groupby("method").seconds.mean().sort_values()
    plt.figure(figsize=(8, 0.45 * len(rt) + 1)); plt.barh(rt.index, rt.values, color="slategray")
    plt.xscale("log"); plt.xlabel("seconds per graph (log)"); plt.title("Runtime per graph")
    plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, "fig_runtime.png"), dpi=150); plt.close()


# -----------------------------------------------------------------------------
# transfer experiment: train on some graph types, test on all
# -----------------------------------------------------------------------------
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def transfer_matrix(df):
    """Mean RMSE over missing >= TRANSFER_MIN_MISSING: rows = method, cols = test scenario."""
    d = df[df.missing >= 100 * C.TRANSFER_MIN_MISSING]
    cols = [f"{gt}/{nz}" for gt in C.GRAPH_TYPES for nz in C.NOISE_MODELS if f"{gt}/{nz}" in set(d.scenario)]
    piv = d.pivot_table(index="method", columns="scenario", values="rmse", aggfunc="mean")[cols]
    ours = [TRANSFER + k for k in C.TRANSFER_TRAIN_SETS]
    blocks = [[m for m in ours if m in piv.index],
              [m + " + Huber polish" for m in ours if m + " + Huber polish" in piv.index],
              [m for m in dict.fromkeys(df.method) if not m.startswith(TRANSFER) and m in piv.index]]
    return piv.loc[sum(blocks, [])], blocks


def plot_transfer(piv, blocks, fname):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    norm = LogNorm(vmin=1e-3, vmax=max(piv.values.max(), 1e-2))
    V = piv.values
    fig, ax = plt.subplots(figsize=(2.0 * V.shape[1] + 5, 0.42 * V.shape[0] + 2.2))
    ax.imshow(np.clip(V, 1e-3, None), cmap=cmap, norm=norm, aspect="auto")
    best = np.nanargmin(V, axis=0)
    for i, mth in enumerate(piv.index):
        seen = C.train_types(mth[len(TRANSFER):].replace(" + Huber polish", "")) if mth.startswith(TRANSFER) else None
        for j, sc in enumerate(piv.columns):
            v = V[i, j]
            txt = "<0.001" if v < 1e-3 else f"{v:.3f}" if v < 1 else f"{v:.2f}"
            dark = norm(max(v, 1e-3)) > 0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if dark else "#1a1a1a", fontweight="bold" if best[j] == i else "normal")
            if seen is not None and sc.split("/")[0] not in seen:      # graph type never seen in training
                ax.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False, ls="--", lw=1.2,
                                       ec="white" if dark else "#1a1a1a"))
    edge = 0
    for b in blocks[:-1]:                     # white gaps between the three row blocks
        edge += len(b)
        if 0 < edge < len(piv.index): ax.axhline(edge - 0.5, color="white", lw=4)
    ax.set_xticks(range(V.shape[1])); ax.set_xticklabels(piv.columns, fontsize=9)
    ax.set_yticks(range(V.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"Mean coordinate RMSE, missing >= {int(100 * C.TRANSFER_MIN_MISSING)} %  (test graph type / noise; "
                 "random = spanning tree + random edges)\n"
                 "dashed box = graph type never seen in training  ·  bold = best in column  ·  darker = worse",
                 fontsize=9, pad=28)
    plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, fname), dpi=150); plt.close()


def run_transfer(plot_only=False):
    out = os.path.join(C.RESULTS_DIR, "transfer_results.csv")
    if plot_only:
        df = pd.read_csv(out)
    else:
        models = load_models(transfer=True)
        if not models:
            raise SystemExit("No transfer checkpoints -- run  python step2_train.py --train-on <types>  first.")
        main_csv, test_pkl = os.path.join(C.RESULTS_DIR, "all_results.csv"), os.path.join(C.DATA_DIR, "test.pkl")
        if not os.path.exists(main_csv) or os.path.getmtime(main_csv) < os.path.getmtime(test_pkl):
            raise SystemExit("Run  python step3_evaluate.py  (main comparison) first on the current test set: "
                             "the transfer table reuses its existing-method rows.")
        print(f"STEP 3 (transfer): {list(models)}")
        df, _ = evaluate_set(load("test.pkl"), models, {}, label="[transfer]")
        base = pd.read_csv(main_csv)
        base = base[~base.method.str.startswith(("Ours", "Ablation"))]   # existing methods: no training, same graphs
        df = pd.concat([df, base], ignore_index=True)
        df.to_csv(out, index=False)
    piv, blocks = transfer_matrix(df)
    piv.to_csv(os.path.join(C.RESULTS_DIR, "transfer_matrix.csv"))
    plot_transfer(piv, blocks, "fig_transfer_matrix.png")
    print(f"\nTRANSFER MATRIX: mean RMSE, missing >= {int(100 * C.TRANSFER_MIN_MISSING)} %\n", piv.round(3).to_string())


def main():
    import sys
    if "--transfer" in sys.argv:
        run_transfer(plot_only="--plot-only" in sys.argv); return
    if "--plot-only" in sys.argv:          # redraw figures from saved results, no re-running
        main_plots(pd.read_csv(os.path.join(C.RESULTS_DIR, "all_results.csv")))
        print("figures redrawn"); return
    mode_info = f"Dual-GPU mode ({', '.join(DEVICES)})" if len(DEVICES) >= 2 else f"Device: {DEVICE}"
    print(f"STEP 3: evaluation on {mode_info}", flush=True)
    models = load_models()
    if not models:
        raise SystemExit("No checkpoints found -- run step2_train.py first.")
    print("  models:", list(models), flush=True)
    baselines = BL.get_baselines(C, device=DEVICE)
    test = load("test.pkl")


    # ---------------- main comparison ----------------
    df, preds = evaluate_set(test, models, baselines, label="[main]")
    df.to_csv(os.path.join(C.RESULTS_DIR, "all_results.csv"), index=False)
    summ = (df.groupby(["scenario", "missing", "method"])
              .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), rmse_median=("rmse", "median"),
                   rmse_cert_mean=("rmse_certified", "mean"), rel_edm_mean=("rel_edm", "mean"),
                   sec_per_graph=("seconds", "mean"), frac_rigid=("globally_rigid", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(C.RESULTS_DIR, "summary_table.csv"), index=False)

    main_plots(df)

    show = [m for m in [OURS, "MDS-MAP + Huber stress", "SDP (Biswas-Ye)", "Yan et al. GCN (3 anchor labels)"]
            if m in preds]
    plot_examples(test, preds, show, "fig_examples.png")

    # ---------------- printed table ----------------
    piv = summ.pivot_table(index=["scenario", "missing"], columns="method", values="rmse_mean")
    print("\nMEAN COORDINATE RMSE (lower is better)\n", piv.round(3).to_string())

    # ---------------- generalisation to bigger graphs ----------------
    fast = {k: v for k, v in baselines.items() if "Per-graph" not in k and "Yan" not in k}
    gen_rows = []
    for n in C.GENERALISE_SIZES:
        G = load(f"test_n{n}.pkl")
        if G is None: continue
        fb = {k: v for k, v in fast.items() if not (k.startswith("SDP") and n > C.SDP_MAX_N)}
        d, _ = evaluate_set(G, {OURS: models[OURS]} if OURS in models else models, fb, label=f"[n={n}]")
        gen_rows.append(d)
    if gen_rows:
        gd = pd.concat(gen_rows); gd.to_csv(os.path.join(C.RESULTS_DIR, "generalisation.csv"), index=False)
        fig, axes = plt.subplots(1, len(C.GENERALISE_SIZES), figsize=(7 * len(C.GENERALISE_SIZES), 4.5), squeeze=False)
        for ax, n in zip(axes[0], C.GENERALISE_SIZES):
            d = gd[gd.n == n]
            for mth in d.method.unique():
                s = d[d.method == mth].groupby("missing").rmse.mean()
                ax.plot(s.index, s.values, label=mth, **style_for(mth, list(d.method.unique())))
            ax.set_yscale("log"); ax.set_title(f"n = {n} (never seen in training)")
            ax.set_xlabel("missing (%)"); ax.set_ylabel("coordinate RMSE"); ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(C.RESULTS_DIR, "fig_generalisation.png"), dpi=150); plt.close()

    # ---------------- Barabasi-Albert rigidity threshold ----------------
    ba = load("test_ba.pkl")
    if ba is not None:
        d, _ = evaluate_set(ba, {OURS: models[OURS]} if OURS in models else models, fast,
                            polish=False, label="[BA]")
        d.to_csv(os.path.join(C.RESULTS_DIR, "ba_results.csv"), index=False)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for mth in d.method.unique():
            s = d[d.method == mth].groupby("ba_m").rmse.mean()
            ax[0].plot(s.index, s.values, label=mth, **style_for(mth, list(d.method.unique())))
        ax[0].set_yscale("log"); ax[0].set_xlabel("BA parameter m (links per new node)")
        ax[0].set_ylabel("coordinate RMSE"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        r = d.drop_duplicates(["ba_m", "graph_id"]).groupby("ba_m")[["globally_rigid", "frac_certified"]].mean()
        ax[1].bar(r.index - 0.2, r.globally_rigid, 0.4, label="graphs globally rigid")
        ax[1].bar(r.index + 0.2, r.frac_certified, 0.4, label="nodes certified unique")
        ax[1].set_xlabel("m"); ax[1].set_ylim(0, 1.05); ax[1].legend(); ax[1].set_title("Is the answer unique?")
        plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, "fig_ba_rigidity.png"), dpi=150); plt.close()

    print(f"\nSTEP 3 done. Figures and CSVs in {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
