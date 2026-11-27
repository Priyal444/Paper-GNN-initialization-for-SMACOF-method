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
from common import coord_rmse, rel_edm_error, cramer_rao_lower_bound, dump_run_config
from model import AnchorGNNLocalizer, collate, to_original_units
import baselines as BL

warnings.filterwarnings("ignore")
FORCE_CPU = os.environ.get("DEVICE", "").lower() == "cpu"
NUM_GPUS = 0 if FORCE_CPU else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
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
    "huber_alpha": "Ablation: Huber weights + learned alpha",
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
        huber_delta = ck.get("huber_delta", C.HUBER_DELTA)
        m = AnchorGNNLocalizer(ck["in_dim"], ck["edge_dim"], ck["H"], ck["enc_layers"],
                               ck["refine_steps"], ck["variant"], huber_delta=huber_delta).to(DEVICE)
        m.load_state_dict(ck["state"]); m.eval()
        name = TRANSFER + ck["train_on"] if transfer else VARIANT_NAMES.get(tag, tag)
        models[name] = m
    return models



GNN_CPU_FALLBACKS = 0          # graphs whose GNN forward pass had to run on CPU (GPU OOM)
MAX_EDGES_PER_BATCH = 150_000  # directed edges per forward batch; bounds GPU memory on big graphs


def _forward_chunks(m, chk, dev, bs):
    """Forward pass over `chk` in batches of `bs` graphs. Graphs in a batch are disconnected
    components, so batching does not change any prediction, only memory use and timing.
    On CUDA OOM: retry that batch one graph at a time; if a single graph still does not fit,
    run it on CPU (counted in GNN_CPU_FALLBACKS, because its timing is then a CPU timing)."""
    import copy
    global GNN_CPU_FALLBACKS
    T_eval = lambda mm: None if mm.T == 0 else C.TEST_REFINE_STEPS
    out, t = [], 0.0
    cpu_model = None
    with torch.no_grad():
        for i in range(0, len(chk), bs):
            b_chk = chk[i:i + bs]
            try:
                B = collate(b_chk, dev)
                t0 = time.perf_counter(); x, _ = m(B, T=T_eval(m))
                if "cuda" in dev: torch.cuda.synchronize(dev)
                t += time.perf_counter() - t0
                out += to_original_units(B, x)
                continue
            except torch.OutOfMemoryError:
                B = x = None
                torch.cuda.empty_cache()
            for g in b_chk:
                try:
                    B = collate([g], dev)
                    t0 = time.perf_counter(); x, _ = m(B, T=T_eval(m))
                    if "cuda" in dev: torch.cuda.synchronize(dev)
                except torch.OutOfMemoryError:
                    B = x = None
                    torch.cuda.empty_cache()
                    if cpu_model is None:
                        cpu_model = copy.deepcopy(m).to("cpu").eval()
                    GNN_CPU_FALLBACKS += 1
                    print(f"    [WARN] GPU OOM on a graph with n={g.get('n')}; running it on CPU "
                          f"(GNN CPU fallbacks so far: {GNN_CPU_FALLBACKS})", flush=True)
                    B = collate([g], "cpu")
                    t0 = time.perf_counter(); x, _ = cpu_model(B, T=T_eval(cpu_model))
                t += time.perf_counter() - t0
                out += to_original_units(B, x)
            torch.cuda.empty_cache()
    return out, t


def run_gnn(model, graphs, bs=64):
    """One forward pass per batch; parallel across available GPUs."""
    import copy, concurrent.futures
    avg_n = int(np.mean([g.get('n', 50) for g in graphs])) if graphs else 50
    avg_e = int(np.mean([g["edge_index"].shape[1] for g in graphs])) if graphs else 1
    bs = max(1, min(bs, 2000 // max(1, avg_n), MAX_EDGES_PER_BATCH // max(1, avg_e)))
    if len(DEVICES) >= 2 and len(graphs) >= 2:
        models_per_dev = [model if dev == DEVICE else copy.deepcopy(model).to(dev).eval()
                          for dev in DEVICES]
        split_indices = np.array_split(np.arange(len(graphs)), len(DEVICES))

        def run_dev(m, chk, dev):
            return _forward_chunks(m, chk, dev, bs)

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
        out, t = _forward_chunks(model, graphs, DEVICE, bs)
        return out, t / len(graphs)


def score_rows(g, method, X, secs):
    r = coord_rmse(g, X)
    n_nodes = g.get("n", len(g.get("true_xy", [])))
    cert = g.get("certified")
    if cert is None:
        cert = np.ones(n_nodes, dtype=bool)
    rc = coord_rmse(g, X, cert)
    re = rel_edm_error(g, X)
    r_eval = rc if not np.isnan(rc) else r
    scen = g.get("scenario", f"{g.get('graph_type', 'rgg')}/{g.get('noise', 'clean')}")
    crlb = cramer_rao_lower_bound(g, sigma=0.05) if "gauss" in scen else np.nan
    rmse_crlb_subset = r if not np.isnan(crlb) else np.nan
    miss_nom = g.get("missing_nominal", g.get("missing_ratio", 0.5))
    frac_cert = float(cert.mean()) if hasattr(cert, "mean") else 1.0
    return dict(scenario=scen, missing=int(round(100 * miss_nom)), n=n_nodes,
                graph_id=g.get("graph_id", 0), method=method, rmse=r,
                rmse_certified=rc, rmse_crlb_subset=rmse_crlb_subset, rel_edm=re, crlb=crlb,
                catastrophic_fail=bool(r > 1.5),
                fail_05=bool(r_eval > 0.5),
                fail_10=bool(r_eval > 1.0),
                fail_15=bool(r_eval > 1.5),
                fail_20=bool(r_eval > 2.0),
                seconds=secs, globally_rigid=bool(g.get("globally_rigid", True)),
                frac_certified=frac_cert, ba_m=g.get("ba_m"),
                anchor_k=g.get("anchor_k", len(g.get("anchors", []))),
                p_nlos=g.get("p_nlos", 0.0))



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
        "DV-distance + dwMDS (no learning)", "Classical MDS (direct)",
        "Rank Alternation (EDM book Alg 2)", "OptSpace Spectral EDM (Alg 3 init)",
        "s-stress L-BFGS (Alg 4 s-stress)", "Yan et al. GCN (3 anchor labels)",
        "MPNN-Loc (Xu & Lin 2025)", "Per-graph GraphSAGE (previous)"
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
            n_cpu_workers = min(os.cpu_count() or 4, 8)
            print(f"{label} [Polish] Running Huber L-BFGS polish on {n_graphs} graphs (parallel across {n_cpu_workers} CPU workers)...", flush=True)
            t_pol_start = time.time()
            log_step = min(50, max(5, n_graphs // 20))
            last_log_t = time.time()
            
            def polish_one(item):
                idx, g, X = item
                t0 = time.perf_counter()
                Y = BL.stress_lbfgs(g, X, robust=True)
                dur = time.perf_counter() - t0
                return idx, Y, dur

            P2 = [None] * n_graphs
            polish_rows = [None] * n_graphs
            completed = 0
            print_lock = threading.Lock()

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu_workers) as pool:
                futures = [pool.submit(polish_one, (idx, g, X)) for idx, (g, X) in enumerate(zip(graphs, P))]
                for f in concurrent.futures.as_completed(futures):
                    idx, Y, dur = f.result()
                    P2[idx] = Y
                    polish_rows[idx] = score_rows(graphs[idx], name + " + Huber polish", Y, sec + dur)
                    with print_lock:
                        completed += 1
                        now = time.time()
                        if completed % log_step == 0 or completed == n_graphs or (now - last_log_t >= 10.0):
                            elapsed = now - t_pol_start
                            eta = (elapsed / completed) * (n_graphs - completed)
                            last_log_t = now
                            print(f"    [Polish progress] {completed}/{n_graphs} graphs ({completed*100//n_graphs}%) | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

            rows.extend(polish_rows)
            preds[name + " + Huber polish"] = P2
            pol_rmse = np.nanmean([coord_rmse(g, Y) for g, Y in zip(graphs, P2)])
            print(f"{label} [Polish] Finished in {time.time() - t_pol_start:.1f}s -> Mean RMSE: {pol_rmse:.4f}", flush=True)


    # 2. Baselines (cached across seeds on identical test graphs)
    cache_lbl = label.replace("[", "").replace("]", "").strip()
    cache_file = os.path.join(C.DATA_DIR, f"baseline_cache_{cache_lbl}.pkl" if cache_lbl else "baseline_cache.pkl")
    cached_data = {}
    cached_fallbacks = 0
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and "data" in loaded:
                    cached_data = loaded["data"]
                    cached_fallbacks = loaded.get("sdp_fallbacks", 0)
                elif isinstance(loaded, dict):
                    cached_data = loaded
        except Exception:
            cached_data = {}

    to_run = {}
    for name, fn in baselines.items():
        if name in cached_data and len(cached_data[name]["rows"]) == n_graphs:
            print(f"{label} [Baseline Cache] Reusing cached predictions for '{name}' ({n_graphs} graphs)", flush=True)
            preds[name] = cached_data[name]["preds"]
            rows.extend(cached_data[name]["rows"])
            if "SDP" in name and cached_fallbacks > 0:
                BL.SDP_FALLBACK_COUNT = cached_fallbacks
        else:
            to_run[name] = fn

    n_base = len(to_run)
    cache_dirty = False
    for b_idx, (name, fn) in enumerate(to_run.items(), 1):
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
        log_step = min(50, max(5, n_graphs // 20))
        last_log_t = time.time()
        completed = 0
        print_lock = threading.Lock()

        def run_one_baseline(item):
            k, g = item
            dev = DEVICES[k % len(DEVICES)] if is_gpu else DEVICE
            target_fn = base_per_dev[dev].get(name, fn)
            had_sdp = g.get("_sdp") is not None
            X, sec = BL.timed(target_fn, g)
            if name.startswith("SDP + ") and had_sdp:
                sec += g.get("_sdp_time", 0.0)   # count reused SDP solve duration
            return k, X, sec

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(run_one_baseline, (k, g)) for k, g in enumerate(graphs)]
            for f in concurrent.futures.as_completed(futures):
                k, X, sec = f.result()
                P[k] = X
                base_rows[k] = score_rows(graphs[k], name, X, sec)
                with print_lock:
                    completed += 1
                    now = time.time()
                    if completed % log_step == 0 or completed == n_graphs or (now - last_log_t >= 10.0):
                        elapsed = now - t0
                        eta = (elapsed / completed) * (n_graphs - completed)
                        last_log_t = now
                        curr_rmse = np.nanmean([coord_rmse(graphs[j], P[j]) for j in range(n_graphs) if P[j] is not None])
                        print(f"    progress: {completed}/{n_graphs} graphs ({completed*100//n_graphs}%) | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | avg RMSE so far: {curr_rmse:.4f}", flush=True)

        preds[name] = P
        rows.extend(base_rows)
        cached_data[name] = dict(preds=P, rows=base_rows)
        cache_dirty = True
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": cached_data, "sdp_fallbacks": BL.get_sdp_fallbacks()}, f)
        except Exception:
            pass
        b_rmse = np.nanmean([coord_rmse(g, X) for g, X in zip(graphs, P)])
        print(f"{label} [{b_idx}/{n_base}] '{name}' completed in {time.time() - t0:.1f}s -> Mean RMSE: {b_rmse:.4f}", flush=True)

    if cache_dirty:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"data": cached_data, "sdp_fallbacks": BL.get_sdp_fallbacks()}, f)
        except Exception:
            pass

    return pd.DataFrame(rows), preds




# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
STYLE = {OURS: dict(color="crimson", lw=2.6, marker="o"),
         OURS + " + Huber polish": dict(color="darkred", lw=2.2, marker="s", ls="--"),
         "SDP + Huber polish": dict(color="purple", lw=1.8, marker="^", ls=":"),
         "MDS-MAP + Huber stress": dict(color="teal", lw=1.8, marker="d", ls="-."),
         "DV-distance + Huber polish": dict(color="darkorange", lw=1.8, marker="v", ls="-.")}



PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#17becf", "#bcbd22", "#393b79"]
FLOOR = 1e-6     # errors below this are numerically exact; drawn on the floor


def style_for(mth, methods):
    if mth in STYLE:
        return STYLE[mth]
    others = [m for m in methods if m not in STYLE]
    return dict(color=PALETTE[others.index(mth) % len(PALETTE)], lw=1.3, marker=".")


def save_fig_both(fname, dpi=150):
    p_png = os.path.join(C.RESULTS_DIR, fname)
    p_eps = os.path.join(C.RESULTS_DIR, fname.replace(".png", ".eps"))
    plt.savefig(p_png, dpi=dpi)
    plt.savefig(p_eps, format="eps")
    plt.close()


def plot_vs_missing(df, metric, methods, fname, title, ylabel):
    """Line = MEAN over the test graphs at each missing level.
    Shaded band = 25th-75th percentile (a +/- std band is meaningless on a log axis)."""
    scen = list(dict.fromkeys(df.scenario))
    cols = 2; rows = int(np.ceil(len(scen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.6 * rows), squeeze=False)
    for ax, sc in zip(axes.ravel(), scen):
        d = df[df.scenario == sc]
        for mth in methods:
            use_col = "rmse_crlb_subset" if (metric == "rmse" and "gauss" in sc and "rmse_crlb_subset" in d.columns) else metric
            g = d[d.method == mth].groupby("missing")[use_col]
            if g.ngroups == 0: continue
            mean, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
            st = style_for(mth, methods)
            ax.plot(mean.index, np.maximum(mean, FLOOR), label=mth, **st)
            ax.fill_between(mean.index, np.maximum(q1, FLOOR), np.maximum(q3, FLOOR),
                            color=st.get("color"), alpha=0.10, lw=0)
        if metric in ("rmse", "rmse_certified") and "crlb" in d.columns and "gauss" in sc:
            crlb_s = d.drop_duplicates(["scenario", "missing", "graph_id"]).groupby("missing")["crlb"].mean()
            if not crlb_s.isna().all():
                ax.plot(crlb_s.index, np.maximum(crlb_s.values, FLOOR), label="CRLB (bound)",
                        color="black", ls=":", lw=1.8, marker="")
        ax.set_yscale("log"); ax.set_ylim(bottom=FLOOR * 0.7); ax.set_title(sc)


        ax.set_xlim(df.missing.min() - 3, df.missing.max() + 3)   # same x range in every panel (BA starts at 50 %)
        ax.set_xlabel("missing distances (%)")
        ax.set_ylabel(f"{ylabel}  (floor = exact)"); ax.grid(alpha=.3, which="both")
    for ax in axes.ravel()[len(scen):]: ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title + "\n(line = mean over test graphs, band = 25-75th percentile)"); plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_fig_both(fname, dpi=150)


def plot_examples(graphs, preds, methods, fname, levels=(30, 70, 90)):
    pick = []
    for lv in levels:
        for i, g in enumerate(graphs):
            if int(round(100 * g.get("missing_nominal", g.get("missing_ratio", 0.5)))) == lv:
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
    plt.tight_layout()
    save_fig_both(fname, dpi=130)


def main_plots(df):
    """All figures that only need all_results.csv (re-drawable with --plot-only)."""
    ablations = [m for m in df.method.unique() if m.startswith("Ablation") or m.startswith("Ours, self")]
    main_methods = [m for m in df.method.unique() if m not in ablations]
    plot_vs_missing(df, "rmse", main_methods, "fig_main_rmse.png",
                    "Average coordinate RMSE on identical test graphs", "coordinate RMSE")
    plot_vs_missing(df, "rmse_certified", main_methods, "fig_certified_rmse.png",
                    "RMSE on nodes whose position is provably unique", "certified-node RMSE")
    plot_vs_missing(df, "rel_edm", main_methods, "fig_relative_error.png",
                    "Average Relative EDM Error ||D - D_hat||_F / ||D||_F (Dokmanic et al.)", "relative EDM error")
    if ablations:
        plot_vs_missing(df, "rmse", [OURS] + ablations + ["DV-distance + dwMDS (no learning)"],
                        "fig_ablation.png", "Ablation: what does each part contribute?", "coordinate RMSE")
    rt = df.groupby("method").seconds.mean().sort_values()
    plt.figure(figsize=(8, 0.45 * len(rt) + 1)); plt.barh(rt.index, rt.values, color="slategray")
    plt.xscale("log"); plt.xlabel("seconds per graph (log)"); plt.title("Runtime per graph")
    plt.tight_layout()
    save_fig_both("fig_runtime.png", dpi=150)



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
    plt.tight_layout(); save_fig_both(fname, dpi=150)


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


def plot_extended_sweeps(df_anc, df_nlos):
    # 1. Anchor Sensitivity Plot (K = 3..8)
    if df_anc is not None and not df_anc.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        methods = [m for m in df_anc["method"].unique() if not m.startswith("Ablation") or m == OURS]
        ax0, ax1 = axes[0], axes[1]
        for mth in methods:
            sub = df_anc[df_anc["method"] == mth]
            agg = sub.groupby("anchor_k")[["rmse", "rel_edm"]].mean()
            st = style_for(mth, methods)
            ax0.plot(agg.index, agg["rmse"], label=mth, **st)
            ax1.plot(agg.index, agg["rel_edm"], label=mth, **st)
        ax0.set_xlabel("Number of Anchors (K)", fontsize=10)
        ax0.set_ylabel("Coordinate RMSE (m)", fontsize=10)
        ax0.set_title("Anchor Sensitivity: Coordinate RMSE vs K", fontsize=11)
        ax0.grid(True, alpha=0.3); ax0.set_yscale("log")
        ax1.set_xlabel("Number of Anchors (K)", fontsize=10)
        ax1.set_ylabel("Relative EDM Error ||D - D^||_F / ||D||_F", fontsize=10)
        ax1.set_title("Anchor Sensitivity: Relative Error vs K", fontsize=11)
        ax1.grid(True, alpha=0.3); ax1.set_yscale("log")
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(methods))), fontsize=8, frameon=False)
        plt.tight_layout(rect=(0, 0.08, 1, 0.95))
        save_fig_both("fig_anchor_sensitivity.png", dpi=150)

    # 2. NLOS Contamination Sweep Plot (p_nlos = 0..0.5)
    if df_nlos is not None and not df_nlos.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        methods = [m for m in df_nlos["method"].unique() if not m.startswith("Ablation") or m == OURS]
        ax0, ax1 = axes[0], axes[1]
        for mth in methods:
            sub = df_nlos[df_nlos["method"] == mth]
            agg = sub.groupby("p_nlos")[["rmse", "rel_edm"]].mean()
            st = style_for(mth, methods)
            ax0.plot(agg.index, agg["rmse"], label=mth, **st)
            ax1.plot(agg.index, agg["rel_edm"], label=mth, **st)
        ax0.set_xlabel("NLOS Probability p_nlos", fontsize=10)
        ax0.set_ylabel("Coordinate RMSE (m)", fontsize=10)
        ax0.set_title("NLOS Robustness: Coordinate RMSE vs p_nlos", fontsize=11)
        ax0.grid(True, alpha=0.3)
        ax1.set_xlabel("NLOS Probability p_nlos", fontsize=10)
        ax1.set_ylabel("Relative EDM Error ||D - D^||_F / ||D||_F", fontsize=10)
        ax1.set_title("NLOS Robustness: Relative Error vs p_nlos", fontsize=11)
        ax1.grid(True, alpha=0.3)
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(methods))), fontsize=8, frameon=False)
        plt.tight_layout(rect=(0, 0.08, 1, 0.95))
        save_fig_both("fig_nlos_sweep.png", dpi=150)


def run_extended_sweeps(plot_only=False):
    anc_csv = os.path.join(C.RESULTS_DIR, "anchor_sweep_results.csv")
    nlos_csv = os.path.join(C.RESULTS_DIR, "nlos_sweep_results.csv")
    df_anc, df_nlos = None, None
    if plot_only:
        if os.path.exists(anc_csv): df_anc = pd.read_csv(anc_csv)
        if os.path.exists(nlos_csv): df_nlos = pd.read_csv(nlos_csv)
    else:
        models = load_models(transfer=False)
        models = {k: v for k, v in models.items() if k in (OURS, "Ablation: encoder only (no refiner)")}
        base_all = BL.get_baselines(C, device=DEVICE)
        eval_base = {k: v for k, v in base_all.items() if k in (
            "MDS-MAP", "MDS-MAP + Huber stress", "DV-distance + Huber polish", "SDP (Biswas-Ye)", "SDP + Huber polish",
            "s-stress L-BFGS (Alg 4 s-stress)", "Rank Alternation (EDM book Alg 2)",
            "MPNN-Loc (Xu & Lin 2025)"
        )}
        # 1. Anchor sweep
        anc_graphs = load("anchor_sweep_test.pkl")
        if anc_graphs:
            print(f"\n[Extended] Evaluating Anchor Sensitivity on {len(anc_graphs)} graphs...", flush=True)
            df_anc, _ = evaluate_set(anc_graphs, models, eval_base, polish=True, label="[anchors]")
            df_anc.to_csv(anc_csv, index=False)
            print("Anchor sweep saved ->", anc_csv)
        else:
            print("\n[Extended] Warning: anchor_sweep_test.pkl not found! Run step1_data.py first.")

        # 2. NLOS sweep
        nlos_graphs = load("nlos_sweep_test.pkl")
        if nlos_graphs:
            print(f"\n[Extended] Evaluating NLOS Contamination on {len(nlos_graphs)} graphs...", flush=True)
            df_nlos, _ = evaluate_set(nlos_graphs, models, eval_base, polish=True, label="[nlos]")
            df_nlos.to_csv(nlos_csv, index=False)
            print("NLOS sweep saved ->", nlos_csv)
        else:
            print("\n[Extended] Warning: nlos_sweep_test.pkl not found! Run step1_data.py first.")

    plot_extended_sweeps(df_anc, df_nlos)
def compute_rank_auc(y_true, scores):
    """Computes ROC AUC via Wilcoxon-Mann-Whitney rank sum with average ranks for ties."""
    from scipy.stats import rankdata
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    rank_sum_pos = ranks[y_true].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def analyze_nlos_weights(models, graphs):
    """
    Evaluates whether the GNN's learned edge weights specifically detect and
    downweight positive NLOS outliers compared to classical Huber IRLS weights.
    Generates results/nlos_weight_analysis.csv and results/fig_nlos_edge_weights.png.
    """
    if OURS not in models:
        return
    model = models[OURS]
    model.eval()

    nlos_graphs = [g for g in graphs if g.get("is_nlos") is not None and g["is_nlos"].any()]
    if not nlos_graphs:
        return
    print(f"\n[Analysis] Evaluating NLOS outlier edge detection across {len(nlos_graphs)} NLOS graphs...", flush=True)

    rows = []
    all_w_clean_learned, all_w_nlos_learned = [], []
    all_w_clean_huber, all_w_nlos_huber = [], []

    for g in nlos_graphs:
        B = collate([g], DEVICE)
        with torch.no_grad():
            x, xA, w_learned, x_prev = model(B, T=C.TEST_REFINE_STEPS, return_weights=True)
            w_learned = w_learned.detach().cpu().numpy()

            # Classical Huber weights evaluated at initial Stage-A estimate xA (before unrolled refiner)
            ei = B["edge_index"].cpu().numpy()
            d = B["d_edge"].cpu().numpy()
            src, dst = ei[0], ei[1]
            diff = (xA[dst] - xA[src]).cpu().numpy()
            dhat = np.maximum(np.linalg.norm(diff, axis=-1), 1e-4)
            r = np.abs(d - dhat)
            w_huber = np.clip(C.HUBER_DELTA / np.maximum(r, 1e-9), 1e-3, 1.0)

        # Labels: clean = 1 (higher weight expected), nlos = 0
        is_nlos = np.r_[g["is_nlos"], g["is_nlos"]]
        is_clean = ~is_nlos

        auc_learned = compute_rank_auc(is_clean, w_learned)
        auc_huber = compute_rank_auc(is_clean, w_huber)

        all_w_clean_learned.extend(w_learned[is_clean])
        all_w_nlos_learned.extend(w_learned[is_nlos])
        all_w_clean_huber.extend(w_huber[is_clean])
        all_w_nlos_huber.extend(w_huber[is_nlos])

        rows.append(dict(
            scenario=g["scenario"], missing=int(round(100 * g["missing_nominal"])),
            graph_id=g["graph_id"], auc_learned=auc_learned, auc_huber=auc_huber,
            frac_nlos=float(is_nlos.mean())
        ))

    df_auc = pd.DataFrame(rows)
    df_auc.to_csv(os.path.join(C.RESULTS_DIR, "nlos_weight_analysis.csv"), index=False)

    mean_auc_learned = float(df_auc["auc_learned"].mean())
    mean_auc_huber = float(df_auc["auc_huber"].mean())
    print(f"  [Analysis Result] NLOS Edge Detection ROC AUC:")
    print(f"    Learned SMACOF Weights: AUC = {mean_auc_learned:.4f}")
    print(f"    Classical Huber Weights: AUC = {mean_auc_huber:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if all_w_clean_learned and all_w_nlos_learned:
        axes[0].boxplot([all_w_clean_learned, all_w_nlos_learned], labels=["Clean Edges", "NLOS Edges"])
        axes[0].set_ylabel("Edge Weight $w_{ij}$")
        axes[0].set_title(f"Learned SMACOF Gate (AUC = {mean_auc_learned:.3f})")
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot([all_w_clean_huber, all_w_nlos_huber], labels=["Clean Edges", "NLOS Edges"])
        axes[1].set_ylabel("Edge Weight $w_{ij}$")
        axes[1].set_title(f"Classical Huber IRLS (AUC = {mean_auc_huber:.3f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig_both("fig_nlos_edge_weights.png", dpi=150)


def run_fair_runtime_benchmark(graphs, models, baselines):
    """
    Executes a 100% fair single-threaded CPU runtime benchmark across all methods.
    - Representative sampling: samples 1 graph per (scenario, missing level) cell.
    - Single-threaded environment: sets BLAS/OMP/MKL threads to 1 and torch.set_num_threads(1).
    - Measures both feature extraction and inference for GNN, plus Ours + Huber polish.
    - Runs untimed warm-up passes to eliminate JIT/initialisation bias.
    - Avoids in-place model mutation via copy.deepcopy.
    """
    import copy
    from common import build_features

    # 1. Representative sampling: 1 graph per (scenario, missing) cell
    cell_dict = {}
    for g in graphs:
        cell_key = (g.get("scenario", "default"), g.get("missing_nominal", 0.0))
        if cell_key not in cell_dict:
            cell_dict[cell_key] = g
    sample_graphs = list(cell_dict.values())
    n_sample = len(sample_graphs)
    print(f"\n[Benchmark] Running fair single-threaded CPU runtime benchmark on {n_sample} representative graphs (1 per cell)...", flush=True)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = "1"

    records = []

    # 2. GNN on CPU with feature building and polish included
    if OURS in models:
        gnn_cpu = copy.deepcopy(models[OURS]).to("cpu").eval()

        # Warm-up passes (untimed)
        warmup_B = collate([sample_graphs[0]], "cpu")
        with torch.no_grad():
            for _ in range(3):
                _ = gnn_cpu(warmup_B, T=C.TEST_REFINE_STEPS)

        t_feats, t_fwds, t_polish = [], [], []
        for g in sample_graphs:
            # Dijkstra shortest path + multilateration feature building
            t0 = time.perf_counter()
            _ = build_features(g["n"], g["edges"], g["d_obs"], g["anchors"], g["anchor_xy"])
            t_feats.append(time.perf_counter() - t0)

            # GNN Forward Pass
            B = collate([g], "cpu")
            t0 = time.perf_counter()
            with torch.no_grad():
                x_out, _ = gnn_cpu(B, T=C.TEST_REFINE_STEPS)
            t_fwds.append(time.perf_counter() - t0)
            X_pred = to_original_units(B, x_out)[0]

            # Huber Polish
            t0 = time.perf_counter()
            _ = BL.stress_lbfgs(g, X_pred, robust=True)
            t_polish.append(time.perf_counter() - t0)

        mean_feat_ms = float(np.mean(t_feats)) * 1000.0
        mean_fwd_ms = float(np.mean(t_fwds)) * 1000.0
        mean_pol_ms = float(np.mean(t_polish)) * 1000.0

        records.append(dict(
            method="Ours: AnchorGNN (features + CPU forward)",
            total_ms=mean_feat_ms + mean_fwd_ms,
            feature_ms=mean_feat_ms,
            inference_ms=mean_fwd_ms,
            polish_ms=0.0,
            hardware="Single-core CPU"
        ))
        records.append(dict(
            method="Ours: AnchorGNN + Huber polish (features + forward + polish)",
            total_ms=mean_feat_ms + mean_fwd_ms + mean_pol_ms,
            feature_ms=mean_feat_ms,
            inference_ms=mean_fwd_ms,
            polish_ms=mean_pol_ms,
            hardware="Single-core CPU"
        ))

    # 3. Key baselines sequentially on single-core CPU
    eval_base = ["MDS-MAP", "MDS-MAP + Huber stress", "DV-distance + Huber polish", "SDP (Biswas-Ye)", "SDP + Huber polish"]
    for b_name in eval_base:
        if b_name not in baselines:
            continue
        fn = baselines[b_name]
        t_list = []
        for g in sample_graphs:
            g.pop("_sdp", None); g.pop("_sdp_time", None)   # no cached SDP: time the full solve
            t0 = time.perf_counter()
            _ = fn(g)
            t_list.append(time.perf_counter() - t0)
        mean_base_ms = float(np.mean(t_list)) * 1000.0
        records.append(dict(
            method=b_name,
            total_ms=mean_base_ms,
            feature_ms=0.0,
            inference_ms=mean_base_ms,
            polish_ms=0.0,
            hardware="Single-core CPU"
        ))

    torch.set_num_threads(prev_threads)
    df_rt = pd.DataFrame(records)
    df_rt.to_csv(os.path.join(C.RESULTS_DIR, "fair_runtime_benchmark.csv"), index=False)
    print("\nFAIR RUNTIME BENCHMARK (Single-core CPU, representative sample, features included):")
    print(df_rt.to_string(index=False), flush=True)


def run_paired_wilcoxon_tests(df, out_dir=None):
    """
    Computes rigorous non-parametric paired Wilcoxon signed-rank tests
    on globally rigid graphs at 60-85% missing distance constraints.
    Saves results/paired_tests.csv.
    """
    from scipy.stats import wilcoxon
    out_dir = out_dir or C.RESULTS_DIR
    d = df[df.globally_rigid.astype(bool) & df.missing.between(10, 80)]
    if d.empty:
        return None

    pairs = [
        (OURS, "SDP + Huber polish", True),
        (OURS, "SDP + Asym polish (NLOS)", True),
        (OURS, "SDP (Biswas-Ye)", True),
        (OURS, "MDS-MAP + Huber stress", True),
        (OURS, "DV-distance + Huber polish", True),
        (OURS + " + Huber polish", "SDP + Huber polish", False),
        (OURS + " + Huber polish", "MDS-MAP + Huber stress", False),
        (OURS + " + Huber polish", "DV-distance + Huber polish", False),
        (OURS + " + Huber polish", "Random (multi-start) + Huber polish", False),
        (OURS, "Ablation: encoder + Huber-weight steps", True),
        (OURS, "Ablation: Huber weights + learned alpha", True),
    ]

    rows = []
    for a, b, nlos_only in pairs:
        dd = d[d.scenario.str.endswith("nlos")] if nlos_only else d
        p = dd.pivot_table(index=["scenario", "missing", "graph_id"], columns="method", values="rmse")
        if a not in p or b not in p:
            continue
        valid = p[[a, b]].dropna()
        if len(valid) < 5:
            continue
        diff = valid[a] - valid[b]
        if (diff.abs() <= 1e-4).all():
            stat, p_val = 0.0, 1.0
        else:
            try:
                res = wilcoxon(diff[diff.abs() > 1e-4], alternative="two-sided")
                stat, p_val = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, p_val = np.nan, np.nan
        w = int((diff < -1e-4).sum())
        l = int((diff > 1e-4).sum())
        t = int((diff.abs() <= 1e-4).sum())
        p_fail = dd.pivot_table(index=["scenario", "missing", "graph_id"], columns="method", values="fail_10")
        f_a = float(p_fail[a].mean()) if a in p_fail else np.nan
        f_b = float(p_fail[b].mean()) if b in p_fail else np.nan
        rows.append(dict(
            method_A=a, method_B=b, nlos_only=nlos_only, n_pairs=len(valid),
            wins_A=w, losses_A=l, ties=t, win_rate_A=round(w / max(1, w + l), 3),
            mean_rmse_A=round(float(valid[a].mean()), 4),
            mean_rmse_B=round(float(valid[b].mean()), 4),
            fail_rate_A=round(f_a, 4), fail_rate_B=round(f_b, 4),
            wilcoxon_stat=stat, p_value=p_val
        ))
    if rows:
        res_df = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, "paired_tests.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"\n[Statistical Hypothesis Testing] Paired Wilcoxon Signed-Rank Tests (Globally Rigid, 10-80% Missing):")
        print(res_df[["method_A", "method_B", "n_pairs", "win_rate_A", "mean_rmse_A", "mean_rmse_B", "p_value"]].to_string(index=False), flush=True)
        return res_df
    return None


def main():
    import sys
    if "--transfer" in sys.argv:
        run_transfer(plot_only="--plot-only" in sys.argv); return
    if "--extended" in sys.argv:
        run_extended_sweeps(plot_only="--plot-only" in sys.argv); return
    if "--fair-runtime" in sys.argv:
        test_file = getattr(C, "TEST_FILE", "test.pkl")
        run_fair_runtime_benchmark(load(test_file), load_models(), BL.get_baselines(C, device="cpu")); return
    if "--gen-only" in sys.argv:           # resume after a crash in stage 3/4 (stages 1-2 already saved)
        models = load_models()
        if not models:
            raise SystemExit("No checkpoints found -- run step2_train.py first.")
        print(f"STEP 3 (--gen-only): stages 3-4 only, device {DEVICE}; models: {list(models)}", flush=True)
        run_generalisation_and_ba(models, BL.get_baselines(C, device=DEVICE)); return
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
    test_file = getattr(C, "TEST_FILE", "test.pkl")
    test = load(test_file)
    dump_run_config(C.RESULTS_DIR, step_name="step3")
    print(f"  test set: {test_file} ({len(test)} graphs)", flush=True)


    # ---------------- main comparison ----------------
    print(f"\n=======================================================", flush=True)
    print(f" [STAGE 1/4] Evaluating Main Benchmark (N=50, {len(test)} graphs)", flush=True)
    print(f"=======================================================", flush=True)
    df, preds = evaluate_set(test, models, baselines, label="[main]")
    df.to_csv(os.path.join(C.RESULTS_DIR, "all_results.csv"), index=False)
    summ = (df.groupby(["scenario", "missing", "method"])
              .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), rmse_median=("rmse", "median"),
                   rmse_p90=("rmse", lambda x: float(np.percentile(x, 90))),
                   catastrophic_fail_rate=("catastrophic_fail", "mean"),
                   fail_rate_05=("fail_05", "mean"),
                   fail_rate_10=("fail_10", "mean"),
                   fail_rate_15=("fail_15", "mean"),
                   fail_rate_20=("fail_20", "mean"),
                   rmse_cert_mean=("rmse_certified", "mean"),
                   rmse_crlb_subset_mean=("rmse_crlb_subset", "mean"),
                   rel_edm_mean=("rel_edm", "mean"),
                   crlb_mean=("crlb", "mean"),
                   sec_per_graph=("seconds", "mean"), frac_rigid=("globally_rigid", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(C.RESULTS_DIR, "summary_table.csv"), index=False)
    
    print(f"\n[STAGE 2/4] Statistical Wilcoxon Tests & Generating Publication Plots...", flush=True)
    run_paired_wilcoxon_tests(df, C.RESULTS_DIR)

    sdp_fb = BL.get_sdp_fallbacks()
    if sdp_fb > 0:
        print(f"\n[NOTE] SDP Solver Fallbacks: {sdp_fb} evaluation graph(s) fell back to MDS-MAP.", flush=True)
    else:
        print("\n[NOTE] SDP Solver: 0 fallbacks (all evaluated SDP instances solved directly).", flush=True)

    main_plots(df)

    show = [m for m in [OURS, OURS + " + Huber polish", "MDS-MAP + Huber stress", "DV-distance + Huber polish", "SDP (Biswas-Ye)", "SDP + Huber polish", "MPNN-Loc (Xu & Lin 2025)"]
            if m in preds]
    plot_examples(test, preds, show, "fig_examples.png")

    analyze_nlos_weights(models, test)
    if getattr(C, "RUN_CPU_TIMING", False):
        run_fair_runtime_benchmark(test, models, baselines)

    # ---------------- printed table ----------------
    piv = summ.pivot_table(index=["scenario", "missing"], columns="method", values="rmse_mean")
    print("\nMEAN COORDINATE RMSE (lower is better)\n", piv.round(3).to_string())
    piv_med = summ.pivot_table(index=["scenario", "missing"], columns="method", values="rmse_median")
    print("\nMEDIAN COORDINATE RMSE (lower is better)\n", piv_med.round(3).to_string())

    # Stages 1-2 finished: a later crash only needs `--gen-only` (stages 3-4), not a full rerun.
    with open(os.path.join(C.RESULTS_DIR, "step3_main.done"), "w") as fh:
        fh.write(f"stages 1-2 complete; SDP fallbacks: {sdp_fb}; GNN CPU fallbacks: {GNN_CPU_FALLBACKS}\n")

    run_generalisation_and_ba(models, baselines)


def run_generalisation_and_ba(models, baselines):
    """Stages 3-4: size generalisation and Barabasi-Albert sweep."""
    # ---------------- generalisation to bigger graphs ----------------
    print(f"\n=======================================================", flush=True)
    print(f" [STAGE 3/4] Size Generalisation to Large Graphs (N = {C.GENERALISE_SIZES})", flush=True)
    print(f"=======================================================", flush=True)
    fast = {k: v for k, v in baselines.items() if "Per-graph" not in k and "Yan" not in k and "MPNN-Loc" not in k}
    gen_rows = []
    for n in C.GENERALISE_SIZES:
        G = load(f"test_n{n}.pkl")
        if G is None: continue
        fb = {k: v for k, v in fast.items() if not (k.startswith("SDP") and n > C.SDP_MAX_N)}
        print(f"\n--> Evaluating Size N = {n} ({len(G)} graphs) across {len(fb)+1} scalable methods...", flush=True)
        d, _ = evaluate_set(G, {OURS: models[OURS]} if OURS in models else models, fb, label=f"[n={n}]")
        gen_rows.append(d)
    if gen_rows:
        gd = pd.concat(gen_rows); gd.to_csv(os.path.join(C.RESULTS_DIR, "generalisation.csv"), index=False)
        gen_piv = gd.groupby(["n", "missing", "method"]).rmse.mean().unstack("method")
        print("\nGENERALISATION COORDINATE RMSE (by node count n):\n", gen_piv.round(3).to_string(), flush=True)
        fig, axes = plt.subplots(1, len(C.GENERALISE_SIZES), figsize=(5.5 * len(C.GENERALISE_SIZES), 4.5), squeeze=False)
        for ax, n in zip(axes[0], C.GENERALISE_SIZES):
            d = gd[gd.n == n]
            for mth in d.method.unique():
                s = d[d.method == mth].groupby("missing").rmse.mean()
                ax.plot(s.index, s.values, label=mth, **style_for(mth, list(d.method.unique())))
            ax.set_yscale("log"); ax.set_title(f"n = {n} (never seen in training)")
            ax.set_xlabel("missing (%)"); ax.set_ylabel("coordinate RMSE"); ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=7); plt.tight_layout()
        save_fig_both("fig_generalisation.png", dpi=150)

    # ---------------- Barabasi-Albert rigidity threshold ----------------
    ba = load("test_ba.pkl")
    if ba is not None:
        print(f"\n=======================================================", flush=True)
        print(f" [STAGE 4/4] Barabasi-Albert Rigidity Phase Transition (m = {C.BA_M_VALUES})", flush=True)
        print(f"=======================================================", flush=True)
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
        plt.tight_layout(); save_fig_both("fig_ba_rigidity.png", dpi=150)

    print(f"\n[NOTE] GNN CPU fallbacks (GPU OOM) in this process: {GNN_CPU_FALLBACKS}", flush=True)
    with open(os.path.join(C.RESULTS_DIR, "step3.done"), "w") as fh:
        fh.write(f"all stages complete; GNN CPU fallbacks in final process: {GNN_CPU_FALLBACKS}\n")
    print(f"\nSTEP 3 done. Figures and CSVs in {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
