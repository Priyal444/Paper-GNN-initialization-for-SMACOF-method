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

Run:  python step3_evaluate.py
      python step3_evaluate.py --plot-only     # only redraw the figures from results/all_results.csv
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def load_models():
    models = {}
    for path in sorted(glob.glob(os.path.join(C.CKPT_DIR, "*.pt"))):
        tag = os.path.basename(path)[:-3]
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        m = AnchorGNNLocalizer(ck["in_dim"], ck["edge_dim"], ck["H"], ck["enc_layers"],
                               ck["refine_steps"], ck["variant"], C.HUBER_DELTA).to(DEVICE)
        m.load_state_dict(ck["state"]); m.eval()
        models[VARIANT_NAMES.get(tag, tag)] = m
    return models


def run_gnn(model, graphs, bs=64):
    """One forward pass per batch; time is reported per graph."""
    out, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            chunk = graphs[i:i + bs]
            B = collate(chunk, DEVICE)
            t0 = time.perf_counter(); x, _ = model(B, T=None if model.T == 0 else C.TEST_REFINE_STEPS)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t += time.perf_counter() - t0
            out += to_original_units(B, x)
    return out, t / len(graphs)


def score_rows(g, method, X, secs):
    return dict(scenario=g["scenario"], missing=int(round(100 * g["missing_nominal"])), n=g["n"],
                graph_id=g["graph_id"], method=method, rmse=coord_rmse(g, X),
                rmse_certified=coord_rmse(g, X, g["certified"]), rel_edm=rel_edm_error(g, X),
                seconds=secs, globally_rigid=bool(g["globally_rigid"]),
                frac_certified=float(g["certified"].mean()), ba_m=g.get("ba_m"))


def evaluate_set(graphs, models, baselines, polish=True, label=""):
    rows, preds = [], {}
    for name, m in models.items():
        P, sec = run_gnn(m, graphs)
        preds[name] = P
        rows += [score_rows(g, name, X, sec) for g, X in zip(graphs, P)]
        if polish and name == OURS:
            # hybrid: our output as the starting point of a classical robust solver
            P2 = []
            for g, X in zip(graphs, P):
                t0 = time.perf_counter(); Y = BL.stress_lbfgs(g, X, robust=True)
                P2.append(Y); rows.append(score_rows(g, OURS + " + Huber polish", Y,
                                                     sec + time.perf_counter() - t0))
            preds[OURS + " + Huber polish"] = P2
    for name, fn in baselines.items():
        t0 = time.time(); P = []
        for k, g in enumerate(graphs):
            X, sec = BL.timed(fn, g); P.append(X)
            rows.append(score_rows(g, name, X, sec))
        preds[name] = P
        print(f"  {label} {name:42s} {time.time() - t0:7.1f}s")
    return pd.DataFrame(rows), preds


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
STYLE = {OURS: dict(color="crimson", lw=2.6, marker="o"),
         OURS + " + Huber polish": dict(color="darkred", lw=2.0, marker="s", ls="--")}


PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#17becf", "#bcbd22", "#393b79"]
FLOOR = 1e-4     # errors below this are numerically exact; drawn on the floor


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


def main():
    import sys
    if "--plot-only" in sys.argv:          # redraw figures from saved results, no re-running
        main_plots(pd.read_csv(os.path.join(C.RESULTS_DIR, "all_results.csv")))
        print("figures redrawn"); return
    print(f"STEP 3: evaluation on {DEVICE}")
    models = load_models()
    if not models:
        raise SystemExit("No checkpoints found -- run step2_train.py first.")
    print("  models:", list(models))
    baselines = BL.get_baselines(C)
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
