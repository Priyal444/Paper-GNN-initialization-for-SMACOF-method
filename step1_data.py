"""
STEP 1 -- build the datasets.

    train / val : many SMALL graphs, mixed everything (n in 30..80, missing in
                  10..90 %, noise model, graph type) -> the GNN learns the
                  general skill, not one graph.
    test        : TEST_PER_LEVEL graphs for every (scenario, missing level).
                  EVERY method (ours and all baselines) is scored on these
                  exact same graphs in step 3.
    generalise  : bigger graphs (n = 100, 200) never seen in training.
    ba          : Barabasi-Albert graphs, m = 1..5 (rigidity threshold).
    pools       : (RUN_TRANSFER) one train + val pool per graph type, for the
                  train-on-some-types / test-on-all transfer experiment.

Each graph stores what methods may use (edges, d_obs, anchors, anchor_xy,
features) and, separately, true_xy (used ONLY for the training loss on
synthetic training graphs and for scoring).

Run:  python step1_data.py
"""
import pickle, time, os, glob
import numpy as np
import config as C
from common import make_graph, ba_reachable, dump_run_config


def save(obj, name):
    path = os.path.join(C.DATA_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  saved {len(obj):5d} graphs -> {path}")



def mixed_set(rng, count, types=None, label=""):
    types = types or C.GRAPH_TYPES
    out = []
    noise_names = list(C.NOISE_MODELS)
    log_step = max(1, count // 5)
    t0 = time.time()
    while len(out) < count:
        n = int(rng.integers(C.TRAIN_N_RANGE[0], C.TRAIN_N_RANGE[1] + 1))
        gt = types[rng.integers(len(types))]
        lo, hi = C.TRAIN_MISSING
        if gt == "ba":                     # BA cannot go below ~50 % missing (see common.ba_m_for)
            lo = max(lo, 0.5)
        mp = float(rng.uniform(lo, hi))
        nz = noise_names[rng.integers(len(noise_names))]
        g = make_graph(rng, n, gt, mp, C.NOISE_MODELS[nz], C.N_ANCHORS, C.COORD_RANGE)
        g["noise"] = nz
        out.append(g)
        if len(out) % log_step == 0 or len(out) == count:
            print(f"    ... {label} generated {len(out)}/{count} graphs ({len(out)/count*100:.0f}%) in {time.time() - t0:.1f}s", flush=True)
    return out


def scenario_set(rng, n, scenarios, levels, per_level):
    out = []
    for gt, nz in scenarios:
        for mp in levels:
            if gt == "ba" and not ba_reachable(n, mp):
                print(f"  skipping ba/{nz} at {mp:.0%} missing: BA needs >= ~50 % missing at n = {n}", flush=True)
                continue
            for k in range(per_level):
                g = make_graph(rng, n, gt, mp, C.NOISE_MODELS[nz], C.N_ANCHORS, C.COORD_RANGE)
                g.update(noise=nz, scenario=f"{gt}/{nz}", graph_id=k)
                out.append(g)
            print(f"    test scenario {gt:6s} | noise: {nz:5s} | missing: {mp:.0%} ({len(out)} test graphs generated)", flush=True)
    return out


def summary(graphs, title):
    import pandas as pd
    df = pd.DataFrame([dict(scenario=g.get("scenario", "mixed"), missing=g["missing_nominal"],
                            edges=len(g["edges"]), rigid=g["globally_rigid"],
                            certified=g["certified"].mean()) for g in graphs])
    print(f"\n  {title}: how many graphs are even uniquely solvable?", flush=True)
    df["missing_actual"] = [g["missing_actual"] for g in graphs]
    print(df.groupby(["scenario", "missing"]).agg(edges=("edges", "mean"),
          missing_actual=("missing_actual", "mean"),
          globally_rigid=("rigid", "mean"), certified_nodes=("certified", "mean"))
          .round(2).to_string(), flush=True)


def make_extended_sets(rng):
    if not getattr(C, "RUN_EXTENDED_EXPERIMENTS", False) and os.environ.get("RUN_EXTENDED") != "1":
        return
    print("  Generating extended experiment test sets...", flush=True)
    k_vals = getattr(C, "ANCHOR_COUNTS", [3, 4, 5, 6, 7, 8])
    n_per_k = getattr(C, "EXTENDED_TEST_PER_LEVEL", 20)

    # 1. Anchor sweep graphs: varying K, N=50, 50% missing
    anchor_graphs = []
    for k in k_vals:
        for gt, nz in [("rgg", "nlos"), ("random", "nlos")]:
            for i in range(n_per_k):
                g = make_graph(rng, C.N_NODES, gt, 0.50, C.NOISE_MODELS[nz], n_anchors=k)
                g.update(scenario=f"{gt}/{nz}", noise=nz, anchor_k=k, graph_id=len(anchor_graphs))
                anchor_graphs.append(g)
    save(anchor_graphs, "anchor_sweep_test.pkl")
    print(f"    Anchor sweep test set generated ({len(anchor_graphs)} graphs)", flush=True)

    # 2. NLOS sweep graphs: varying p_nlos, N=50, 50% missing, 3 anchors
    nlos_graphs = []
    p_vals = getattr(C, "NLOS_PROBABILITIES", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for p in p_vals:
        for gt in ["rgg", "random"]:
            nz = dict(sigma=0.02, p_nlos=float(p), nlos_max=4.0)
            for i in range(n_per_k):
                g = make_graph(rng, C.N_NODES, gt, 0.50, nz, n_anchors=3)
                g.update(scenario=f"{gt}/nlos", p_nlos=float(p), graph_id=len(nlos_graphs))
                nlos_graphs.append(g)
    save(nlos_graphs, "nlos_sweep_test.pkl")
    print(f"    NLOS sweep test set generated ({len(nlos_graphs)} graphs)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    dump_run_config(step_name="step1")
    # Clear baseline evaluation caches if regenerating data
    for _bf in glob.glob(os.path.join(C.DATA_DIR, "baseline_cache*.pkl")):
        try: os.remove(_bf)
        except Exception: pass

    print("STEP 1: generating datasets", flush=True)

    print("  [1/4] Generating training set...", flush=True)
    save(mixed_set(np.random.default_rng(C.SEED), C.N_TRAIN, label="train"), "train.pkl")
    print("  [2/4] Generating validation set...", flush=True)
    save(mixed_set(np.random.default_rng(C.SEED + 1), C.N_VAL, label="val"), "val.pkl")

    if C.RUN_TRANSFER:
        print("  Generating transfer pools...", flush=True)
        for i, gt in enumerate(C.GRAPH_TYPES):
            save(mixed_set(np.random.default_rng(C.SEED + 10 + i), C.N_TRAIN, [gt], label=f"train_{gt}"), f"train_pool_{gt}.pkl")
            save(mixed_set(np.random.default_rng(C.SEED + 20 + i), C.N_VAL, [gt], label=f"val_{gt}"), f"val_pool_{gt}.pkl")

    print("  [3/4] Generating test set...", flush=True)
    test = scenario_set(np.random.default_rng(C.SEED + 2), C.N_NODES,
                        C.TEST_SCENARIOS, C.MISSING_LEVELS, C.TEST_PER_LEVEL)
    save(test, "test.pkl")
    summary(test, "TEST SET")

    if C.RUN_GENERALISATION:
        print("  [4/4] Generating generalisation test sets...", flush=True)
        for n in C.GENERALISE_SIZES:
            gen = scenario_set(np.random.default_rng(C.SEED + 3 + n), n,
                               [C.GENERALISE_SCENARIO], C.GENERALISE_LEVELS, C.TEST_PER_LEVEL)
            save(gen, f"test_n{n}.pkl")

    if C.RUN_BA_EXPERIMENT:
        print("  Generating Barabasi-Albert test set...", flush=True)
        rng = np.random.default_rng(C.SEED + 4)
        ba = []
        for m in C.BA_M_VALUES:
            for k in range(C.BA_PER_M):
                g = make_graph(rng, C.N_NODES, "ba", 0.0, C.NOISE_MODELS["clean"],
                               C.N_ANCHORS, C.COORD_RANGE, ba_m=m)
                g.update(noise="clean", scenario=f"ba/m={m}", ba_m=m, graph_id=k)
                ba.append(g)
            print(f"    BA m={m} generated ({len(ba)} graphs)", flush=True)
        save(ba, "test_ba.pkl")

    make_extended_sets(np.random.default_rng(C.SEED + 99))
    print(f"\nSTEP 1 done in {time.time() - t0:.0f}s", flush=True)

