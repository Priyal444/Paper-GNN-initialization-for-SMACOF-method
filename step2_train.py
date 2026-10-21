"""
STEP 2 -- train the proposed GNN ONCE on many graphs.

After training, the model is FROZEN. In step 3 it localises every new test
graph with one forward pass (no per-graph training) -- this is what
"the GNN learned the geometry" actually means.

Run:
    python step2_train.py                       # proposed model
    python step2_train.py --variant uniform     # ablation: w_ij = 1 (plain dwMDS steps)
    python step2_train.py --variant huber       # ablation: classical robust weights
    python step2_train.py --variant encoder_only
    python step2_train.py --loss selfsup        # no ground truth at all (stress loss)

Transfer experiment (train on some graph types, test on all in step 3 --transfer):
    python step2_train.py --train-on rgg        # one type
    python step2_train.py --train-on rgg+ba     # two types
    python step2_train.py --train-on all        # all three
"""
import argparse, os, pickle, time, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from model import (AnchorGNNLocalizer, collate, supervised_loss, selfsup_stress_loss,
                   to_original_units)
from common import coord_rmse


def load(name):
    with open(os.path.join(C.DATA_DIR, name), "rb") as f:
        return pickle.load(f)


def load_pools(key):
    """N_TRAIN / N_VAL graphs split equally over the types in `key` (same total for every key)."""
    types = C.train_types(key)
    train, val = [], []
    for gt in types:
        train += load(f"train_pool_{gt}.pkl")[:C.N_TRAIN // len(types)]
        val += load(f"val_pool_{gt}.pkl")[:C.N_VAL // len(types)]
    return train, val


def evaluate(model, graphs, device, bs=64, model1=None, dev1=None):
    model.eval(); errs = []
    if model1 is not None and dev1 is not None and len(graphs) >= 2:
        import concurrent.futures
        model1.eval()
        mid = len(graphs) // 2
        g0, g1 = graphs[:mid], graphs[mid:]
        
        def run_chunk(m, chk, dev):
            local_errs = []
            with torch.no_grad():
                for i in range(0, len(chk), bs):
                    b_chk = chk[i:i + bs]
                    B = collate(b_chk, dev)
                    x, _ = m(B, T=None if m.T == 0 else C.TEST_REFINE_STEPS)
                    for g, X in zip(b_chk, to_original_units(B, x)):
                        local_errs.append(coord_rmse(g, X))
            return local_errs
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f0 = pool.submit(run_chunk, model, g0, device)
            f1 = pool.submit(run_chunk, model1, g1, dev1)
            errs = f0.result() + f1.result()
    else:
        with torch.no_grad():
            for i in range(0, len(graphs), bs):
                chunk = graphs[i:i + bs]
                B = collate(chunk, device)
                x, _ = model(B, T=None if model.T == 0 else C.TEST_REFINE_STEPS)
                for g, X in zip(chunk, to_original_units(B, x)):
                    errs.append(coord_rmse(g, X))
    return float(np.nanmean(errs)), float(np.nanmedian(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=C.VARIANT)
    ap.add_argument("--loss", default=C.LOSS_MODE, choices=["supervised", "selfsup"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--train-on", default=None,
                    help="transfer experiment: graph types to train on, e.g. rgg, rgg+ba, all")
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        print(f"  Dual-GPU acceleration active: utilizing {num_gpus} GPUs (cuda:0, cuda:1)", flush=True)
        for g_idx in range(num_gpus):
            print(f"    GPU {g_idx}: {torch.cuda.get_device_name(g_idx)}", flush=True)
        device = "cuda:0"
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        multi_gpu = True
    elif num_gpus == 1:
        device = "cuda:0"
        dev0, dev1 = torch.device("cuda:0"), None
        multi_gpu = False
        print(f"  Single GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = "cpu"
        dev0, dev1 = torch.device("cpu"), None
        multi_gpu = False
        print("  CPU  (no GPU found -- on Colab/Kaggle: Enable GPU accelerator)", flush=True)

    tag = args.variant + ("" if args.loss == "supervised" else "_selfsup")
    if args.train_on:
        train, val = load_pools(args.train_on)
        tag += f"__train-{args.train_on}"
    else:
        train, val = load("train.pkl"), load("val.pkl")
    print(f"STEP 2: training '{tag}' on {len(train)} graphs ({'Dual-GPU' if multi_gpu else device})", flush=True)

    # Pre-cache normalized coordinates and anchor masks to eliminate CPU bottleneck
    for g in train + val:
        if "y_norm" not in g:
            g["y_norm"] = ((g["true_xy"] - g["center"]) / g["scale"]).astype(np.float32)
        if "anchor_mask" not in g:
            am = np.zeros(g["n"], bool)
            am[g["anchors"]] = True
            g["anchor_mask"] = am

    model = AnchorGNNLocalizer(in_dim=train[0]["x"].shape[1], edge_dim=train[0]["edge_attr"].shape[1],
                               H=C.HIDDEN, enc_layers=C.ENC_LAYERS, refine_steps=C.REFINE_STEPS,
                               variant=args.variant, huber_delta=C.HUBER_DELTA).to(dev0)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    model1 = None
    if multi_gpu:
        import copy
        model1 = copy.deepcopy(model).to(dev1)
        s0 = torch.cuda.Stream(device=dev0)
        s1 = torch.cuda.Stream(device=dev1)

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    batch_size = C.BATCH_GRAPHS * 2 if multi_gpu else C.BATCH_GRAPHS
    steps_per_epoch = int(np.ceil(len(train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, total_steps=args.epochs * steps_per_epoch)
    loss_fn = supervised_loss if args.loss == "supervised" else selfsup_stress_loss

    best, log = np.inf, []
    ckpt = os.path.join(C.CKPT_DIR, f"{tag}.pt")
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train); t0 = time.time(); tot = 0.0
        for i in range(0, len(train), batch_size):
            b_idx = i // batch_size + 1
            if multi_gpu:
                chunk0 = train[i:i + C.BATCH_GRAPHS]
                chunk1 = train[i + C.BATCH_GRAPHS:i + batch_size]
                B0 = collate(chunk0, dev0)
                if chunk1:
                    B1 = collate(chunk1, dev1)
                    model1.load_state_dict(model.state_dict())
                    with torch.cuda.stream(s0):
                        x0, xA0 = model(B0)
                        loss0 = loss_fn(x0, B0, C.HUBER_DELTA) + C.AUX_WEIGHT * loss_fn(xA0, B0, C.HUBER_DELTA)
                        opt.zero_grad()
                        loss0.backward()
                    with torch.cuda.stream(s1):
                        x1, xA1 = model1(B1)
                        loss1 = loss_fn(x1, B1, C.HUBER_DELTA) + C.AUX_WEIGHT * loss_fn(xA1, B1, C.HUBER_DELTA)
                        model1.zero_grad()
                        loss1.backward()
                    torch.cuda.synchronize(dev0)
                    torch.cuda.synchronize(dev1)
                    for p0, p1 in zip(model.parameters(), model1.parameters()):
                        if p1.grad is not None:
                            p0.grad.add_(p1.grad.to(dev0))
                            p0.grad.mul_(0.5)
                    step_loss = (loss0.item() + loss1.item()) / 2.0
                else:
                    x0, xA0 = model(B0)
                    loss0 = loss_fn(x0, B0, C.HUBER_DELTA) + C.AUX_WEIGHT * loss_fn(xA0, B0, C.HUBER_DELTA)
                    opt.zero_grad()
                    loss0.backward()
                    step_loss = loss0.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step(); sched.step(); tot += step_loss
            else:
                B = collate(train[i:i + C.BATCH_GRAPHS], device)
                x, xA = model(B)
                loss = loss_fn(x, B, C.HUBER_DELTA) + C.AUX_WEIGHT * loss_fn(xA, B, C.HUBER_DELTA)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step(); sched.step(); tot += loss.item()
                step_loss = loss.item()

            if b_idx % max(1, steps_per_epoch // 4) == 0 or b_idx == steps_per_epoch:
                print(f"    epoch {ep:2d}/{args.epochs} | batch {b_idx:3d}/{steps_per_epoch} | batch loss: {step_loss:.4f}", flush=True)

        v_mean, v_med = evaluate(model, val, dev0, model1=model1, dev1=dev1)
        log.append(dict(epoch=ep, train_loss=tot / steps_per_epoch, val_rmse_mean=v_mean, val_rmse_median=v_med))
        flag = ""
        if v_mean < best:
            best = v_mean; flag = "  *saved*"
            torch.save(dict(state=model.state_dict(), variant=args.variant, loss=args.loss,
                            train_on=args.train_on,
                            in_dim=train[0]["x"].shape[1], edge_dim=train[0]["edge_attr"].shape[1],
                            H=C.HIDDEN, enc_layers=C.ENC_LAYERS, refine_steps=C.REFINE_STEPS), ckpt)
        print(f"  epoch {ep:3d} | loss {tot / steps_per_epoch:.4f} | val RMSE mean {v_mean:.3f} "
              f"median {v_med:.3f} | {time.time() - t0:.0f}s{flag}", flush=True)

    df = pd.DataFrame(log); df.to_csv(os.path.join(C.RESULTS_DIR, f"train_log_{tag}.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.epoch, df.train_loss); ax[0].set_yscale("log"); ax[0].set_title("training loss")
    ax[1].plot(df.epoch, df.val_rmse_mean, label="mean"); ax[1].plot(df.epoch, df.val_rmse_median, label="median")
    ax[1].set_title("validation coordinate RMSE"); ax[1].legend()
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, f"train_curve_{tag}.png"), dpi=150)
    print(f"STEP 2 done. best val RMSE {best:.3f} -> {ckpt}", flush=True)



if __name__ == "__main__":
    main()
