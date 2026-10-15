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


def evaluate(model, graphs, device, bs=64):
    model.eval(); errs = []
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
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load("train.pkl"), load("val.pkl")
    tag = args.variant + ("" if args.loss == "supervised" else "_selfsup")
    print(f"STEP 2: training '{tag}' on {len(train)} graphs ({device})")

    model = AnchorGNNLocalizer(in_dim=train[0]["x"].shape[1], edge_dim=train[0]["edge_attr"].shape[1],
                               H=C.HIDDEN, enc_layers=C.ENC_LAYERS, refine_steps=C.REFINE_STEPS,
                               variant=args.variant, huber_delta=C.HUBER_DELTA).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(train) / C.BATCH_GRAPHS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, total_steps=args.epochs * steps_per_epoch)
    loss_fn = supervised_loss if args.loss == "supervised" else selfsup_stress_loss

    best, log = np.inf, []
    ckpt = os.path.join(C.CKPT_DIR, f"{tag}.pt")
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train); t0 = time.time(); tot = 0.0
        for i in range(0, len(train), C.BATCH_GRAPHS):
            B = collate(train[i:i + C.BATCH_GRAPHS], device)
            x, xA = model(B)
            # deep supervision: final output + (weighted) encoder output
            loss = loss_fn(x, B, C.HUBER_DELTA) + C.AUX_WEIGHT * loss_fn(xA, B, C.HUBER_DELTA)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step(); sched.step(); tot += loss.item()
        v_mean, v_med = evaluate(model, val, device)
        log.append(dict(epoch=ep, train_loss=tot / steps_per_epoch, val_rmse_mean=v_mean, val_rmse_median=v_med))
        flag = ""
        if v_mean < best:
            best = v_mean; flag = "  *saved*"
            torch.save(dict(state=model.state_dict(), variant=args.variant, loss=args.loss,
                            in_dim=train[0]["x"].shape[1], edge_dim=train[0]["edge_attr"].shape[1],
                            H=C.HIDDEN, enc_layers=C.ENC_LAYERS, refine_steps=C.REFINE_STEPS), ckpt)
        print(f"  epoch {ep:3d} | loss {tot / steps_per_epoch:.4f} | val RMSE mean {v_mean:.3f} "
              f"median {v_med:.3f} | {time.time() - t0:.0f}s{flag}")

    df = pd.DataFrame(log); df.to_csv(os.path.join(C.RESULTS_DIR, f"train_log_{tag}.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.epoch, df.train_loss); ax[0].set_yscale("log"); ax[0].set_title("training loss")
    ax[1].plot(df.epoch, df.val_rmse_mean, label="mean"); ax[1].plot(df.epoch, df.val_rmse_median, label="median")
    ax[1].set_title("validation coordinate RMSE"); ax[1].legend()
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.RESULTS_DIR, f"train_curve_{tag}.png"), dpi=150)
    print(f"STEP 2 done. best val RMSE {best:.3f} -> {ckpt}")


if __name__ == "__main__":
    main()
