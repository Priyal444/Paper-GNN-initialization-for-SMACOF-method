"""
diagnose_seed1.py -- Multi-Seed Diagnostic Tool.
Tracks layer gradient norms, gate weight saturation, alpha parameter,
and step update norms across Epochs 1 to 7 using the exact 120-epoch schedule.
Allows running across multiple seeds (e.g. 0, 1, 2) to test reproducibility of Seed 1 collapse.
"""
import os, sys, pickle, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import config as C
from model import AnchorGNNLocalizer, collate, supervised_loss, to_original_units
from common import coord_rmse

def run_seed_diagnostic(seed, train, val, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    print(f"\n" + "=" * 95)
    print(f"   DIAGNOSTIC RUN: SEED {seed} (Epochs 1-7 using 120-epoch schedule on {device})")
    print("=" * 95)
    print(" EPOCH | LR        | TRAIN LOSS | VAL RMSE | ALPHA  | W_MEAN | W_MIN    | W_MAX  | % AT FLOOR | GRAD NORM (MAX)")
    print("-" * 95)

    model = AnchorGNNLocalizer(
        in_dim=train[0]["x"].shape[1],
        edge_dim=train[0]["edge_attr"].shape[1],
        H=C.HIDDEN,
        enc_layers=C.ENC_LAYERS,
        refine_steps=C.REFINE_STEPS,
        variant="learned",
        huber_delta=C.HUBER_DELTA
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    batch_size = C.BATCH_GRAPHS
    steps_per_epoch = int(np.ceil(len(train) / batch_size))
    total_steps = 120 * steps_per_epoch
    warmup_steps = min(3 * steps_per_epoch, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.02, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for ep in range(1, 8):
        model.train()
        random.shuffle(train)
        tot_loss = 0.0
        w_stats = []
        grad_norms = []

        for i in range(0, len(train), batch_size):
            B = collate(train[i:i + batch_size], device)
            x, xA, w_last, _ = model(B, return_weights=True)
            loss = supervised_loss(x, B, C.HUBER_DELTA) + C.AUX_WEIGHT * supervised_loss(xA, B, C.HUBER_DELTA)

            opt.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()

            tot_loss += loss.item()
            if w_last is not None:
                w_stats.append(w_last.detach().cpu())

        model.eval()
        val_errs_cert = []
        with torch.no_grad():
            for i in range(0, len(val), 64):
                chunk = val[i:i + 64]
                B_val = collate(chunk, device)
                x_val, _ = model(B_val, T=C.TEST_REFINE_STEPS)
                for g, X in zip(chunk, to_original_units(B_val, x_val)):
                    val_errs_cert.append(coord_rmse(g, X, g.get("certified")))

        val_rmse = float(np.nanmean(val_errs_cert))
        current_lr = opt.param_groups[0]["lr"]
        avg_loss = tot_loss / steps_per_epoch
        alpha_val = torch.sigmoid(model.step.alpha_logit).item()

        if w_stats:
            all_w = torch.cat(w_stats)
            w_mean = all_w.mean().item()
            w_min = all_w.min().item()
            w_max = all_w.max().item()
            floor_pct = (all_w <= 1.05e-3).float().mean().item() * 100.0
        else:
            w_mean, w_min, w_max, floor_pct = 0, 0, 0, 0

        avg_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)

        print(f" {ep:5d} | {current_lr:.2e} | {avg_loss:10.5f} | {val_rmse:8.3f} | {alpha_val:.4f} | "
              f"{w_mean:6.4f} | {w_min:8.2e} | {w_max:6.4f} | {floor_pct:9.1f}% | {avg_grad:6.2f} (max {max_grad:5.1f})")

    print("-" * 95)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to diagnose (default: 0 1 2)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting Multi-Seed Diagnostic across Seeds: {args.seeds} on {device}...")

    with open(os.path.join(C.DATA_DIR, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(C.DATA_DIR, "val.pkl"), "rb") as f:
        val = pickle.load(f)

    for g in train + val:
        if "y_norm" not in g:
            g["y_norm"] = (((g["true_xy"] - g["center"]) / g["scale"]) @ g.get("rot", np.eye(2, dtype=np.float32))).astype(np.float32)
        if "anchor_mask" not in g:
            am = np.zeros(g["n"], bool)
            am[g["anchors"]] = True
            g["anchor_mask"] = am

    for s in args.seeds:
        run_seed_diagnostic(s, train, val, device)

    print("\n[*] Diagnostic completed. Compare seed trajectories to verify whether Seed 1 failure replicates.\n")

if __name__ == "__main__":
    main()
