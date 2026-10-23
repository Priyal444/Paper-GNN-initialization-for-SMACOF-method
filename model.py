"""
model.py -- the proposed network:  AnchorGNN encoder  +  learned unrolled SMACOF.

Pure PyTorch (no torch_geometric needed): message passing uses index_add_.

    features --> [ENCODER: L edge-aware MPNN layers] --> h_i, e_ij, x_A
                                                          |
                 [REFINER: T unrolled learned-SMACOF steps, anchors pinned]
                                                          v
                                                       x_T  (prediction)

Variants (for the ablation):
    "learned"      : weights w_ij from an MLP           (proposed)
    "uniform"      : w_ij = 1   -> exactly Jacobi dwMDS / SMACOF, no learning
    "huber"        : w_ij = min(1, delta/|r_ij|) -> classical robust IRLS
    "encoder_only" : T = 0, prediction = x_A
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def scatter_sum(src, index, n):
    out = torch.zeros((n,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_mean(src, index, n):
    s = scatter_sum(src, index, n)
    c = scatter_sum(torch.ones_like(src[:, :1]), index, n).clamp_min(1.0)
    return s / c


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


# -----------------------------------------------------------------------------
# batching: many graphs -> one big disconnected graph
# -----------------------------------------------------------------------------
def collate(graphs, device="cpu"):
    xs, x0s, eis, eas, des, ys, am, batch, cen, sca, rots = [], [], [], [], [], [], [], [], [], [], []
    off = 0
    for b, g in enumerate(graphs):
        n = g["n"]
        R = g.get("rot", np.eye(2, dtype=np.float32))
        xs.append(g["x"]); x0s.append(g["x0"])
        eis.append(g["edge_index"] + off); eas.append(g["edge_attr"]); des.append(g["d_edge"])
        if "y_norm" in g:
            ys.append(g["y_norm"])
        else:
            ys.append(((g["true_xy"] - g["center"]) / g["scale"]) @ R)  # E(2)-canonicalised GT
        if "anchor_mask" in g:
            am.append(g["anchor_mask"])
        else:
            a = np.zeros(n, bool); a[g["anchors"]] = True; am.append(a)
        batch.append(np.full(n, b)); cen.append(g["center"]); sca.append(g["scale"]); rots.append(R)
        off += n
    T = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)
    return dict(
        x=T(xs), x0=T(x0s), edge_index=torch.as_tensor(np.concatenate(eis, 1), device=device),
        edge_attr=T(eas), d_edge=T(des), y=T(ys), anchor_mask=T(am, torch.bool),
        batch=T(batch, torch.long), center=torch.as_tensor(np.stack(cen), device=device),
        scale=torch.as_tensor(np.array(sca), device=device),
        rot=torch.as_tensor(np.stack(rots), dtype=torch.float32, device=device),
        n_graphs=len(graphs), sizes=[g["n"] for g in graphs],
    )


# -----------------------------------------------------------------------------
# Encoder layer: edge-aware message passing (MPNN / GINE-style)
# -----------------------------------------------------------------------------
class MPLayer(nn.Module):
    """
    m_ij = MLP_m([h_i, h_j, e_ij])                 message along edge j -> i
    e_ij <- LN(e_ij + m_ij)                         edge state update
    h_i  <- LN(h_i + MLP_u([h_i, mean_j m_ij]))     node state update
    Unlike SAGEConv, the measured distance enters EVERY message via e_ij.
    """
    def __init__(self, H):
        super().__init__()
        self.msg = mlp(3 * H, H, H)
        self.upd = mlp(2 * H, H, H)
        self.ln_h, self.ln_e = nn.LayerNorm(H), nn.LayerNorm(H)

    def forward(self, h, e, ei):
        src, dst = ei
        m = self.msg(torch.cat([h[dst], h[src], e], -1))
        e = self.ln_e(e + m)
        agg = scatter_mean(m, dst, h.size(0))
        h = self.ln_h(h + self.upd(torch.cat([h, agg], -1)))
        return h, e


# -----------------------------------------------------------------------------
# Refiner step: one Guttman / dwMDS majorisation step with learned weights
# -----------------------------------------------------------------------------
class LearnedSMACOFStep(nn.Module):
    """
    For every node i (anchors are re-pinned afterwards):
        dhat_ij = ||x_i - x_j||,   r_ij = d_ij - dhat_ij
        t_ij    = x_j + d_ij * (x_i - x_j) / dhat_ij       (where edge ij wants x_i)
        w_ij    = softplus( MLP_w([h_i, h_j, e_ij, d_ij, dhat_ij, |r_ij|]) )
        x_i^+   = sum_j w_ij t_ij / sum_j w_ij             (Guttman transform, local)
        x_i     <- x_i + alpha (x_i^+ - x_i)               (damped Jacobi update)
    """
    def __init__(self, H, variant="learned", huber_delta=0.05):
        super().__init__()
        self.variant, self.delta = variant, huber_delta
        self.w_static = nn.Linear(3 * H, H)
        self.w_dyn = nn.Linear(3, H, bias=False)
        self.w_out = nn.Linear(H, 1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))   # alpha = 0.5 at init

    def prepare(self, h, e, ei):
        if self.variant != "learned":
            return None
        src, dst = ei
        return self.w_static(torch.cat([h[dst], h[src], e], -1))

    def weights(self, static, d, dhat, r):
        if self.variant == "uniform":
            return torch.ones_like(d)
        if self.variant == "huber":
            return torch.clamp(self.delta / r.abs().clamp_min(EPS), max=1.0)
        z = F.silu(static + self.w_dyn(torch.stack([d, dhat, r.abs()], -1)))
        return F.softplus(self.w_out(z)).squeeze(-1) + 1e-4

    def forward(self, x, static, ei, d, anchor_mask, x_anchor):
        src, dst = ei
        diff = x[dst] - x[src]
        dhat = diff.norm(dim=-1).clamp_min(EPS)
        r = d - dhat
        w = self.weights(static, d, dhat, r)
        target = x[src] + d[:, None] * diff / dhat[:, None]
        num = scatter_sum(w[:, None] * target, dst, x.size(0))
        den = scatter_sum(w[:, None], dst, x.size(0)).clamp_min(EPS)
        alpha = torch.sigmoid(self.alpha_logit) if self.variant == "learned" else 0.5
        x_new = x + alpha * (num / den - x)
        return torch.where(anchor_mask[:, None], x_anchor, x_new), w


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------
class AnchorGNNLocalizer(nn.Module):
    def __init__(self, in_dim=19, edge_dim=3, H=128, enc_layers=6, refine_steps=10,
                 variant="learned", huber_delta=0.05):
        super().__init__()
        self.variant = variant
        self.T = 0 if variant == "encoder_only" else refine_steps
        self.node_in = mlp(in_dim, H, H)
        self.edge_in = mlp(edge_dim, H, H)
        self.layers = nn.ModuleList([MPLayer(H) for _ in range(enc_layers)])
        self.head = mlp(H, H, 2)
        # ONE shared step -> it is an iterative algorithm, so T can change at test time
        self.step = LearnedSMACOFStep(H, variant, huber_delta)

    def forward(self, B, T=None, return_all=False):
        x0, am, ei, d = B["x0"], B["anchor_mask"], B["edge_index"], B["d_edge"]
        x_anchor = x0                                   # anchors' x0 == true anchor xy
        h = self.node_in(B["x"])
        e = self.edge_in(B["edge_attr"])
        for L in self.layers:
            h, e = L(h, e, ei)
        xA = torch.where(am[:, None], x_anchor, x0 + self.head(h))   # stage-A estimate
        x, traj = xA, [xA]
        static = self.step.prepare(h, e, ei)
        for _ in range(self.T if T is None else T):
            x, _ = self.step(x, static, ei, d, am, x_anchor)
            traj.append(x)
        return (x, xA, traj) if return_all else (x, xA)


def to_original_units(B, x_norm):
    """E(2)-canonicalised prediction -> original coordinates, list of (n_i, 2) arrays."""
    out, off = [], 0
    for b, n in enumerate(B["sizes"]):
        R = B["rot"][b]
        xb = (x_norm[off:off + n] @ R.T) * B["scale"][b] + B["center"][b]
        out.append(xb.detach().cpu().numpy()); off += n
    return out


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def supervised_loss(x, B, delta):
    """Huber on coordinates of NON-anchor nodes (normalised units)."""
    m = ~B["anchor_mask"]
    return F.huber_loss(x[m], B["y"][m], delta=delta)


def selfsup_stress_loss(x, B, delta):
    """Asymmetric NLOS-aware Huber stress loss on observed edges:
    Since NLOS biases are positive (d_edge >= d_true), dhat > d_edge is penalized
    quadratically, while dhat < d_edge uses robust Huber to discount positive NLOS outliers."""
    src, dst = B["edge_index"]
    dhat = (x[dst] - x[src]).norm(dim=-1).clamp_min(EPS)
    res = dhat - B["d_edge"]
    quad_mask = res >= 0.0
    loss_pos = 0.5 * (res ** 2)
    loss_neg = F.huber_loss(dhat, B["d_edge"], delta=delta, reduction="none")
    return torch.where(quad_mask, loss_pos, loss_neg).mean()
