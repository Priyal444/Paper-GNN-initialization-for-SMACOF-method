"""
baselines.py -- existing methods, all run on EXACTLY the same graphs.

Every function takes one graph dict g (from common.make_graph) and returns
an (n, 2) array of coordinates in ORIGINAL units. They use only
g["edges"], g["d_obs"], g["anchors"], g["anchor_xy"] -- never g["true_xy"].

    mds_map            Shang et al. 2003: geodesic completion + classical MDS + Procrustes
    mds_map_stress     MDS-MAP followed by full stress minimisation (L-BFGS)   <- strong
    mds_map_huber      same, robust Huber stress (classical NLOS-robust choice)
    dv_dwmds           DV-distance multilateration + 300 dwMDS iterations (w = 1)
                       = the classical algorithm our refiner unrolls, with NO learning
    sdp_biswas_ye      Biswas & Ye 2004 SDP relaxation (L1 slack), cvxpy
    yan_gcn            Yan et al. ICASSP 2021 GCN (per-graph, anchors are the labels)
    pergraph_sage      your previous approach: GraphSAGE trained per graph on stress
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from common import geodesic_matrix, multilaterate


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def classical_mds(D, d=2):
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)
    o = np.argsort(w)[::-1][:d]
    return V[:, o] * np.sqrt(np.clip(w[o], 0, None))


def procrustes_O2(X, anchors, A):
    """Best rotation OR reflection + translation mapping X[anchors] onto A."""
    S = X[anchors]
    sc, ac = S.mean(0), A.mean(0)
    U, _, Vt = np.linalg.svd((S - sc).T @ (A - ac))
    best, br = None, np.inf
    for sg in (1.0, -1.0):
        V = Vt.copy(); V[-1] *= sg
        R = V.T @ U.T
        r = np.linalg.norm((S - sc) @ R.T - (A - ac))
        if r < br:
            best, br = R, r
    return (X - sc) @ best.T + ac


def _pin(g, X):
    X = X.copy(); X[g["anchors"]] = g["anchor_xy"]; return X


def stress_lbfgs(g, X0, robust=False, max_iter=3000):
    """Minimise sum_edges rho(||x_i - x_j|| - d_ij) over NON-anchor nodes."""
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(float)
    I, J = E[:, 0], E[:, 1]
    free = np.ones(n, bool); free[g["anchors"]] = False
    Xfix = _pin(g, X0).astype(float)
    delta = 0.1 * np.median(d)

    def f(z):
        X = Xfix.copy(); X[free] = z.reshape(-1, 2)
        diff = X[I] - X[J]
        dh = np.sqrt((diff ** 2).sum(1) + 1e-12)
        r = dh - d
        if robust:
            a = np.abs(r)
            val = np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta)).sum()
            dr = np.where(a <= delta, r, delta * np.sign(r))
        else:
            val, dr = (r ** 2).sum(), 2 * r
        c = (dr / dh)[:, None] * diff
        G = np.zeros_like(X)
        np.add.at(G, I, c); np.add.at(G, J, -c)
        return val, G[free].ravel()

    z = minimize(f, Xfix[free].ravel(), jac=True, method="L-BFGS-B",
                 options={"maxiter": max_iter}).x
    X = Xfix.copy(); X[free] = z.reshape(-1, 2)
    return X


# -----------------------------------------------------------------------------
# baselines
# -----------------------------------------------------------------------------
def mds_map(g):
    D = geodesic_matrix(g["n"], g["edges"], g["d_obs"].astype(float))
    X = classical_mds(D)
    return _pin(g, procrustes_O2(X, g["anchors"], g["anchor_xy"]))


def mds_map_stress(g):
    return stress_lbfgs(g, mds_map(g), robust=False)


def mds_map_huber(g):
    return stress_lbfgs(g, mds_map(g), robust=True)


def dv_dwmds(g, iters=300, alpha=0.5):
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(float)
    G = geodesic_matrix(n, E, d)[:, g["anchors"]]
    X = _pin(g, multilaterate(g["anchor_xy"].astype(float), G))
    src, dst = np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]]
    dd = np.r_[d, d]
    deg = np.bincount(dst, minlength=n)[:, None]
    for _ in range(iters):
        diff = X[dst] - X[src]
        dh = np.linalg.norm(diff, axis=1).clip(1e-9)
        t = X[src] + dd[:, None] * diff / dh[:, None]
        num = np.zeros_like(X); np.add.at(num, dst, t)
        X = _pin(g, X + alpha * (num / deg - X))
    return X


def sdp_biswas_ye(g):
    """
    Z = [[I_2, X], [X^T, Y]] >= 0,  X = unknown positions (2 x m).
      unknown-unknown edge: Y_ii + Y_jj - 2 Y_ij      = d^2 + s_ij
      anchor-unknown edge : ||a||^2 - 2 a^T x_i + Y_ii = d^2 + s_ij
    minimise sum |s_ij|   (L1 slack = robust to outliers)
    """
    import cvxpy as cp
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(float)
    anchors = list(g["anchors"])
    is_a = np.zeros(n, bool); is_a[anchors] = True
    axy = np.zeros((n, 2)); axy[anchors] = g["anchor_xy"]
    unk = np.nonzero(~is_a)[0]
    pos = -np.ones(n, int); pos[unk] = np.arange(len(unk)) + 2   # index inside Z
    m = len(unk)
    Z = cp.Variable((m + 2, m + 2), PSD=True)
    dg = cp.diag(Z)
    uu = ~is_a[E[:, 0]] & ~is_a[E[:, 1]]                 # unknown-unknown edges
    au = is_a[E[:, 0]] ^ is_a[E[:, 1]]                    # anchor-unknown edges
    res = []
    if uu.any():
        P, Q = pos[E[uu, 0]], pos[E[uu, 1]]
        res.append(dg[P] + dg[Q] - 2 * Z[P, Q] - d[uu] ** 2)
    if au.any():
        Ea = E[au]
        a_node = np.where(is_a[Ea[:, 0]], Ea[:, 0], Ea[:, 1])
        u_node = np.where(is_a[Ea[:, 0]], Ea[:, 1], Ea[:, 0])
        K = pos[u_node]; a = axy[a_node]
        res.append((a ** 2).sum(1) - 2 * (cp.multiply(a[:, 0], Z[0, K]) + cp.multiply(a[:, 1], Z[1, K]))
                   + dg[K] - d[au] ** 2)
    cons = [Z[0, 0] == 1, Z[1, 1] == 1, Z[0, 1] == 0]
    prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.abs(r)) for r in res)), cons)
    try:
        prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-5, verbose=False)
        Xu = np.asarray(Z.value[:2, 2:]).T
    except Exception:
        return mds_map(g)
    X = np.zeros((n, 2)); X[unk] = Xu
    return _pin(g, X)


def yan_gcn(g, epochs=200, hidden=256, lr=0.01, dropout=0.5, seed=0):
    """
    Yan et al. 2021 (ICASSP), reproduced on our graphs:
      features = rows of the (thresholded = observed) distance matrix, row-normalised
      2-layer GCN with augmented normalised adjacency, trained per graph,
      loss = MSE on the ANCHOR nodes only (here: 3 labels).
    Paper used 2000 hidden units; 256 is used here for speed.
    """
    torch.manual_seed(seed)
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    Xd = np.zeros((n, n), np.float32)
    Xd[E[:, 0], E[:, 1]] = d; Xd[E[:, 1], E[:, 0]] = d
    Xf = Xd / np.maximum(np.linalg.norm(Xd, axis=1, keepdims=True), 1e-9)
    A = (Xd > 0).astype(np.float32) + np.eye(n, dtype=np.float32)
    Dm = 1.0 / np.sqrt(A.sum(1))
    Ah = torch.tensor(Dm[:, None] * A * Dm[None, :])
    Xf = torch.tensor(Xf)
    W1, W2 = nn.Linear(n, hidden), nn.Linear(hidden, 2)
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=lr)
    a = torch.tensor(g["anchors"]); ya = torch.tensor(g["anchor_xy"])
    def fwd(train):                       # R = A_h relu(A_h X W1) W2   (paper eq. 5)
        h = F.dropout(F.relu(W1(Ah @ Xf)), dropout, train)
        return Ah @ W2(h)
    for _ in range(epochs):
        opt.zero_grad(); loss = F.mse_loss(fwd(True)[a], ya); loss.backward(); opt.step()
    with torch.no_grad():
        return _pin(g, fwd(False).numpy())


class _SAGE(nn.Module):
    def __init__(self, H=64):
        super().__init__()
        self.l1s, self.l1n = nn.Linear(2, H), nn.Linear(2, H, bias=False)
        self.l2s, self.l2n = nn.Linear(H, H), nn.Linear(H, H, bias=False)
        self.out = nn.Linear(H, 2)
    def forward(self, x, An):
        h = F.relu(self.l1s(x) + self.l1n(An @ x))
        h = F.relu(self.l2s(h) + self.l2n(An @ h))
        return x + self.out(h)


def pergraph_sage(g, epochs=1000, lr=1e-3, seed=0):
    """Your previous pipeline (step1_corrected): MDS-MAP init -> GraphSAGE with a
    residual output, trained from scratch on THIS graph with Huber stress + anchors."""
    torch.manual_seed(seed)
    n, E = g["n"], g["edges"]
    x0 = torch.tensor(mds_map(g), dtype=torch.float32)
    A = np.zeros((n, n), np.float32); A[E[:, 0], E[:, 1]] = 1; A[E[:, 1], E[:, 0]] = 1
    An = torch.tensor(A / A.sum(1, keepdims=True))
    I, J = torch.tensor(E[:, 0]), torch.tensor(E[:, 1])
    d = torch.tensor(g["d_obs"]); s = d.mean()
    a = torch.tensor(g["anchors"]); ya = torch.tensor(g["anchor_xy"])
    model = _SAGE(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        X = model(x0, An)
        dh = (X[I] - X[J]).norm(dim=-1).clamp_min(1e-6)
        loss = F.huber_loss((dh - d) / s, torch.zeros_like(d)) + (((X[a] - ya) / s) ** 2).sum(-1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return _pin(g, model(x0, An).numpy())


def get_baselines(cfg):
    B = {
        "MDS-MAP": mds_map,
        "MDS-MAP + stress (L-BFGS)": mds_map_stress,
        "MDS-MAP + Huber stress": mds_map_huber,
        "DV-distance + dwMDS (no learning)": dv_dwmds,
        "Yan et al. GCN (3 anchor labels)": lambda g: yan_gcn(g, epochs=cfg.YAN_GCN_EPOCHS),
    }
    if cfg.RUN_SDP:
        B["SDP (Biswas-Ye)"] = sdp_biswas_ye
    if cfg.RUN_PERGRAPH_SAGE:
        B["Per-graph GraphSAGE (previous)"] = lambda g: pergraph_sage(g, epochs=cfg.PERGRAPH_EPOCHS)
    return B


def timed(fn, g):
    t = time.perf_counter(); X = fn(g); return X, time.perf_counter() - t
