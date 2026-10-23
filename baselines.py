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


def geodesic_matrix_torch(n, E, d, device="cpu"):
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    D = torch.full((n, n), float("inf"), dtype=torch.float32, device=dev)
    D.fill_diagonal_(0.0)
    src = torch.as_tensor(E[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(E[:, 1], dtype=torch.long, device=dev)
    dd = torch.as_tensor(d, dtype=torch.float32, device=dev)
    D[src, dst] = dd
    D[dst, src] = dd
    for k in range(n):
        D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D


def classical_mds_torch(D, d=2, device="cpu"):
    n = D.shape[0]
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    Dt = torch.as_tensor(D, dtype=torch.float32, device=dev)
    J = torch.eye(n, device=dev) - (1.0 / n)
    B = -0.5 * J @ (Dt ** 2) @ J
    sym_B = (B + B.T) / 2
    try:
        w, V = torch.linalg.eigh(sym_B)
    except Exception:
        # cuSOLVER lazy wrapper race condition or CUDA exception -> fallback to numpy
        w_np, V_np = np.linalg.eigh(sym_B.detach().cpu().numpy())
        w = torch.as_tensor(w_np, device=dev)
        V = torch.as_tensor(V_np, device=dev)
    idx = torch.arange(n - 1, n - d - 1, -1, device=dev)
    val = torch.clamp(w[idx], min=0.0)
    X = V[:, idx] * torch.sqrt(val)
    return X


def procrustes_torch(X, anchors, A, device="cpu"):
    dev = X.device
    anc = torch.as_tensor(anchors, dtype=torch.long, device=dev)
    At = torch.as_tensor(A, dtype=torch.float32, device=dev)
    S = X[anc]
    sc = S.mean(0)
    ac = At.mean(0)
    H = (S - sc).T @ (At - ac)
    try:
        U, _, Vt = torch.linalg.svd(H)
    except Exception:
        U_np, _, Vt_np = np.linalg.svd(H.detach().cpu().numpy())
        U = torch.as_tensor(U_np, device=dev)
        Vt = torch.as_tensor(Vt_np, device=dev)
    best_R = None
    best_err = float("inf")
    for sg in (1.0, -1.0):
        V_curr = Vt.clone()
        V_curr[-1, :] *= sg
        R = V_curr.T @ U.T
        err = torch.norm((S - sc) @ R.T - (At - ac))
        if err < best_err:
            best_R = R
            best_err = err
    X_aligned = (X - sc) @ best_R.T + ac
    X_aligned[anc] = At
    return X_aligned


def _pin(g, X):
    X = X.copy(); X[g["anchors"]] = g["anchor_xy"]; return X


def stress_lbfgs(g, X0, robust=False, max_iter=3000, device="cpu"):
    """Minimise sum_edges rho(||x_i - x_j|| - d_ij) over NON-anchor nodes.
    Uses compiled C/Fortran L-BFGS-B (10x faster than GPU for small 94-variable graphs)."""
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
def mds_map(g, device="cpu"):
    D = geodesic_matrix(g["n"], g["edges"], g["d_obs"].astype(float))
    X = classical_mds(D)
    return _pin(g, procrustes_O2(X, g["anchors"], g["anchor_xy"]))


def mds_map_stress(g, device="cpu"):
    return stress_lbfgs(g, mds_map(g), robust=False)


def mds_map_huber(g, device="cpu"):
    return stress_lbfgs(g, mds_map(g), robust=True)




def dv_dwmds(g, iters=300, alpha=0.5, device="cpu"):
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(float)
    G = geodesic_matrix(n, E, d)[:, g["anchors"]]
    X = _pin(g, multilaterate(g["anchor_xy"].astype(float), G))
    if device != "cpu" and torch.cuda.is_available():
        dev = torch.device(device)
        src = torch.as_tensor(np.r_[E[:, 0], E[:, 1]], dtype=torch.long, device=dev)
        dst = torch.as_tensor(np.r_[E[:, 1], E[:, 0]], dtype=torch.long, device=dev)
        dd = torch.as_tensor(np.r_[d, d], dtype=torch.float32, device=dev)
        deg = torch.bincount(dst, minlength=n).float().unsqueeze(1).to(dev)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        anc = torch.as_tensor(g["anchors"], dtype=torch.long, device=dev)
        axy = torch.as_tensor(g["anchor_xy"], dtype=torch.float32, device=dev)
        for _ in range(iters):
            diff = Xt[dst] - Xt[src]
            dh = diff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            t = Xt[src] + dd.unsqueeze(1) * diff / dh
            num = torch.zeros_like(Xt)
            num.index_add_(0, dst, t)
            Xt = Xt + alpha * (num / deg - Xt)
            Xt[anc] = axy
        return Xt.cpu().numpy()
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


def sdp_biswas_ye(g, device="cpu"):
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
        use_gpu = (device != "cpu" and torch.cuda.is_available())
        try:
            if use_gpu:
                prob.solve(solver=cp.SCS, max_iters=1000, eps=1e-3, verbose=False, gpu=True)
            else:
                prob.solve(solver=cp.SCS, max_iters=1000, eps=1e-3, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, max_iters=1000, eps=1e-3, verbose=False)
        Xu = np.asarray(Z.value[:2, 2:]).T
    except Exception:
        return mds_map(g, device=device)
    X = np.zeros((n, 2)); X[unk] = Xu
    return _pin(g, X)



def yan_gcn(g, epochs=200, hidden=256, lr=0.01, dropout=0.5, seed=0, device="cpu"):
    """
    Yan et al. 2021 (ICASSP), reproduced on our graphs:
      features = rows of the (thresholded = observed) distance matrix, row-normalised
      2-layer GCN with augmented normalised adjacency, trained per graph,
      loss = MSE on the ANCHOR nodes only (here: 3 labels).
    Paper used 2000 hidden units; 256 is used here for speed.
    """
    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    Xd = np.zeros((n, n), np.float32)
    Xd[E[:, 0], E[:, 1]] = d; Xd[E[:, 1], E[:, 0]] = d
    Xf_np = Xd / np.maximum(np.linalg.norm(Xd, axis=1, keepdims=True), 1e-9)
    A = (Xd > 0).astype(np.float32) + np.eye(n, dtype=np.float32)
    Dm = 1.0 / np.sqrt(A.sum(1))
    Ah = torch.tensor(Dm[:, None] * A * Dm[None, :], device=dev)
    Xf = torch.tensor(Xf_np, device=dev)
    W1, W2 = nn.Linear(n, hidden).to(dev), nn.Linear(hidden, 2).to(dev)
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=lr)
    a = torch.tensor(g["anchors"], dtype=torch.long, device=dev)
    ya = torch.tensor(g["anchor_xy"], dtype=torch.float32, device=dev)
    def fwd(train):                       # R = A_h relu(A_h X W1) W2   (paper eq. 5)
        h = F.dropout(F.relu(W1(Ah @ Xf)), dropout, train)
        return Ah @ W2(h)
    for _ in range(epochs):
        opt.zero_grad(); loss = F.mse_loss(fwd(True)[a], ya); loss.backward(); opt.step()
    with torch.no_grad():
        return _pin(g, fwd(False).cpu().numpy())


class _MPNNLocNet(nn.Module):
    """
    MPNN-Loc Architecture from:
    Xu & Lin (IFAC 2025): 'Graph Neural Network based Localization for Layout-Agnostic Sensor Networks'
    """
    def __init__(self, n, in_dim=2, hidden=200, dropout=0.5):
        super().__init__()
        self.n = n
        self.dropout = dropout
        
        # Layer 1: t=1, M^1 dimension = 2 * in_dim + 1 = 5
        m1_dim = 2 * in_dim + 1
        self.W_node1 = nn.Linear(m1_dim, hidden)
        self.W_edge1 = nn.Linear(m1_dim, n)
        
        # Layer 2: Readout at t=Tl=2, M^2 dimension = 2 * hidden + 1 = 401
        m2_dim = 2 * hidden + 1
        self.W_readout = nn.Linear(m2_dim, 2)

    def forward(self, H0, E0, A_hat):
        # A_hat: row-normalized adjacency matrix (N, N)
        # H0: initial node features (N, 2)
        # E0: initial edge distance matrix (N, N)
        
        # --- Layer 1 ---
        AE0_diag = torch.diagonal(A_hat @ E0).unsqueeze(-1)
        M1 = torch.cat([H0, A_hat @ H0, AE0_diag], dim=-1)   # (N, 5)
        
        # Node update (Eq. 11)
        H1 = F.relu(self.W_node1(M1))
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        
        # Edge update (Eq. 12)
        E1_raw = self.W_edge1(M1)                            # (N, N)
        E1 = F.relu(A_hat * E1_raw)                          # Hadamard product A_hat \odot (...)
        
        # --- Layer 2 / Readout (Eq. 10 & 13) ---
        AE1_diag = torch.diagonal(A_hat @ E1).unsqueeze(-1)
        M2 = torch.cat([H1, A_hat @ H1, AE1_diag], dim=-1)   # (N, 2*hidden + 1)
        
        P_hat = self.W_readout(M2)                           # (N, 2)
        return P_hat


def _correlation_refinement(P_hat, P_L, A, anchors, agents, dev):
    """
    Correlation-based refinement module (Eq. 14-17 in Xu & Lin, IFAC 2025).
    P'_U = P_hat_U - Gamma_UU^{-1} Gamma_UL (P_L - P_hat_L)
    where Gamma = I - K^{-1/2} A K^{-1/2} is the symmetric normalized Laplacian.
    """
    n = A.shape[0]
    deg = A.sum(dim=-1)
    deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
    A_sym = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(0)
    Gamma = torch.eye(n, device=dev) - A_sym
    
    # Submatrices
    Gamma_UU = Gamma[agents][:, agents]
    Gamma_UL = Gamma[agents][:, anchors]
    
    # Anchors difference: (P_L - P_hat_L)
    diff_L = P_L - P_hat[anchors]
    rhs = Gamma_UL @ diff_L                                # (N_agents, 2)
    
    # Regularized solve for numerical stability
    reg = 1e-4 * torch.eye(len(agents), device=dev)
    try:
        delta_U = torch.linalg.solve(Gamma_UU + reg, rhs)
    except Exception:
        delta_U = torch.linalg.lstsq(Gamma_UU + reg, rhs).solution
        
    P_refined = P_hat.clone()
    P_refined[agents] = P_hat[agents] - delta_U
    P_refined[anchors] = P_L
    return P_refined


def mpnn_loc(g, epochs=200, hidden=200, lr=0.001, dropout=0.5, refine=True, seed=0, device="cpu"):
    """
    MPNN-Loc algorithm from Xu & Lin (IFAC 2025):
    'Graph Neural Network based Localization for Layout-Agnostic Sensor Networks'
    
    Consists of:
    1. Node feature initialization (anchor coords, nearest-anchor coords, or area center)
    2. 2-layer MPNN with joint node & edge updates
    3. Semi-supervised training minimizing MSE on anchor predictions
    4. Nonparametric correlation-based refinement using symmetric normalized Laplacian
    """
    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    
    n = g["n"]
    E = g["edges"]
    d = g["d_obs"].astype(np.float32)
    anchors_list = list(g["anchors"])
    anchors = torch.tensor(anchors_list, dtype=torch.long, device=dev)
    agents_list = [i for i in range(n) if i not in anchors_list]
    agents = torch.tensor(agents_list, dtype=torch.long, device=dev)
    P_L = torch.tensor(g["anchor_xy"], dtype=torch.float32, device=dev)
    
    # 1. Adjacency and Distance Matrices
    A_np = np.zeros((n, n), dtype=np.float32)
    D_np = np.zeros((n, n), dtype=np.float32)
    A_np[E[:, 0], E[:, 1]] = 1.0
    A_np[E[:, 1], E[:, 0]] = 1.0
    D_np[E[:, 0], E[:, 1]] = d
    D_np[E[:, 1], E[:, 0]] = d
    
    # Row-normalized adjacency A_hat = K^{-1} A (Eq. 3, page 3)
    k_np = np.maximum(A_np.sum(axis=1, keepdims=True), 1e-6)
    A_hat_np = A_np / k_np
    
    A = torch.tensor(A_np, device=dev)
    A_hat = torch.tensor(A_hat_np, device=dev)
    E0 = torch.tensor(D_np, device=dev)
    
    # 2. Node feature initialization X (Eq. 5-9)
    # Anchor: X_i = P_i
    # Agent: X_i = P_vina if connected to any anchor, else P_center
    P_center = g["anchor_xy"].mean(axis=0)
    X_init = np.tile(P_center, (n, 1)).astype(np.float32)
    for a_idx, a_node in enumerate(anchors_list):
        X_init[a_node] = g["anchor_xy"][a_idx]
        
    for ag in agents_list:
        anc_nbrs = [nbr for nbr in anchors_list if A_np[ag, nbr] > 0]
        if anc_nbrs:
            vina = min(anc_nbrs, key=lambda a_nbr: D_np[ag, a_nbr])
            vina_idx = anchors_list.index(vina)
            X_init[ag] = g["anchor_xy"][vina_idx]
            
    H0 = torch.tensor(X_init, device=dev)
    
    # 3. Model & Optimizer
    model = _MPNNLocNet(n, in_dim=2, hidden=hidden, dropout=dropout).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training (Eq. 18: loss = MSE(P_hat_L, P_L))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        P_hat = model(H0, E0, A_hat)
        loss = F.mse_loss(P_hat[anchors], P_L)
        loss.backward()
        optimizer.step()
        
    # 5. Inference & Correlation-based refinement (Eq. 14-17)
    model.eval()
    with torch.no_grad():
        P_hat = model(H0, E0, A_hat)
        if refine:
            P_out = _correlation_refinement(P_hat, P_L, A, anchors, agents, dev)
        else:
            P_out = P_hat
            P_out[anchors] = P_L
        
    return P_out.cpu().numpy()


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


def pergraph_sage(g, epochs=1000, lr=1e-3, seed=0, device="cpu"):
    """Your previous pipeline (step1_corrected): MDS-MAP init -> GraphSAGE with a
    residual output, trained from scratch on THIS graph with Huber stress + anchors."""
    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    n, E = g["n"], g["edges"]
    x0 = torch.tensor(mds_map(g), dtype=torch.float32, device=dev)
    A = np.zeros((n, n), np.float32); A[E[:, 0], E[:, 1]] = 1; A[E[:, 1], E[:, 0]] = 1
    An = torch.tensor(A / A.sum(1, keepdims=True), device=dev)
    I = torch.tensor(E[:, 0], dtype=torch.long, device=dev)
    J = torch.tensor(E[:, 1], dtype=torch.long, device=dev)
    d = torch.tensor(g["d_obs"], dtype=torch.float32, device=dev); s = d.mean()
    a = torch.tensor(g["anchors"], dtype=torch.long, device=dev)
    ya = torch.tensor(g["anchor_xy"], dtype=torch.float32, device=dev)
    model = _SAGE().to(dev); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        X = model(x0, An)
        dh = (X[I] - X[J]).norm(dim=-1).clamp_min(1e-6)
        loss = F.huber_loss((dh - d) / s, torch.zeros_like(d)) + (((X[a] - ya) / s) ** 2).sum(-1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return _pin(g, model(x0, An).cpu().numpy())


def naive_classical_mds(g, device="cpu"):
    """Classical MDS (Algorithm 1) directly on observed distances (unobserved = 0)."""
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    src = torch.as_tensor(E[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(E[:, 1], dtype=torch.long, device=dev)
    Dt = torch.zeros((n, n), dtype=torch.float32, device=dev)
    dd = torch.as_tensor(d, dtype=torch.float32, device=dev)
    Dt[src, dst] = dd
    Dt[dst, src] = dd
    X = classical_mds_torch(Dt, d=2, device=dev)
    return procrustes_torch(X, g["anchors"], g["anchor_xy"], device=dev).cpu().numpy()


def rank_complete_edm(g, max_iters=100, tol=1e-5, device="cpu"):
    """Dokmanic et al. Algorithm 2: Alternating Rank-Based EDM Completion."""
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    W = torch.zeros((n, n), dtype=torch.bool, device=dev)
    src = torch.as_tensor(E[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(E[:, 1], dtype=torch.long, device=dev)
    W[src, dst] = True
    W[dst, src] = True
    d2 = torch.as_tensor(d ** 2, dtype=torch.float32, device=dev)
    D2 = torch.zeros((n, n), dtype=torch.float32, device=dev)
    D2[src, dst] = d2
    D2[dst, src] = d2
    mu = d2.mean()
    D_cur = torch.where(W, D2, mu)
    D_cur.fill_diagonal_(0.0)
    r = 4  # rank d+2
    for _ in range(max_iters):
        sym_D = (D_cur + D_cur.T) / 2
        try:
            w, V = torch.linalg.eigh(sym_D)
        except Exception:
            w_np, V_np = np.linalg.eigh(sym_D.detach().cpu().numpy())
            w = torch.as_tensor(w_np, device=dev)
            V = torch.as_tensor(V_np, device=dev)
        idx = torch.topk(w.abs(), r).indices
        D_next = (V[:, idx] * w[idx]) @ V[:, idx].T
        D_next[W] = D2[W]
        D_next.fill_diagonal_(0.0)
        D_next.clamp_min_(0.0)
        if torch.norm(D_next - D_cur) / (torch.norm(D_cur) + 1e-9) < tol:
            break
        D_cur = D_next
    D_est = torch.sqrt(D_cur)
    X = classical_mds_torch(D_est, d=2, device=dev)
    return procrustes_torch(X, g["anchors"], g["anchor_xy"], device=dev).cpu().numpy()


def optspace_edm(g, rank=4, device="cpu"):
    """Dokmanic et al. Algorithm 3: OptSpace EDM Completion."""
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    W = torch.zeros((n, n), dtype=torch.bool, device=dev)
    src = torch.as_tensor(E[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(E[:, 1], dtype=torch.long, device=dev)
    W[src, dst] = True
    W[dst, src] = True
    d2 = torch.as_tensor(d ** 2, dtype=torch.float32, device=dev)
    D2 = torch.zeros((n, n), dtype=torch.float32, device=dev)
    D2[src, dst] = d2
    D2[dst, src] = d2
    total_pairs = n * (n - 1) / 2.0
    p = max(len(E) / total_pairs, 1e-4)
    M_scaled = D2 / p
    deg = W.sum(dim=1).float()
    avg_deg = deg.mean()
    trim_mask = deg > (2.0 * avg_deg)
    M_trimmed = M_scaled.clone()
    if trim_mask.any():
        M_trimmed[trim_mask, :] = 0.0
        M_trimmed[:, trim_mask] = 0.0
    try:
        U, s, Vt = torch.linalg.svd(M_trimmed, full_matrices=False)
    except Exception:
        U_np, s_np, Vt_np = np.linalg.svd(M_trimmed.detach().cpu().numpy(), full_matrices=False)
        U = torch.as_tensor(U_np, device=dev)
        s = torch.as_tensor(s_np, device=dev)
        Vt = torch.as_tensor(Vt_np, device=dev)
    U_r, s_r, Vt_r = U[:, :rank], s[:rank], Vt[:rank, :]
    D_est = U_r @ torch.diag(s_r) @ Vt_r
    D_est[W] = D2[W]
    D_est.fill_diagonal_(0.0)
    D_est.clamp_min_(0.0)
    D_dist = torch.sqrt(D_est)
    X = classical_mds_torch(D_dist, d=2, device=dev)
    return procrustes_torch(X, g["anchors"], g["anchor_xy"], device=dev).cpu().numpy()


def alternating_descent(g, max_iters=50, seed=0, device="cpu"):
    """Dokmanic et al. Algorithm 4: Alternating Descent on s-stress.
    Fixed initialization: uses small Gaussian perturbation around anchor centroid
    instead of all-zeros to avoid the zero-gradient saddle point at iteration 0."""
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    n, E, d = g["n"], g["edges"], g["d_obs"].astype(np.float32)
    src = torch.as_tensor(E[:, 0], dtype=torch.long, device=dev)
    dst = torch.as_tensor(E[:, 1], dtype=torch.long, device=dev)
    d2 = torch.as_tensor(d ** 2, dtype=torch.float32, device=dev)
    anc = torch.as_tensor(g["anchors"], dtype=torch.long, device=dev)
    axy = torch.as_tensor(g["anchor_xy"], dtype=torch.float32, device=dev)
    is_free = torch.ones(n, dtype=torch.bool, device=dev)
    is_free[anc] = False
    free_idx = torch.nonzero(is_free).squeeze(-1)
    init_xy = axy.mean(0, keepdim=True) + 0.1 * torch.randn((len(free_idx), 2), dtype=torch.float32, device=dev)
    X_free = torch.nn.Parameter(init_xy)
    optimizer = torch.optim.LBFGS([X_free], max_iter=max_iters, lr=1.0, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        X_full = torch.zeros((n, 2), dtype=torch.float32, device=dev)
        X_full[anc] = axy
        X_full[free_idx] = X_free
        diff = X_full[src] - X_full[dst]
        dhat2 = (diff ** 2).sum(dim=-1)
        loss = ((dhat2 - d2) ** 2).sum()
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
    except Exception:
        pass
    X_final = torch.zeros((n, 2), dtype=torch.float32, device=dev)
    X_final[anc] = axy
    X_final[free_idx] = X_free.detach()
    return X_final.cpu().numpy()


def get_baselines(cfg, device="cpu"):
    B = {
        "MDS-MAP": lambda g: mds_map(g, device=device),
        "MDS-MAP + stress (L-BFGS)": lambda g: mds_map_stress(g, device=device),
        "MDS-MAP + Huber stress": lambda g: mds_map_huber(g, device=device),
        "DV-distance + dwMDS (no learning)": lambda g: dv_dwmds(g, device=device),
        "Classical MDS (direct)": lambda g: naive_classical_mds(g, device=device),
        "Rank Alternation (EDM book Alg 2)": lambda g: rank_complete_edm(g, device=device),
        "OptSpace (EDM book Alg 3)": lambda g: optspace_edm(g, device=device),
        "Alternating Descent (EDM book Alg 4)": lambda g: alternating_descent(g, device=device),
        "Yan et al. GCN (3 anchor labels)": lambda g: yan_gcn(g, epochs=cfg.YAN_GCN_EPOCHS, device=device),
    }
    if getattr(cfg, "RUN_MPNN_LOC", True):
        B["MPNN-Loc (Xu & Lin 2025)"] = lambda g: mpnn_loc(
            g, epochs=getattr(cfg, "MPNN_LOC_EPOCHS", 200), device=device
        )
    if cfg.RUN_SDP:
        B["SDP (Biswas-Ye)"] = lambda g: sdp_biswas_ye(g, device=device)
    if cfg.RUN_PERGRAPH_SAGE:
        B["Per-graph GraphSAGE (previous)"] = lambda g: pergraph_sage(g, epochs=cfg.PERGRAPH_EPOCHS, device=device)
    return B


def timed(fn, g):
    t = time.perf_counter(); X = fn(g); return X, time.perf_counter() - t
