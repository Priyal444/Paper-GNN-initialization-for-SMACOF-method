"""
common.py -- shared building blocks used by step1 / step2 / step3.

    1. geometry           : points, distance matrices
    2. graph generators   : random (spanning tree + extra edges), RGG, BA
    3. noise              : multiplicative Gaussian + positive NLOS bias
    4. anchors            : well-spread anchor selection
    5. rigidity           : is the answer even unique?  (graph + per node)
    6. features           : everything the GNN is allowed to see
    7. metrics            : coordinate RMSE, certified RMSE, relative EDM error

GOAL COMPLIANCE
    The GNN and every baseline see ONLY: the observed (noisy) distances on
    the observed edges + the coordinates of the anchor nodes.
    Ground-truth coordinates of non-anchor nodes are stored in the dataset
    ONLY for (a) the supervised training loss on synthetic TRAINING graphs and
    (b) scoring. They never enter the features of any graph.
"""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree


# =============================================================================
# 1. Geometry
# =============================================================================
def random_points(n, rng, coord_range=10.0):
    """n points uniform in [-R, R]^2, shape (n, 2)."""
    return rng.uniform(-coord_range, coord_range, size=(n, 2))


def dist_matrix(P):
    """Raw (NOT squared) Euclidean distance matrix of points P (n, 2)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# =============================================================================
# 2. Graph generators  -- all return an (m, 2) array of undirected edges i<j
# =============================================================================
def _target_edges(n, missing):
    total = n * (n - 1) // 2
    return max(n - 1, int(round(total * (1.0 - missing))))


def graph_random(n, missing, rng):
    """
    "Spanning-tree" generator.
      1. random spanning tree: node order is shuffled and each node attaches
         to a uniformly random earlier node   -> connected, n-1 edges
      2. add uniformly random extra pairs until the target edge count.
    Much faster than "delete an edge, check connectivity, undo", and the graph
    is connected by construction.  NOTE: connected != rigid (see rigidity()).
    """
    order = rng.permutation(n)
    edges = set()
    for k in range(1, n):
        a, b = order[k], order[rng.integers(0, k)]
        edges.add((min(a, b), max(a, b)))
    target = _target_edges(n, missing)
    iu = np.array(np.triu_indices(n, 1)).T
    rng.shuffle(iu)
    for a, b in iu:
        if len(edges) >= target:
            break
        edges.add((int(a), int(b)))
    return np.array(sorted(edges), dtype=np.int64)


def graph_rgg(P, missing):
    """
    Random geometric (radio-range) graph: keep the SHORTEST pairs until the
    target edge count, then add the Euclidean minimum spanning tree so the
    graph is connected (only short links are ever added).
    """
    n = len(P)
    D = dist_matrix(P)
    iu = np.triu_indices(n, 1)
    order = np.argsort(D[iu])
    target = _target_edges(n, missing)
    keep = order[:target]
    edges = set(zip(iu[0][keep].tolist(), iu[1][keep].tolist()))
    mst = minimum_spanning_tree(csr_matrix(D)).tocoo()
    for a, b in zip(mst.row, mst.col):
        edges.add((int(min(a, b)), int(max(a, b))))
    return np.array(sorted(edges), dtype=np.int64)


def ba_m_for(n, missing):
    """m whose BA edge count m*(n-m) is closest to the target for this missing %.
    BA has at most n^2/4 edges (m = n/2), so it cannot go below ~50 % missing;
    low missing levels are capped at m = n // 2."""
    target = _target_edges(n, missing)
    ms = np.arange(1, n // 2 + 1)
    return int(ms[np.argmin(np.abs(ms * (n - ms) - target))])


def ba_reachable(n, missing, tol=0.05):
    """True if a BA graph can match this missing % to within tol."""
    m = ba_m_for(n, missing)
    total = n * (n - 1) // 2
    return abs((1.0 - m * (n - m) / total) - missing) <= tol


def graph_ba(n, m, rng):
    """Barabasi-Albert graph: each new node attaches to m existing nodes.
    m = 1 is a tree (never rigid), m = 2 gives flip ambiguities, m >= 3 is
    close to a trilateration construction (usually uniquely localisable).
    Edge count is m * (n - m) (networkx starts from a star on m + 1 nodes)."""
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1 << 31)))
    e = np.array([(min(a, b), max(a, b)) for a, b in G.edges()], dtype=np.int64)
    return e


def is_connected(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges.tolist())
    return nx.is_connected(G)


# =============================================================================
# 3. Noise (on RAW distances)
# =============================================================================
def noisy_distances(d_true, rng, sigma=0.0, p_nlos=0.0, nlos_max=0.0):
    """d_obs = d (1 + sigma*eps) + b * U[0, nlos_max],  b ~ Bernoulli(p_nlos)."""
    d = d_true * (1.0 + sigma * rng.standard_normal(d_true.shape))
    if p_nlos > 0:
        bias = rng.uniform(0, nlos_max, d_true.shape) * (rng.random(d_true.shape) < p_nlos)
        d = d + bias
    return np.maximum(d, 1e-3)


# =============================================================================
# 4. Anchors
# =============================================================================
def pick_anchors(P, k, rng, tries=60):
    """k anchors, chosen among `tries` random k-subsets to maximise the
    smallest triangle area (well-spread, never collinear). In a deployment
    you PLACE anchors, so choosing them is part of the experimental design."""
    n = len(P)
    best, best_score = None, -1.0
    for _ in range(tries):
        idx = rng.choice(n, size=k, replace=False)
        A = P[idx]
        score = np.inf
        for a in range(k):
            for b in range(a + 1, k):
                for c in range(b + 1, k):
                    u, v = A[b] - A[a], A[c] - A[a]
                    score = min(score, abs(u[0] * v[1] - u[1] * v[0]) / 2)
        if score > best_score:
            best, best_score = idx, score
    return np.sort(best)


# =============================================================================
# 5. Rigidity: is the answer unique?
# =============================================================================
def _trilaterate_from(Adj, seed):
    """Grow a set from `seed`: add any node with >= 3 neighbours in the set."""
    done = np.zeros(len(Adj), bool)
    done[list(seed)] = True
    while True:
        new = (Adj @ done >= 3) & ~done
        if not new.any():
            return done
        done |= new


def trilateration_certified(n, edges, anchors, max_seeds=3000, rng=None):
    """
    Per-node SUFFICIENT certificate of unique localisability (generic positions).
      * Growing a set by trilateration (each new node has >= 3 neighbours
        already in the set) from a triangle gives a globally rigid subgraph.
      * Two globally rigid subgraphs sharing >= 3 nodes form a globally rigid
        union, so every grown set that contains all 3 anchors is pinned down
        uniquely by the anchors.
    certified = union of those sets (anchors themselves are seeds too).
    Nodes left uncertified MAY still be unique -- the certificate is one-sided.
    """
    Adj = np.zeros((n, n), np.int32)
    Adj[edges[:, 0], edges[:, 1]] = 1
    Adj[edges[:, 1], edges[:, 0]] = 1
    anchors = list(anchors)
    cert = np.zeros(n, bool)
    comp = _trilaterate_from(Adj, anchors)          # seed = the anchor triangle
    cert |= comp
    # seeds = triangles of the measurement graph
    tris = []
    for i, j in edges:
        for k in np.nonzero(Adj[i] & Adj[j])[0]:
            if k > j:
                tris.append((i, j, k))
    if rng is not None and len(tris) > max_seeds:
        tris = [tris[t] for t in rng.choice(len(tris), max_seeds, replace=False)]
    seen = np.zeros(n, bool)
    for t in tris:
        if seen[list(t)].all():
            continue
        comp = _trilaterate_from(Adj, t)
        seen |= comp
        if comp[anchors].all():
            cert |= comp
    return cert


def globally_rigid(n, edges, anchors, rng, d=2, tol=1e-8):
    """
    Randomised generic GLOBAL rigidity test (Gortler-Healy-Thurston 2010).
    Anchors are joined into a clique (their mutual distances are known).
    G is generically globally rigid in R^d  <=>  for a generic configuration
    q there is an equilibrium stress Omega with rank(Omega) = n - d - 1
    (and the framework is infinitesimally rigid, rank R = dn - d(d+1)/2).
    """
    e = set(map(tuple, edges.tolist()))
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a, b = sorted((int(anchors[i]), int(anchors[j])))
            e.add((a, b))
    E = np.array(sorted(e))
    q = rng.standard_normal((n, d))
    R = np.zeros((len(E), d * n))
    for k, (i, j) in enumerate(E):
        diff = q[i] - q[j]
        R[k, d * i:d * i + d] = diff
        R[k, d * j:d * j + d] = -diff
    s = np.linalg.svd(R, compute_uv=False)
    rankR = int((s > tol * s[0]).sum())
    if rankR < d * n - d * (d + 1) // 2:
        return False
    # random equilibrium stress = projection of a random vector onto null(R^T)
    z = rng.standard_normal(len(E))
    w = z - R @ np.linalg.lstsq(R, z, rcond=None)[0]
    Om = np.zeros((n, n))
    for k, (i, j) in enumerate(E):
        Om[i, j] -= w[k]
        Om[j, i] -= w[k]
        Om[i, i] += w[k]
        Om[j, j] += w[k]
    ev = np.abs(np.linalg.eigvalsh(Om))
    rankOm = int((ev > 1e-7 * max(ev.max(), 1e-12)).sum())
    return rankOm == n - d - 1


# =============================================================================
# 6. Features  -- built ONLY from observed distances + anchor coordinates
# =============================================================================
def geodesic_matrix(n, edges, d_obs):
    """Shortest-path lengths through the observed graph (n, n)."""
    W = csr_matrix((np.r_[d_obs, d_obs], (np.r_[edges[:, 0], edges[:, 1]],
                                         np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    return shortest_path(W, directed=False)


def hop_matrix(n, edges, rows):
    W = csr_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]],
                                              np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    return shortest_path(W, directed=False, unweighted=True, indices=rows)


def multilaterate(anchor_xy, g):
    """
    Linearised least-squares multilateration (DV-distance style).
    ||x - a_k||^2 = g_k^2, subtract the k = 1 equation:
        2 (a_k - a_1)^T x = ||a_k||^2 - ||a_1||^2 - g_k^2 + g_1^2
    anchor_xy: (K, 2), g: (n, K) estimated distances to anchors -> (n, 2)
    """
    a1 = anchor_xy[0]
    A = 2.0 * (anchor_xy[1:] - a1)                                  # (K-1, 2)
    b = ((anchor_xy[1:] ** 2).sum(1) - (a1 ** 2).sum())[None, :] \
        - g[:, 1:] ** 2 + g[:, :1] ** 2                             # (n, K-1)
    X, *_ = np.linalg.lstsq(A, b.T, rcond=None)
    return X.T


def build_features(n, edges, d_obs, anchors, anchor_xy):
    """
    Returns a dict of arrays in a NORMALISED frame:
        origin = anchor centroid, unit = mean pairwise anchor distance (s).
    Node features (19):
        is_anchor (1) | own anchor xy (2) | x0 multilateration init (2)
        geodesic dist to each anchor (K=3) | all anchor xy broadcast (6)
        hop count to each anchor / 10 (3) | log(1+degree) (1) | mean edge length (1)
    Edge features (3, per DIRECTED edge):
        d_obs | d_obs^2 | |d_obs - ||x0_i - x0_j|||   (last one flags NLOS-like edges)
    """
    K = len(anchors)
    c = anchor_xy.mean(0)
    s = np.mean([np.linalg.norm(anchor_xy[i] - anchor_xy[j])
                 for i in range(K) for j in range(i + 1, K)])
    A_n = (anchor_xy - c) / s
    d_n = d_obs / s

    G = geodesic_matrix(n, edges, d_n)[:, anchors]                  # (n, K)
    H = hop_matrix(n, edges, anchors).T                             # (n, K)
    x0 = multilaterate(A_n, G)
    x0[anchors] = A_n

    deg = np.bincount(edges.ravel(), minlength=n).astype(float)
    sum_len = np.bincount(edges[:, 0], weights=d_n, minlength=n) + \
              np.bincount(edges[:, 1], weights=d_n, minlength=n)
    is_a = np.zeros(n)
    is_a[anchors] = 1
    own = np.zeros((n, 2))
    own[anchors] = A_n
    X = np.concatenate([
        is_a[:, None], own, x0, G,
        np.repeat(A_n.reshape(1, -1), n, 0),
        H / 10.0, np.log1p(deg)[:, None], (sum_len / np.maximum(deg, 1))[:, None],
    ], axis=1)

    src = np.r_[edges[:, 0], edges[:, 1]]
    dst = np.r_[edges[:, 1], edges[:, 0]]
    dd = np.r_[d_n, d_n]
    incons = np.abs(dd - np.linalg.norm(x0[src] - x0[dst], axis=1))
    EF = np.stack([dd, dd ** 2, incons], 1)
    return dict(x=X.astype(np.float32), x0=x0.astype(np.float32),
                edge_index=np.stack([src, dst]).astype(np.int64),
                edge_attr=EF.astype(np.float32), d_edge=dd.astype(np.float32),
                center=c.astype(np.float32), scale=np.float32(s))


# =============================================================================
# One complete graph sample
# =============================================================================
def make_graph(rng, n, graph_type, missing, noise, n_anchors=3, coord_range=10.0,
               ba_m=None, check_global=True):
    """
    Build one sample. Everything a method may use lives under 'obs';
    ground truth lives under 'true_xy' (training loss + scoring only).
    """
    P = random_points(n, rng, coord_range)
    if graph_type == "random":
        edges = graph_random(n, missing, rng)
    elif graph_type == "rgg":
        edges = graph_rgg(P, missing)
    elif graph_type == "ba":
        if ba_m is None:                  # main comparison: m chosen from the missing %
            ba_m = ba_m_for(n, missing)
        edges = graph_ba(n, ba_m, rng)
    else:
        raise ValueError(graph_type)
    d_true = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    d_obs = noisy_distances(d_true, rng, **noise)
    anchors = pick_anchors(P, n_anchors, rng)
    feats = build_features(n, edges, d_obs, anchors, P[anchors])
    total = n * (n - 1) // 2
    gr = globally_rigid(n, edges, anchors, rng) if check_global else None
    cert = np.ones(n, bool) if gr else trilateration_certified(n, edges, anchors, rng=rng)
    return dict(
        n=n, graph_type=graph_type, missing_nominal=float(missing),
        missing_actual=1.0 - len(edges) / total,
        edges=edges, d_obs=d_obs.astype(np.float32),
        anchors=anchors, anchor_xy=P[anchors].astype(np.float32),
        true_xy=P.astype(np.float32),
        certified=cert,            # nodes whose position is provably unique
        globally_rigid=gr,         # whole graph uniquely localisable?
        ba_m=ba_m,                 # None unless graph_type == "ba"
        **feats,
    )


# =============================================================================
# 7. Metrics (ground truth used HERE only)
# =============================================================================
def unknown_mask(g):
    m = np.ones(g["n"], bool)
    m[g["anchors"]] = False
    return m


def coord_rmse(g, X, mask=None):
    """RMSE over non-anchor nodes (or a sub-mask of them)."""
    m = unknown_mask(g) if mask is None else (mask & unknown_mask(g))
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(((X[m] - g["true_xy"][m]) ** 2).sum(1).mean()))


def rel_edm_error(g, X):
    """||D_hat - D||_F / ||D||_F on raw distances (blind to reflections)."""
    D, Dh = dist_matrix(g["true_xy"]), dist_matrix(X)
    return float(np.linalg.norm(D - Dh) / np.linalg.norm(D))
