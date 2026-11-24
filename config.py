"""
config.py -- every experiment setting in ONE place.
Change things here, never inside the step files.
"""
import os

# ----------------------------------------------------------------------------
# Problem size
# ----------------------------------------------------------------------------
N_NODES        = 50          # nodes per test graph (the paper's main setting)
N_ANCHORS      = 3           # the theoretical minimum in 2-D (d + 1)
COORD_RANGE    = 10.0        # points are uniform in [-COORD_RANGE, COORD_RANGE]^2

# ----------------------------------------------------------------------------
# How edges go missing
#   "random" : spanning tree + random extra edges (fast, always connected)
#   "rgg"    : random geometric graph -- keep the SHORTEST pairs
#              (+ Euclidean MST so it is connected). This is how real
#              sensor networks lose links (radio range).
#   "ba"     : Barabasi-Albert, m chosen so m*(n-m) edges match the missing %.
#              BA has at most n^2/4 edges, so it only exists at >= ~50 %
#              missing; lower test levels are skipped for BA.
# ----------------------------------------------------------------------------
GRAPH_TYPES     = ["random", "rgg", "ba"]      # training mix
MISSING_LEVELS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ----------------------------------------------------------------------------
# Noise on the RAW distances (not squared):
#   d_obs = d_true * (1 + sigma * N(0,1))  +  Bernoulli(p_nlos) * U[0, nlos_max]
# The second term is a POSITIVE non-line-of-sight (NLOS) bias, as in
# Yan et al. (ICASSP 2021). nlos_max is in coordinate units (area is 20 x 20).
# ----------------------------------------------------------------------------
NOISE_MODELS = {
    "clean": dict(sigma=0.00, p_nlos=0.0, nlos_max=0.0),
    "gauss": dict(sigma=0.05, p_nlos=0.0, nlos_max=0.0),
    "nlos":  dict(sigma=0.02, p_nlos=0.2, nlos_max=4.0),
}

# ----------------------------------------------------------------------------
# Datasets (step 1)
# ----------------------------------------------------------------------------
N_TRAIN          = 10000     # training graphs
N_VAL            = 1000      # validation graphs
TRAIN_N_RANGE    = (50, 50)  # train strictly on n=50 -> tests zero-shot size generalization
TRAIN_MISSING    = (0.10, 0.90)
TEST_PER_LEVEL   = 40        # 20 per cell * 64 cells = 1,280 test graphs for n=50 (>= 1,000 graphs)
TEST_SCENARIOS   = [("random", "clean"), ("random", "gauss"), ("random", "nlos"),
                    ("rgg", "clean"),    ("rgg", "gauss"),    ("rgg", "nlos"),
                    ("ba", "clean"),     ("ba", "nlos")]
GENERALISE_SIZES = [100, 200, 500, 1000]  # size generalisation: n=100, 200, 500, 1000
GENERALISE_SCENARIO = ("rgg", "nlos")
GENERALISE_LEVELS = [0.5, 0.7, 0.9]
GEN_TEST_PER_LEVEL = 15      # 15 * 3 levels = 45 graphs per size (up to 45,000 node evaluations)
BA_M_VALUES      = [1, 2, 3, 4, 5]     # rigidity-threshold experiment
BA_PER_M         = 20

# Cross-graph-type transfer: train on a subset of graph types, test on all.
# Every training set has the SAME total size (N_TRAIN, split equally over its
# types), so differences come from variety, not from amount of data.
# Validation (checkpoint selection) uses only the training types: no peeking
# at the held-out type.
RUN_TRANSFER        = True
TRANSFER_TRAIN_SETS = ["random", "rgg", "ba",                   # train on 1, test on the other 2
                       "random+rgg", "random+ba", "rgg+ba",     # train on 2, test on the 3rd
                       "all"]                                   # train on all 3
TRANSFER_MIN_MISSING = 0.5   # matrix averages over missing >= 50 % (BA exists only there)
SEED             = 1234

# ----------------------------------------------------------------------------
# Model (step 2)
# ----------------------------------------------------------------------------
HIDDEN          = 96
ENC_LAYERS      = 5          # message-passing layers in the encoder
REFINE_STEPS    = 10         # unrolled learned-SMACOF steps during TRAINING
TEST_REFINE_STEPS = 50       # steps at TEST time (same shared step, run longer:
                             # exact fits are fixed points, so more steps only converge further)
VARIANT         = "learned"  # "learned" | "uniform" | "huber" | "huber_alpha" | "encoder_only"
LOSS_MODE       = "supervised"   # "supervised" (synthetic GT) | "selfsup" (no GT)
HUBER_DELTA     = 0.05       # in normalised units (anchor-triangle scale = 1)
AUX_WEIGHT      = 0.3        # weight of the encoder-output loss (deep supervision)
EPOCHS          = 100        # Pinned pre-registered training epochs
BATCH_GRAPHS    = 32
LR              = 5e-4       # Pinned pre-registered learning rate (stabilized)
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 0.5        # Pinned pre-registered gradient clipping
STEP_CLAMP      = 1.0        # Pinned pre-registered step update clamp
TEST_FILE       = "test_fresh.pkl"  # Held-out fresh test set
TEST_SEED       = 8888       # Independent fresh test generation seed

# ----------------------------------------------------------------------------
# Evaluation (step 3)
# ----------------------------------------------------------------------------
RUN_SDP            = True    # Biswas-Ye SDP via cvxpy (slowest baseline)
RUN_PERGRAPH_SAGE  = True    # your previous approach (per-graph training)
RUN_MPNN_LOC       = True    # MPNN-Loc (Xu & Lin, IFAC 2025)
RUN_CPU_TIMING     = True    # Set to True for single-threaded CPU runtime benchmark
RUN_EXTENDED_EXPERIMENTS = True
ANCHOR_COUNTS      = [3, 4, 5, 6, 7, 8]  # K = 3..8 anchor sensitivity sweep
EXTENDED_TEST_PER_LEVEL = 50             # graphs per level for extended sweeps
PERGRAPH_EPOCHS    = 1000
YAN_GCN_EPOCHS     = 200
MPNN_LOC_EPOCHS    = 200
RUN_GENERALISATION = True
RUN_BA_EXPERIMENT  = True


def train_types(key):
    """'rgg+ba' -> ['rgg', 'ba'];  'all' -> GRAPH_TYPES."""
    types = list(GRAPH_TYPES) if key == "all" else key.split("+")
    bad = [t for t in types if t not in GRAPH_TYPES]
    if bad:
        raise ValueError(f"unknown graph type(s) {bad}; choose from {GRAPH_TYPES} or 'all'")
    return types

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
RUN_TAG     = os.environ.get("RUN_TAG", "")
CKPT_DIR    = os.path.join(ROOT, "checkpoints", RUN_TAG)
RESULTS_DIR = os.path.join(ROOT, "results", RUN_TAG)
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# Any of these can be overridden from the environment, e.g.  EPOCHS=10 python step2_train.py
for _k in ("N_TRAIN", "N_VAL", "TEST_PER_LEVEL", "GEN_TEST_PER_LEVEL", "EXTENDED_TEST_PER_LEVEL",
           "BA_PER_M", "EPOCHS", "PERGRAPH_EPOCHS", "REFINE_STEPS",
           "TEST_REFINE_STEPS", "MPNN_LOC_EPOCHS"):
    if _k in os.environ:
        globals()[_k] = int(os.environ[_k])
for _k in ("RUN_SDP", "RUN_PERGRAPH_SAGE", "RUN_TRANSFER", "RUN_MPNN_LOC"):       # e.g.  RUN_SDP=0 python step3_evaluate.py
    if _k in os.environ:
        globals()[_k] = os.environ[_k] == "1"
SDP_MAX_N = 100   # SDP gets slow for big graphs (n=200 ~ 1 min/graph): skipped above this

# Set QUICK=1 in the environment for a tiny smoke-test run.
if os.environ.get("QUICK") == "1":
    N_TRAIN, N_VAL, TEST_PER_LEVEL, EPOCHS = 200, 40, 2, 3
    MISSING_LEVELS = [0.3, 0.7, 0.9]
    GENERALISE_LEVELS = [0.7]
    BA_PER_M = 2
    EXTENDED_TEST_PER_LEVEL = 2
    PERGRAPH_EPOCHS, YAN_GCN_EPOCHS, MPNN_LOC_EPOCHS = 50, 30, 30
