import os, json

print("Starting notebook synchronization for 10k dataset & n=50..1000 generalisation...")

# 1. Read updated source files
files = {
    4: ("config.py", open("gnn_loc/config.py", encoding="utf-8").read()),
    5: ("common.py", open("gnn_loc/common.py", encoding="utf-8").read()),
    6: ("model.py", open("gnn_loc/model.py", encoding="utf-8").read()),
    7: ("baselines.py", open("gnn_loc/baselines.py", encoding="utf-8").read()),
    8: ("step1_data.py", open("gnn_loc/step1_data.py", encoding="utf-8").read()),
    9: ("step2_train.py", open("gnn_loc/step2_train.py", encoding="utf-8").read()),
    10: ("step3_evaluate.py", open("gnn_loc/step3_evaluate.py", encoding="utf-8").read()),
}

new_cell2_code = '''# ---------------- settings ----------------
MODE          = "scale_10k" # "scale_10k" (10k train, 1k val, 1.28k test) | "full" | "medium" | "quick"
USE_DRIVE     = True        # True: persist everything in Google Drive (survives disconnects, Colab only)
RUN_ABLATIONS = True        # uniform / huber / huber_alpha / encoder_only models
RUN_TRANSFER  = False       # Set to True for cross-graph-type transfer suite
RUN_SDP       = True        # Biswas-Ye SDP baseline (automatically skipped for n > 100)
TEST_GRAPHS   = 20          # 20 per cell * 64 cells = 1,280 test graphs (meets >= 1000 requirement)
SEEDS         = [0, 1, 2]   # multi-seed evaluation for IEEE SPL statistical rigor (mean +/- std)
# ------------------------------------------
import os
os.environ["PYTHONUNBUFFERED"] = "1"
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    WORKDIR = "/kaggle/working/gnn_loc"
else:
    WORKDIR = "/content/drive/MyDrive/gnn_loc" if USE_DRIVE else "/content/gnn_loc"
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
os.makedirs(WORKDIR, exist_ok=True)
%cd {WORKDIR}

for k in ["QUICK", "N_TRAIN", "N_VAL", "EPOCHS", "TEST_PER_LEVEL", "BA_PER_M", "PERGRAPH_EPOCHS"]:
    os.environ.pop(k, None)

if MODE == "quick":
    os.environ["QUICK"] = "1"
elif MODE == "medium":
    os.environ.update(N_TRAIN="1000", N_VAL="200", EPOCHS="20", TEST_PER_LEVEL="10", PERGRAPH_EPOCHS="300")
elif MODE in ("full", "scale_10k"):
    # 10,000 train graphs, 1,000 val graphs, 1,280 test graphs (64 cells * 20)
    # Generalisation across sizes n = 100, 200, 500, 1000
    os.environ.update(N_TRAIN="10000", N_VAL="1000", EPOCHS="50", TEST_PER_LEVEL="20", PERGRAPH_EPOCHS="300")
else:
    raise ValueError(MODE)

if TEST_GRAPHS and MODE != "quick":
    os.environ.update(TEST_PER_LEVEL=str(TEST_GRAPHS), BA_PER_M=str(TEST_GRAPHS))
os.environ["RUN_SDP"] = "1" if RUN_SDP else "0"
os.environ["RUN_TRANSFER"] = "1" if RUN_TRANSFER else "0"

import subprocess, sys
def sh(cmd, **env):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **env})
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"failed (exit code {p.returncode}): {cmd}")

print("MODE:", MODE, "| Train:", os.environ.get("N_TRAIN"), "| Val:", os.environ.get("N_VAL"),
      "| Test/level:", os.environ.get("TEST_PER_LEVEL"), "| working dir:", WORKDIR)'''

new_cell12_code = '''# Skip if the datasets already exist: regenerating would also wipe the baseline caches.
_need = ["train.pkl", "val.pkl", "test.pkl", "test_ba.pkl"]
if RUN_TRANSFER:
    _need += [f"train_pool_{t}.pkl" for t in ("random", "rgg", "ba")]
for _n in (100, 200, 500, 1000):
    _need.append(f"test_n{_n}.pkl")

need_regen = False
for f in _need:
    if not os.path.exists(os.path.join("data", f)):
        need_regen = True
        break

if not need_regen and os.path.exists("data/train.pkl"):
    import pickle
    try:
        with open("data/train.pkl", "rb") as _f:
            _tr = pickle.load(_f)
        target_train = int(os.environ.get("N_TRAIN", 10000))
        if len(_tr) < target_train:
            print(f"[*] Existing dataset has {len(_tr)} graphs, but target is {target_train}. Regenerating...")
            need_regen = True
    except Exception:
        need_regen = True

if need_regen:
    sh("python step1_data.py")
else:
    print("Datasets already exist and match scale: skipping step 1 (delete data/ to force rebuild)")'''

targets = [
    'gnn_loc/colab_all_in_one.ipynb',
    'gnn_loc/colab_all_in_one_extended.ipynb',
    r'C:\Users\priya\Downloads\colab_all_in_one_fixed.ipynb',
    r'C:\Users\priya\Downloads\notebookfb8f0663b9.ipynb'
]

for p in targets:
    if not os.path.exists(p):
        print(f"Skipping missing: {p}")
        continue
    try:
        with open(p, encoding='utf-8') as f:
            nb = json.load(f)
        
        # Update code files in cells 4-10
        for cell_idx, (fname, content) in files.items():
            if cell_idx < len(nb['cells']):
                first_line = f"%%writefile {fname}\n"
                code_lines = [first_line] + [line + "\n" for line in content.splitlines()]
                nb['cells'][cell_idx]['source'] = code_lines
                print(f"  [{os.path.basename(p)}] Updated Cell {cell_idx} ({fname})")
        
        # Update Cell 2 (settings) if present
        if len(nb['cells']) > 2 and 'MODE' in ''.join(nb['cells'][2]['source']):
            nb['cells'][2]['source'] = [line + '\n' for line in new_cell2_code.splitlines()]
            print(f"  [{os.path.basename(p)}] Updated Cell 2 (Scale settings)")
        
        # Update Cell 12 (data generation check) if present
        for i, c in enumerate(nb['cells']):
            src = ''.join(c['source'])
            if '_need = ["train.pkl"' in src and 'step1_data.py' in src:
                c['source'] = [line + '\n' for line in new_cell12_code.splitlines()]
                print(f"  [{os.path.basename(p)}] Updated Cell {i} (10k dataset check)")
                break

        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully synced: {p}\n")
    except Exception as e:
        print(f"Error syncing {p}: {e}")

print("All notebooks synced successfully!")

