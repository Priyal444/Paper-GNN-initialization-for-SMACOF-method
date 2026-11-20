import json

new_cell20_code = '''import glob, pandas as pd, numpy as np
from scipy.stats import wilcoxon
OURS = "Ours: AnchorGNN + learned SMACOF"
PAIRS = [  # (A, B, NLOS graphs only?)
    (OURS, "SDP + Huber polish", True),
    (OURS, "SDP + Asym polish (NLOS)", True),
    (OURS, "SDP (Biswas-Ye)", True),
    (OURS, "MDS-MAP + Huber stress", True),
    (OURS, "DV-distance + Huber polish", True),
    (OURS + " + Huber polish", "SDP + Huber polish", False),
    (OURS + " + Huber polish", "MDS-MAP + Huber stress", False),
    (OURS + " + Huber polish", "DV-distance + Huber polish", False),
    (OURS + " + Huber polish", "Random (multi-start) + Huber polish", False),
    (OURS, "Ablation: encoder + Huber-weight steps", True),
    (OURS, "Ablation: Huber weights + learned alpha", True),
]
rows = []
for f in sorted(glob.glob("results/seed*/all_results.csv")):
    seed = f.replace('\\\\', '/').split('/')[-2]
    d = pd.read_csv(f)
    d = d[d.globally_rigid.astype(bool) & d.missing.between(10, 80)]
    for a, b, nlos_only in PAIRS:
        dd = d[d.scenario.str.endswith("nlos")] if nlos_only else d
        p = dd.pivot_table(index=["scenario", "missing", "graph_id"], columns="method", values="rmse")
        if a not in p or b not in p:
            continue
        q = p[[a, b]].dropna(); x, y = q[a], q[b]
        diff = x - y
        try:
            pv = 1.0 if (diff.abs() <= 1e-4).all() else wilcoxon(diff[diff.abs() > 1e-4]).pvalue
        except ValueError:
            pv = np.nan
        rows.append(dict(seed=seed, A=a, B=b, nlos_only=nlos_only, n=len(q),
                         win=(diff < -1e-4).mean(), loss=(diff > 1e-4).mean(),
                         mean_A=x.mean(), mean_B=y.mean(), fail_A=(x > 1.0).mean(), fail_B=(y > 1.0).mean(), p=pv))
pt = pd.DataFrame(rows)
if len(pt):
    pt.to_csv("results/paired_tests.csv", index=False)
    display(pt.round(3))
    print("A claim is safe only if A wins in EVERY seed (win > loss, p < 0.05).")
else:
    print("No per-seed results found yet.")'''

targets = [
    'gnn_loc/colab_all_in_one.ipynb',
    'gnn_loc/colab_all_in_one_extended.ipynb',
    r'C:\Users\priya\Downloads\colab_all_in_one_fixed.ipynb',
    r'C:\Users\priya\Downloads\notebookfb8f0663b9.ipynb'
]

for p in targets:
    try:
        with open(p, encoding='utf-8') as f:
            nb = json.load(f)
        for i, c in enumerate(nb['cells']):
            src = ''.join(c['source'])
            if 'PAIRS = [' in src and 'OURS' in src and c['cell_type'] == 'code':
                c['source'] = [line + '\n' for line in new_cell20_code.splitlines()]
                print(f'Updated Wilcoxon code in {p} (Cell {i})')
            if 'Decisive paired tests' in src and c['cell_type'] == 'markdown':
                c['source'] = [
                    '## Decisive paired tests (per seed)\n',
                    'Globally rigid graphs at 10-80 % missing. Wilcoxon signed-rank on per-graph RMSE; `fail` = share of graphs with RMSE > 1.0 m.\n'
                ]
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
    except Exception as e:
        print(f'Error updating {p}: {e}')

print('All notebooks updated successfully!')

# Patch Cell 18 in notebookfb8f0663b9.ipynb to skip test.pkl
try:
    with open(r'C:\Users\priya\Downloads\notebookfb8f0663b9.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    c18 = nb['cells'][18]
    src = ''.join(c18['source'])
    old_copy = 'for fpath in pkl_files:\n    fname = os.path.basename(fpath)\n    shutil.copy2(fpath, f"data/{fname}")'
    new_copy = 'for fpath in pkl_files:\n    fname = os.path.basename(fpath)\n    if fname != "test.pkl":\n        shutil.copy2(fpath, f"data/{fname}")\n    else:\n        print("[*] Skipping test.pkl copy: colab_all_in_one will generate fresh canonical test.pkl")'
    if old_copy in src:
        c18['source'] = [line + '\n' for line in src.replace(old_copy, new_copy).splitlines()]
        with open(r'C:\Users\priya\Downloads\notebookfb8f0663b9.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print('Updated Cell 18 in notebookfb8f0663b9.ipynb successfully!')
except Exception as e:
    print('Cell 18 patch error:', e)

