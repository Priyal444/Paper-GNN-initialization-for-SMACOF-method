# Anchor-scarce GNN localisation (3 anchors)

AnchorGNN encoder + learned unrolled SMACOF (dwMDS) refiner, trained once on
many synthetic graphs, compared with classical and GNN baselines on identical
test graphs.

```
pip install torch numpy scipy networkx pandas matplotlib cvxpy
python step1_data.py                      # datasets           (~3-5 min)
python step2_train.py                     # proposed model     (GPU recommended)
python step2_train.py --variant uniform   # ablations
python step2_train.py --variant huber
python step2_train.py --variant encoder_only
python step3_evaluate.py                  # comparison + figures -> results/
```

Smoke test first: put `QUICK=1` in front of each command (~2 min total).
Override counts without editing: `EPOCHS=10 TEST_PER_LEVEL=5 python ...`.

| file | role |
|---|---|
| config.py | every setting |
| common.py | graph generators (spanning tree / RGG / BA), noise, anchors, rigidity tests, features, metrics |
| model.py | encoder + learned-SMACOF refiner, losses |
| baselines.py | MDS-MAP, +stress, +Huber, dwMDS, SDP (Biswas-Ye), Yan et al. GCN, previous per-graph GraphSAGE |
| step1_data.py / step2_train.py / step3_evaluate.py | the three steps |
| run_all.ipynb | Colab version |

Ground-truth coordinates are used only in the supervised loss on synthetic
training graphs and for scoring. No method ever sees `true_xy` of a test graph.
