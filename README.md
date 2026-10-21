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

Transfer across graph types (train on some of spanning-tree / RGG / BA, test on all three;
every training set has the same total size, N_TRAIN):

```
python step2_train.py --train-on random       # train on 1 type, test on the other 2
python step2_train.py --train-on rgg
python step2_train.py --train-on ba
python step2_train.py --train-on random+rgg   # train on 2 types, test on the 3rd
python step2_train.py --train-on random+ba
python step2_train.py --train-on rgg+ba
python step2_train.py --train-on all          # train on all 3
python step3_evaluate.py --transfer           # after the main step 3; -> fig_transfer_matrix.png
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
