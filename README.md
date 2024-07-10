# DHMAE

This is official code for the SIGIR 2024 full paper:

**DHMAE: A Disentangled Hypergraph Masked Autoencoder for Group Recommendation** 

## Datasets

We use five widely used datasets.

**CAMRa2011**, **Mafengwo** and **Mafengwo-S** are exactly the
same as the [ConsRec](https://github.com/FDUDSDE/WWW2023ConsRec) used.

For **MovieLens**, the dataset is derived from the real-world public dataset [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/).

For **Weeplaces-S**, the dataset is obtained based on Weeplaces used in the [HHGR](https://github.com/0411tony/HHGR).

For the models of the baseline methods, we refer to the publicly available implementations in [WWW2023GroupRecBaselines](https://github.com/FDUDSDE/WWW2023GroupRecBaselines).

## Run
Run all datasets at once:

> bash run.sh

For more running options, please refer to _main.py_

## Note

As suggested by the reviewers, the experiments in this paper employ the **all-ranking protocol** rather than the sampling evaluation of previous methods. However, our model implementation is able to seamlessly migrate to the previous data loading and evaluation codes without more effort. And on sampling evaluation, DHMAE still has a significant performance improvement over other methods.