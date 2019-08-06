#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

bseed = 93311

np.random.seed(bseed)

#try:
feat_pack = np.load(sys.argv[1])

nk, n, v, d = feat_pack['features'].shape

features = feat_pack['features'].reshape(nk, n, d)
labels = feat_pack['labels'].reshape(nk, n, 1)

features_ = features[0, 0:11, :]
labels_ = labels[0, 0:11]

#except:

#    features_ = np.array([[1, 2],
#                          [3, 4],
#                          [5, 6],
#                          [7, 8],
#                          [9, 10],
#                          [11, 12],
#                          [13, 14],
#                          [15, 16],
#                          [17, 18],
#                          [19, 20]], dtype=np.float64)
#
#    labels_ = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8).reshape(-1, 1)


rstratkfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=bseed)
splitgen = rstratkfold.split(features_, labels_)

for i in range(5):
    print("Round", i)

    train_idx, test_idx = next(splitgen)
    print("  Split 1")
    print("    train:", train_idx)
    print("    test:", test_idx)

    train_idx, test_idx = next(splitgen)
    print("  Split 2")
    print("    train:", train_idx)
    print("    test:", test_idx)

    print("---")


