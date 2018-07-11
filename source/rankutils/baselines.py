#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np
from sklearn.metrics import confusion_matrix

np.random.seed(93311)

def baseline_ran(n, k, p=-1):

    if p < 0 or p > 1:
        prob = np.random.random_sample()
    else:
        prob = p

    out = np.random.ranf(size=(n, k))
    out = (out <= prob).astype(dtype=np.uint8)

    return out


def baseline_fulltop(n, k):
    return np.ones((n, k), dtype=np.uint8)


def baseline_halftop(n, k):

    h_k = int(np.floor(float(k)/2))
    return np.hstack([np.ones((n, h_k), dtype=np.uint8), np.zeros((n, k-h_k), dtype=np.uint8)])

def baseline_maxnacc(labels):

    res = []
    for lb in labels:

        res.append(gen_baseline_max_nacc(lb.reshape(-1)))

    return np.vstack(res)

def gen_baseline_max_nacc(gt):

    tst = np.zeros(gt.shape, dtype=gt.dtype)
    res = np.zeros(gt.shape, dtype=gt.dtype)
    best = -np.inf
    i = 0

    while i <= tst.shape[0]:

        #print("i:", i)
        #print("tst: ", tst)
        #print("gt : ", gt)

        nacc = get_norm_acc(gt, tst)
        #print("nacc = {0:0.3f} | best = {1:0.3f}".format(nacc, best))
        if nacc > best:
            res[:i] = 1
            best = nacc
            #print(" -- updating best")
        #print("")

        try:
            tst[i] = 1
        except IndexError:
            pass
        i += 1

    return res

def get_norm_acc(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    cfmat = confusion_matrix(y_true, y_pred)

    if cfmat.size == 1:
        nacc = 1.0

    else:
        TN = cfmat[0, 0]
        FN = cfmat[1, 0]

        FP = cfmat[0, 1]
        TP = cfmat[1, 1]

        TNR = TN / (TN + FP)
        TPR = TP / (TP + FN)
        # print(TNR)
        # print(TPR)

        nacc = (TNR + TPR) / 2

    return nacc
