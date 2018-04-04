#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import ipdb as pdb

import numpy as np

from common.evaluation import Evaluator

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
        try:
            nacc = Evaluator.norm_acc(gt, tst)
        except IndexError:
            pdb.set_trace()
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
