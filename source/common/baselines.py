#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

import numpy as np

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