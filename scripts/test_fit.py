#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import shutil
sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source")

from common.weibull import *
from common.utilities import *

import numpy as np
import pickle

from time import perf_counter

rkprefb = "/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/input-ranks/10-fold/" \
          "brodatz-letor/desc2-CCOM/"

lprefb = "/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/input-labels/10-fold/" \
         "brodatz-letor/desc2-CCOM/rel-prediction/brodatz_desc2_f"

rkprefo = "/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/input-ranks/10-fold/" \
          "oxford/desc1-SURF3000x500/"

lprefo = "/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/input-labels/10-fold/" \
         "oxford/desc1-SURF3000x500/rel-prediction/oxford_desc1_f"

if not os.path.isfile("dump.wbl"):
    X_list = []
    y_list = []
    for f in range(1, 10):
        foldir = rkprefo + "fold_{f:d}".format(f=f)
        lblpath = lprefo + "{f:03d}_top10_irp_lbls.npy".format(f=f)

        X_list.append(preprocess_ranks(foldir, colname='score', maxsz=8000))
        y_list.append(np.load(lblpath))

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    wbl = WeibullMR(opt_metric='MCC', notop=False, verbose=True)

    tic = perf_counter()
    wbl.fit(X, y)
    tac = perf_counter()
else:
    with open("dump.wbl", 'rb') as inpf:
        tic = perf_counter()
        wbl = pickle.load(inpf)
        tac = perf_counter()

print("F:", wbl.F)
print("Z:", wbl.Z)

print("-- Predict --")

testfolddir = rkprefo + "fold_0/"

testlblpath = lprefo + "000_top10_irp_lbls.npy"

test_X = preprocess_ranks(testfolddir, colname='score', maxsz=8000)
gt = np.load(testlblpath).reshape(-1)

predicted, _ = wbl.predict(test_X)
predicted = predicted.reshape(-1)

print("\n   {0:<20s}".format("groundtruth:"), end="", file=sys.stdout, flush=True)
for k in range(gt.shape[0]):
    if k != 0 and k % 10 == 0:
        print("| {0:d}".format(gt[k]), end="", file=sys.stdout, flush=True)
    else:
        print(" {0:d}".format(gt[k]), end="", file=sys.stdout, flush=True)
print("\n   {0:<20s}".format("predicted:"), end="", file=sys.stdout, flush=True)
for k in range(predicted.shape[0]):
    if k != 0 and k % 10 == 0:
        print("| {0:d}".format(predicted[k]), end="", file=sys.stdout, flush=True)
    else:
        print(" {0:d}".format(predicted[k]), end="", file=sys.stdout, flush=True)

print("\n -> FITTING TIME ELAPSED: {0:0.5f}s".format(tac-tic))

try:
    with open('dump.wbl', 'wb') as outf:
        pickle.dump(wbl, outf)
except:
    print("Problem dumping file")
    os.remove('dump.wbl')




