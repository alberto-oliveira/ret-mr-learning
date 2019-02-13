#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append('../source/')
import glob

from rankutils.cfgloader import cfgloader

import argparse

import numpy as np

completedir = lambda d: d if d[-1] == '/' else d+'/'

pathtcfg = cfgloader('../source/path_2.cfg')

def gen_fold_assign(n):

    idx = np.arange(n)
    fold_idx = np.zeros((n, 1), dtype=np.uint8)

    np.random.shuffle(idx)
    aux = idx[0:(n//2)]

    fold_idx[aux] = 1

    return fold_idx


for section in sys.argv[1:]:

    if section != "DEFAULT":

        print("-> ", section)
        rkdir = completedir(pathtcfg.get(section, 'rank'))
        rkflist = glob.glob(rkdir + "*.rk")
        n = len(rkflist)

        folds = []
        for i in range(5):
            folds.append(gen_fold_assign(n))

        folds = np.hstack(folds)

        outfile = "{0:s}{1:s}_folds.npy".format(rkdir, section)
        print("   ->", folds.shape)
        print("   ->", outfile)
        np.save(outfile, folds)