#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import glob

import argparse

import numpy as np

completedir = lambda d: d if d[-1] == '/' else d+'/'

def gen_fold_idx(n, F):

    f_size = np.floor(n/F).astype(np.int32)
    f_rem = n % F

    l = []

    f_num = 0
    added = 0

    while added < n:

        if f_rem > 0:
            aux = np.full((1, f_size + 1), f_num, dtype=np.int32)
            added += f_size + 1
            f_rem -= 1
            l.append(aux)
        else:
            aux = np.full((1, f_size), f_num, dtype=np.int32)
            added += f_size
            l.append(aux)

        f_num += 1

    idx = np.hstack(l)
    assert idx.shape[1] == n, "Inconsistent shape between indexes <{0:d}> and number of files <{1:d}>."\
                              .format(idx.shape[1], n)

    return idx

def generate_folds(rkdir, outfile, F, R):

    np.random.seed(93311)

    flist = glob.glob(rkdir + "*.rk")
    total = len(flist)

    fold_list = []

    idx = gen_fold_idx(total, F).reshape(-1, 1)

    for i in range(R):
        idx_s = np.array(idx, copy=True)
        np.random.shuffle(idx_s)

        fold_list.append(idx_s)

    folds = np.hstack(fold_list)
    np.save(outfile, folds)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('rkdir', help="directory containing rk files.", type=str)
    parser.add_argument('outfile', help="output file.", type=str)
    parser.add_argument('nfolds', help="number of folds per round.", type=int)
    parser.add_argument('nrounds', help="number of rounds.", type=int)

    args = parser.parse_args()

    generate_folds(completedir(args.rkdir), args.outfile, args.nfolds, args.nrounds)
