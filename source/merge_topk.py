#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse
import ipdb as pdb

from common.mappings import descriptor_map
from common.cfgloader import cfgloader

import numpy as np

sys.path.append("../source/")
from common.rIO import read_rank, write_rank, rk_dtype

completedir = lambda d: d if d[-1] == "/" else d + "/"

k = 10

def merge(rank_a, pred_a, rank_b, pred_b):

    new_rank = []
    added = set([])

    pos = 0
    reli_a = np.argwhere(pred_a == 1).reshape(-1)
    n_reli_a = np.argwhere(pred_a == 0).reshape(-1)
    reli_b = np.argwhere(pred_b == 1).reshape(-1)
    n_reli_b = np.argwhere(pred_b == 0).reshape(-1)


    for i in reli_a:
        if rank_a['name'][i] not in added:
            new_rank.append(rank_a[i])
            added.add(rank_a['name'][i])
            pos += 1

    for i in reli_b:
        if rank_b['name'][i] not in added:
            new_rank.append(rank_b[i])
            added.add(rank_b['name'][i])
            pos += 1

    for i in n_reli_a:
        if rank_a['name'][i] not in added:
            new_rank.append(rank_a[i])
            added.add(rank_a['name'][i])
            pos += 1

    for i in n_reli_b:
        if rank_b['name'][i] not in added:
            new_rank.append(rank_b[i])
            added.add(rank_b['name'][i])
            pos += 1
    #pdb.set_trace()
    while len(new_rank) < 1000:
        new_rank.append(rank_a[pos])
        pos += 1

    new_rank = np.array(new_rank, dtype=rk_dtype)

    for i in range(new_rank['name'].shape[0]):
        parts = new_rank['name'][i].split('\'')
        new_rank['name'][i] = parts[1]
    #pdb.set_trace()
    return new_rank


def merge_topk(dataset, desc_a, desc_b, method, outdir):
    print(rk_dtype)
    round = 0
    pathcfg = cfgloader("path_2.cfg")

    key_a = "{0:s}_desc{1:d}".format(dataset, desc_a)
    key_b = "{0:s}_desc{1:d}".format(dataset, desc_b)

    rkdir_a = pathcfg['rank'][key_a]
    rkdir_b = pathcfg['rank'][key_b]

    folds_a = np.load(glob.glob(rkdir_a + "*.folds.npy")[0])[:, round]
    folds_b = np.load(glob.glob(rkdir_b + "*.folds.npy")[0])[:, round]

    #print('\n' + pathcfg['output'][key_a] + method + "/rel-prediction/*r{0:03d}*".format(round))
    #print('\n' + pathcfg['output'][key_b] + method + "/rel-prediction/*r{0:03d}*".format(round))

    ppath_a = glob.glob(pathcfg['output'][key_a] + method + "/rel-prediction/*r{0:03d}*.npy".format(round))
    ppath_a.sort()

    ppath_b = glob.glob(pathcfg['output'][key_b] + method + "/rel-prediction/*r{0:03d}*.npy".format(round))
    ppath_b.sort()

    assert len(ppath_a) == 2, "Could not find predictions for {0:s}".format(key_a)
    assert len(ppath_b) == 2, "Could not find predictions for {0:s}".format(key_b)

    pred_pair_a = [np.load(ppath_a[0]), np.load(ppath_a[1])]
    pred_pair_b = [np.load(ppath_b[0]), np.load(ppath_b[1])]

    fullpred_a = []
    fullpred_b = []

    n0 = 0
    n1 = 0
    for fa in folds_a:
        if fa == 0:
            fullpred_a.append(pred_pair_a[fa][n0, :])
            n0+=1
        if fa == 1:
            fullpred_a.append(pred_pair_a[fa][n1, :])
            n1+=1

    n0 = 0
    n1 = 0
    for fb in folds_b:
        if fb == 0:
            fullpred_b.append(pred_pair_b[fb][n0:n0+1, :])
            n0 += 1
        if fb == 1:
            fullpred_b.append(pred_pair_b[fb][n1:n1+1, :])
            n1 += 1

    rkfiles_a = glob.glob(rkdir_a + "/*.rk")
    rkfiles_a.sort()

    rkfiles_b = glob.glob(rkdir_b + "/*.rk")
    rkfiles_b.sort()

    assert len(rkfiles_a) == len(rkfiles_b), "Inconsistent number of ranks for rank directory a <{0:d}> and b <{1:d}>"\
                                             .format(len(rkfiles_a), len(rkfiles_b))

    pos = 0
    for rkfpath_a, rkfpath_b in zip(rkfiles_a, rkfiles_b):

        assert os.path.basename(rkfpath_a) == os.path.basename(rkfpath_b),\
        "Rank files <{0:s}> and <{1:s}> diverge!".format(os.path.basename(rkfpath_a), os.path.basename(rkfpath_b))

        print("->" + os.path.basename(rkfpath_a))

        rank_a = read_rank(rkfpath_a)
        rank_b = read_rank(rkfpath_b)

        pred_a = fullpred_a[pos]
        pred_b = fullpred_b[pos]

        rank_merge = merge(rank_a, pred_a, rank_b, pred_b)
        rankoutname = "{0:s}{1:s}".format(outdir, os.path.basename(rkfpath_a))
        write_rank(rankoutname, rank_merge)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(descriptor_map.keys()))

    parser.add_argument("desc_a", help="First descriptor number for merging.",
                        type=int)

    parser.add_argument("desc_b", help="Second descriptor number for merging.",
                        type=int)

    parser.add_argument("method", help="Name of the method used for the predictions.",
                        type=str)

    parser.add_argument("outdir", help="Output directory.",
                        type=str)

    args = parser.parse_args()

    merge_topk(args.dataset, args.desc_a, args.desc_b, args.method, args.outdir)