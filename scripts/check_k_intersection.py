#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

import numpy as np

sys.path.append("../source/common/")
from common.rIO import read_rank

completedir = lambda d: d if d[-1] == "/" else d+"/"

def check_intersection(rkdir_a, rkdir_b, k):

    rkflist = glob.glob(rkdir_a + "*.rk")
    rkflist.sort()

    isection = []
    for rkfpath_a in rkflist:

        rkfpath_b = rkdir_b + os.path.basename(rkfpath_a)
        assert os.path.isfile(rkfpath_b), "Could not find rank <{0:s}> in directory <{1:s}>."\
                                          .format(os.path.basename(rkfpath_a), rkdir_b)

        rk_a = read_rank(rkfpath_a)
        rk_b = read_rank(rkfpath_b)

        klist_a = set(rk_a['name'][0:k])
        klist_b = set(rk_b['name'][0:k])

        isect_ratio = float(len(klist_a.intersection(klist_b)))/k
        isection.append(isect_ratio)

        print("{0:0.3f}".format(isect_ratio))

    print("{0:0.3f}".format(np.mean(isection)))

    return







if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("rkdir_a", help="First rank directory", type=str)
    parser.add_argument("rkdir_b", help="Second rank directory", type=str)
    parser.add_argument("k", help="# of top positions for comparison", type=int)

    args = parser.parse_args()

    check_intersection(completedir(args.rkdir_a), completedir(args.rkdir_b), args.k)
