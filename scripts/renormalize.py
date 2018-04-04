#!/usr/bin/env python
#-*- coding utf-8 -*-

import sys, os
import glob

sys.path.append("/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/")

from sklearn.preprocessing import MinMaxScaler
import numpy as np

from common.rIO import read_rank, write_rank

if __name__ == "__main__":

    rkdir = sys.argv[1]
    if rkdir[-1] != '/': rkdir += '/'

    outdir = sys.argv[2]
    if outdir[-1] != '/': outdir += '/'

    rkflist = glob.glob(rkdir + "*.rk")
    rkflist.sort()

    for rkfpath in rkflist:

        print(" .", os.path.basename(rkfpath))
        rank = read_rank(rkfpath)

        SCL = MinMaxScaler((1.0, 2.0))
        rank['normd'] = (3.0 - SCL.fit_transform(rank['dists'].reshape(-1, 1))).reshape(-1)

        write_rank(outdir + os.path.basename(rkfpath), rank)

