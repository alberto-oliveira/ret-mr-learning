#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append("../source/")
import argparse
import glob

from rankutils.utilities import get_label, get_query_label, completedir
from rankutils.rIO import read_rank

import numpy as np


def generate_full_labeling(rankdir, outfname, l):

    lfun = np.vectorize(get_label)

    rkflist = glob.glob(rankdir + "*.rk")
    rkflist.sort()

    labels = []

    for i, rkfpath in enumerate(rkflist):
        print(" >", os.path.basename(rkfpath))

        rk = read_rank(rkfpath)
        names = rk['name'][:l]

        qcatg = get_query_label(os.path.basename(rkfpath))
        rkcatg = lfun(names)

        labels.append((rkcatg == qcatg).astype(np.uint8))

    if not outfname.endswith('.npy'):
        outfname += ".npy"

    np.save(outfname, np.vstack(labels))

    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("rankdir", help="Directory with rankfiles.", type=str)
    parser.add_argument("outfname", help="Name of the output file.", type=str)
    parser.add_argument("-l", "--limit", help="Maximum number of ranking results to label. Default is all positions.",
                        type=int, default=np.newaxis)

    args = parser.parse_args()

    generate_full_labeling(completedir(args.rankdir), args.outfname, args.limit)
