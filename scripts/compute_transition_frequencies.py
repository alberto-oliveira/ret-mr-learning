#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
sys.path.append("../source/")
import glob
import argparse

from rankutils.utilities import get_query_classname, get_classname, completedir, read_and_convert

import numpy as np


def compute_transition_frequencies(rankdir, outprefix, scl):

    rkflist = glob.glob(rankdir + "*.rk")
    rkflist.sort()

    t_freq = np.zeros(4, dtype=np.int64)
    diffs_00 = []
    diffs_01 = []
    diffs_10 = []
    diffs_11 = []

    allmax = -1
    allmax_i = -1
    allmax_f = ""

    s = 10
    e = 1000

    for rkfpath in rkflist:

        print("> ", os.path.basename(rkfpath))

        qcat = get_query_classname(os.path.basename(rkfpath))
        scores, names, _ = read_and_convert(rkfpath, scale=scl, convert=True)
        scores = scores[s:e]
        names = names[s:e]

        prevrel = (get_classname(names[0]) == qcat)

        diffs = scores[0:-1] - scores[1:]

        for i in range(1, scores.size):
            cat = get_classname(names[i])
            rel = (cat == qcat)

            if not prevrel and not rel:
                t_freq[0] += 1
                diffs_00.append((i, diffs[i-1]))
            elif not prevrel and rel:
                t_freq[1] += 1
                diffs_01.append((i, diffs[i-1]))
            elif prevrel and not rel:
                t_freq[2] += 1
                diffs_10.append((i, diffs[i-1]))
            elif prevrel and rel:
                t_freq[3] += 1
                diffs_11.append((i, diffs[i-1]))

            prevrel = rel

        maxdiff_i = np.argmax(diffs)

        print("    -> Position {0:d} : {1:0.6f}".format(maxdiff_i+s+1, diffs[maxdiff_i]))

        if diffs[maxdiff_i] > allmax:
            allmax = diffs[maxdiff_i]
            allmax_i = maxdiff_i
            allmax_f = rkfpath

    dt = dict(names=('i', 'd'),
              formats=(np.int32, np.float64))
    print(" \n. File: {0:s} -- Position {1:d} : {2:0.2f}".format(allmax_f, allmax_i+s+1, allmax))
    np.savez("{prefix:s}.t_diffs.{start:d}:{end:d}.npz".format(prefix=outprefix, start=s, end=e),
             diffs_00=np.array(diffs_00, dtype=dt), diffs_01=np.array(diffs_01, dtype=dt),
             diffs_10=np.array(diffs_10, dtype=dt), diffs_11=np.array(diffs_11, dtype=dt))
    np.save("{prefix:s}.t_freqs.{start:d}:{end:d}.npy".format(prefix=outprefix, start=s, end=e), t_freq)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("rankdir", help="Directory with rank files", type=str)
    parser.add_argument("outprefix", help="Prefix for output files", type=str)
    parser.add_argument("-s", "--scale", help="Scale the data before computing differences", action="store_true")

    args = parser.parse_args()

    rankdir = completedir(args.rankdir)
    outprefix = args.outprefix

    compute_transition_frequencies(rankdir, outprefix, args.scale)

