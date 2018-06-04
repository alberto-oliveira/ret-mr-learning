#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse
import shutil

import numpy as np

sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source")
from rankutils.utilities import safe_create_dir

completedir = lambda d: d if d[-1] == '/' else d+'/'

def get_ranks_and_labels(rkdir, lblpath, outdir, k, d):

    rkflist = np.array(glob.glob(rkdir + "*.rk"))
    r = rkflist.shape[0]

    labels = np.load(lblpath)
    print("shape: ", labels.shape)

    idxs = np.arange(0, r, dtype=np.int32)
    np.random.shuffle(idxs)

    f = np.floor(r/k).astype(np.int32)
    f_r = r % k

    print("Fold size:",f)
    print("Fold remainder:", f_r)

    """
    cut = []
    ct = f
    while ct < r:

        if f_r > 0:
            cut.append(ct+1)
            f_r -= 1
            ct += f + 1
        else:
            cut.append(ct)
            ct += f

    fold_idxs = np.split(idxs, cut)
    """

    fold_count = 0
    sidx = 0
    while sidx < r:

        eidx = sidx + f
        if f_r > 0:
            eidx += 1
            f_r -=1

        idx_slice = idxs[sidx:eidx]
        print("[{1:d}:{2:d}] Fold {0:d} shape:".format(fold_count, sidx, eidx), idx_slice.shape)

        outlblfile = "{0:s}f{1:03d}_top10_irp_lbls.npy".format(outdir, fold_count)
        if not d:
            print("  -> labels shape:", labels[idx_slice].shape)
            np.save(outlblfile, labels[idx_slice])

        outfolddir = outdir + "fold_{0:d}/".format(fold_count)
        safe_create_dir(outfolddir)

        if not d:
            for rkfpath in rkflist[idx_slice]:
                shutil.copy2(rkfpath, outfolddir + os.path.basename(rkfpath))
        
        sidx = eidx

        fold_count += 1

    print("\n---")

    """
    for i, fidx in enumerate(fold_idxs):


        outfolddir = outdir + "fold_{0:d}/".format(i)
        safe_create_dir(outfolddir)

        for rkfpath in rkflist[fidx]:
            shutil.copy2(rkfpath, outfolddir + os.path.basename(rkfpath))
    """






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("rkdir", help="Directory with rank files.", type=str)
    parser.add_argument("lblpath", help="Groundtruth labels filepath", type=str)
    parser.add_argument("outdir", help="Output directory.", type=str)
    parser.add_argument("-k", "--nfolds", help="Number of folds.", type=int, default=10)
    parser.add_argument("-d", "--dry_run", help="Does not move files.", action="store_true")

    args = parser.parse_args()

    get_ranks_and_labels(completedir(args.rkdir), args.lblpath, completedir(args.outdir), args.nfolds, args.dry_run)