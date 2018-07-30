#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import glob

import numpy as np


sys.path.append("../source/")
from rankutils.utilities import safe_create_dir

completedir = lambda d: d + '/' if d[-1] != '/' else d

def convert_rk_format(indir, outdir):

    indtype = dict(names=('name', 'votes', 'nvotes', 'dist', 'ndist'),
                   formats=('U100', np.float64, np.float64, np.float64, np.float64))

    outdtype = dict(names=('name', 'score'),
                   formats=('U100', np.float64))

    safe_create_dir(outdir)

    rkflist = glob.glob(indir + "*.rk")
    rkflist.sort()

    for rkfpath in rkflist:
        rkfname = os.path.basename(rkfpath)
        print(". converting ", rkfname)

        outfpath = "{dir:s}{fname:s}".format(dir=outdir, fname=rkfname)

        arr = np.loadtxt(rkfpath, dtype=indtype)
        aux = [tple for tple in zip(arr['name'], arr['votes'])]

        outarr = np.array(aux, dtype=outdtype)

        np.savetxt(outfpath, outarr, fmt="%-50s %10.5f")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Directory with input old format rank files", type=str)
    parser.add_argument("outdir", help="Output directory for new format rank files", type=str)

    args = parser.parse_args()

    indir = completedir(args.indir)
    outdir = completedir(args.outdir)

    convert_rk_format(indir, outdir)
