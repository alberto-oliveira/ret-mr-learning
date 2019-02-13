# /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
import os

import numpy as np

rkdir = sys.argv[1]
if rkdir[-1] != '/':
    rkdir += '/'

old_dt = dict(names=('name', 'score', 'v1', 'v2', 'v3'), formats=('U100', np.float64, np.float64, np.float64, np.float64))

new_dt = dict(names=('name', 'score'),
                formats=('U100', np.float64))


def reformat(rkfpath, out=''):

    if out == '':
        out = rkfpath

    try:
        rk = np.loadtxt(rkfpath, dtype=old_dt)
        np.savetxt(rkfpath, rk[['name', 'score']], fmt="%-50s %10.5f")
    except IndexError:
        pass


rkflist = glob.glob(rkdir + "*.rk")
rkflist.sort()

for rkfpath in rkflist:
    reformat(rkfpath)

