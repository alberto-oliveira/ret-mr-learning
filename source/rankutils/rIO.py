#!/usr/bin/env python
#-*- coding utf-8 -*-

import numpy as np
import glob

rk_dtype = dict(names=('name', 'score'),
                formats=('U100', np.float64))

rel_dtype = dict(names=('name', 'rel'),
                 formats=('U100', np.int32))


def read_rank(fpath, colname='', rows=np.newaxis):

    if colname != '':
        c = rk_dtype['names'].index(colname)
        t = rk_dtype['formats'][c]
        arr = np.loadtxt(fpath, usecols=c, dtype=t)
    else:
        arr = np.loadtxt(fpath, dtype=rk_dtype)

    return np.array(arr[:rows])

def write_rank(fpath, rank):

    np.savetxt(fpath, rank, fmt="%-50s %10.5f %10.5f %10.5f %10.5f")


def read_relfile(fpath):

    arr = np.loadtxt(fpath, dtype=rel_dtype)

    return arr

def write_relfile(fpath, relarray):

    np.savetxt(fpath, relarray, fmt="%-50s %1d")

