#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

sys.path.append("../source/")

import numpy as np

from rankutils.cfgloader import cfgloader
from rankutils.rIO import read_rank

from tqdm import tqdm

limit = 1000


def get_index_array(namerray, namelist):

    nsort = np.argsort(namelist)
    pos = np.searchsorted(namelist[nsort], namerray)

    return nsort[pos]


pathcfg = cfgloader('../source/path_2.cfg')

nl_dtype = dict(names=('name', 'nf', 'cid'), formats=('U100', np.int32, np.int32))


for s in pathcfg.sections():

    if s != 'DEFAULT':

        dset = s.split("_", 1)[0]

        rkdir = pathcfg.get(s, 'rank')

        try:
            namelist = np.loadtxt(pathcfg.get(s, 'namelist'), dtype=nl_dtype)
        except IOError:
            continue

        print("{0:s}_{1:s}...".format(dset, pathcfg.get(s, 'rktpdir')), end='', flush=True)
        indices = []
        scores = []

        rkflist = glob.glob(rkdir + "*.rk")
        rkflist.sort()

        for rkfpath in tqdm(rkflist, ncols=75, desc='Rank File', total=len(rkflist)):
            rank = read_rank(rkfpath)

            scores.append(rank['score'][0:limit].reshape(1, -1).copy())
            indices.append(get_index_array(rank['name'][0:limit], namelist['name']).reshape(1, -1))

            #print("scores bytes: ", scores[-1].nbytes)
            #print("indices bytes: ", indices[-1].nbytes)

            del rank

            assert scores[-1].shape == indices[-1].shape, "Incompatible score and indices shapes"

        outfile = "{0:s}{1:s}_{2:s}_scores.npy".format(rkdir, dset, pathcfg.get(s, 'rktpdir'))
        np.save(outfile, np.vstack(scores))

        outfile = "{0:s}{1:s}_{2:s}_indices.npy".format(rkdir, dset, pathcfg.get(s, 'rktpdir'))
        np.save(outfile, np.vstack(scores))

        print('Done!')







