#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append("../source/")
import argparse
import glob

import numpy as np

from rankutils.utilities import merge_kfolds_rounds
from rankutils.cfgloader import cfgloader


def gen_outfpath(outdir, dataset, rktpdir, sufix):

    d = dataset.split('_')[0]

    return "{0:s}{1:s}.{2:s}.{3:s}.npy".format(outdir, d, rktpdir, sufix)


dataset = sys.argv[1]
method = sys.argv[2]
fi = int(sys.argv[3])
outdir = sys.argv[4]
sufix = sys.argv[5]


pathcfg = cfgloader('../source/path_2.cfg')

foldsfpath = glob.glob("{0:s}*.npy".format(pathcfg.get(dataset, 'rank')))[0]
folds = np.load(foldsfpath)
predlblpath = "{0:s}{1:s}/".format(pathcfg.get(dataset, 'output'), method)

lbllist = merge_kfolds_rounds(predlblpath, folds)

outfpath = gen_outfpath(outdir, dataset, pathcfg.get(dataset, 'rktpdir'), sufix)
np.save(outfpath, lbllist[fi])

