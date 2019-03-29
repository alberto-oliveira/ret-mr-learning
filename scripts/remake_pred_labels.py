#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append("../source/")
import argparse
import glob

import numpy as np

from rankutils.utilities import merge_kfolds_rounds
from rankutils.cfgloader import cfgloader

dataset = sys.argv[1]
method = sys.argv[2]
fi = int(sys.argv[3])
outfpath = sys.argv[4]


pathcfg = cfgloader('../source/path_2.cfg')

foldsfpath = glob.glob("{0:s}*.npy".format(pathcfg.get(dataset, 'rank')))[0]
folds = np.load(foldsfpath)
predlblpath = "{0:s}{1:s}/".format(pathcfg.get(dataset, 'output'), method)

lbllist = merge_kfolds_rounds(predlblpath, folds)


np.save(outfpath, lbllist[fi])

