#!/usr/bin/env python
#-*- coding utf-8 -*-

import sys, os
import glob

import numpy as np

inpdir = sys.argv[1]
if inpdir[-1] != '/':
    inpdir += '/'

outfile = sys.argv[2]


flist = glob.glob(inpdir + '*.txt')
flist.sort()


relarrays = []
for relfile in flist:
    print("-> ", os.path.basename(relfile))
    rel = np.loadtxt(relfile, dtype=np.uint8, usecols=1)
    relarrays.append(rel.reshape(1, -1))

np.save(outfile, np.vstack(relarrays))



