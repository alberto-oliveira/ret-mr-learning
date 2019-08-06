#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append('../source/')

import numpy as np

def load_mapping(mapfile):

    catmap = dict()
    catnum = dict()

    c = 0
    with open(mapfile, 'r') as f:

        for row in f:
            parts = row.strip('\n').split()

            name = parts[0]
            cat = parts[1]

            if cat not in catnum:
                catnum[cat] = c
                c += 1

            catmap[name] = catnum[cat]

    return catmap, catnum, c


mapfile = sys.argv[1]
namelistfile = sys.argv[2]
outfile = sys.argv[3]

dt = dict(names=('name', 'numfeat'), formats=('U100', np.int32))
namearray = np.loadtxt(namelistfile, dtype=dt)

catmap, catnum, c = load_mapping(mapfile)

cat_array = np.zeros(namearray['name'].size, dtype=np.int32)

for i, name in enumerate(namearray['name']):

    name_ = os.path.splitext(name)[0]

    if name_ in catmap:
        cat_array[i] = catmap[name_]

    else:
        cat_array[i] = c
        c += 1

outdt = dict(names=('name', 'numfeat', 'cid'), formats=('U100', np.int32, np.int32))
aux = zip(namearray['name'], namearray['numfeat'], cat_array)
outarray = np.array([a for a in aux], dtype=outdt)

np.savetxt(outfile, outarray, fmt="%-50s %05d %03d")