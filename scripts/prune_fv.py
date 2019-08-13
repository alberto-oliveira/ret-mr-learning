#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

from collections import OrderedDict

sys.path.append("../source/")

from rankutils.cfgloader import cfgloader
from rankutils.utilities import safe_create_dir

import numpy as np

ranks_to_convert = OrderedDict(places365=[1, 2],
                               vggfaces=[1, 2],
                               imagenet=[1, 3],
                               unicamp=[1, 2, 4])

if __name__ == "__main__":

    fvname = sys.argv[1]
    outfvname = sys.argv[2]
    nk = int(sys.argv[3])

    pathcfg = cfgloader("../source/path_2.cfg")

    for key, values in ranks_to_convert.items():

        for v in values:
            expkey = "{0:s}_{1:03d}".format(key, v)

            print("converting: {0:s} - {1:s}".format(expkey, fvname))

            sfvdir = pathcfg.get(expkey, 'seqfeature')

            # pdb.set_trace()

            in_fvpath = glob.glob("{0:s}*{1:s}*".format(sfvdir, fvname))[0]
            out_sfvpath = "{0:s}{1:s}".format(sfvdir, outfvname)

            print("   ->", in_fvpath)
            print("   ->", out_sfvpath)
            print()

            in_fv = np.load(in_fvpath)

            n, ck, d = in_fv['features'].shape

            out_fv = dict()

            out_fv['features'] = in_fv['features'][:, 0:nk, :]

            out_fv['labels'] = in_fv['labels'][:, 0:nk, :]

            np.savez_compressed(out_sfvpath, **out_fv)

