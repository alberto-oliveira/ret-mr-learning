#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

sys.path.append("../source/")

from collections import OrderedDict

import numpy as np

from rankutils.mappings import ranking_type_map
from rankutils.cfgloader import cfgloader
from rankutils.utilities import safe_create_dir

#import ipdb as pdb

ranks_to_convert = OrderedDict(oxford=[1, 2, 3, 4, 5],
                               corel=[1, 2],
                               places365=[1, 2, 3],
                               vggfaces=[1, 2],
                               imagenet=[1, 2, 3],
                               unicamp=[1, 2, 3, 4, 5])#,
                               #MPEG7=[1, 2, 3, 4, 5],
                               #multimodal=[3, 4, 6, 10, 12])


def refactor_array(arr):

    k, s, v, d = arr.shape
    out_arr = arr.reshape(k, s, d)
    out_arr = out_arr.transpose([1, 0, 2])
    #out_arr = out_arr.reshape(-1, d)

    return out_arr


def get_output_name(inname):

    parts = inname.split(".", 2)

    return "{0:s}.seq-label.{1:s}".format(parts[0], parts[2])


if __name__ == "__main__":

    fvname = sys.argv[1]

    pathcfg = cfgloader("../source/path_2.cfg")

    for key, values in ranks_to_convert.items():

        for v in values:
            expkey = "{0:s}_{1:03d}".format(key, v)

            print("converting: {0:s} - {1:s}".format(expkey, fvname))

            in_fvdir = pathcfg.get(expkey, 'feature')
            out_sfvdir = pathcfg.get(expkey, 'seqfeature')

            #pdb.set_trace()

            try:
                in_fvpath = glob.glob("{0:s}*{1:s}*".format(in_fvdir, fvname))[0]
            except IndexError as ie:
                print("Not found. Skipping")
                continue

            out_sfvpath = "{0:s}{1:s}".format(out_sfvdir, get_output_name(os.path.basename(in_fvpath)))

            print("   ->", in_fvpath)
            print("   ->", out_sfvpath)
            print()

            in_fv = np.load(in_fvpath)

            out_fv = dict()

            out_fv['features'] = refactor_array(in_fv['features'])

            out_fv['labels'] = refactor_array(in_fv['labels'])

            safe_create_dir(out_sfvdir)
            np.savez_compressed(out_sfvpath, **out_fv)






