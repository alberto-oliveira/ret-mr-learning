#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import glob
import argparse

sys.path.append("../source/")

import numpy as np

from rankutils.mappings import ranking_type_map
from rankutils.cfgloader import cfgloader


def refactor_array(arr):

    k, s, v, d = arr.shape
    out_arr = arr.reshape(k, s, d)
    out_arr = out_arr.transpose([1, 0, 2])
    out_arr = out_arr.reshape(-1, d)

    return out_arr


def get_output_name(inname):

    parts = inname.split(".", 2)

    return "{0:s}.seq-label.{1:s}".format(parts[0], parts[2])


if __name__ == "__main__":

    fvname = sys.argv[1]

    pathcfg = cfgloader("../source/path_2.cfg")

    for key, values in ranking_type_map.items():

        for v in values:
            expkey = "{0:s}_{1:03d}".format(key, v)

            print("converting: {0:s} - {1:s}".format(expkey, fvname))

            in_fvdir = pathcfg.get(expkey, 'feature')
            out_sfvdir = pathcfg.get(expkey, 'seqfeature')

            in_fvpath = glob.glob("{0:s}*{1:s}*".format(in_fvdir, fvname))[0]
            out_sfvpath = "{0:s}{1:s}".format(out_sfvdir, get_output_name(os.path.basename(in_fvpath)))

            print("   ->", os.path.basename(in_fvpath))
            print("   ->", os.path.basename(out_sfvpath))
            print()

            in_fv = np.load(in_fvpath)

            out_fv = dict()

            out_fv['features'] = refactor_array(in_fv['features'])

            out_fv['labels'] = refactor_array(in_fv['labels'])






