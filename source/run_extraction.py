# /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import glob
import time

import numpy as np

from rankutils.extraction import Extractor
from rankutils.rIO import *
from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, getbasename, get_index
from rankutils.mappings import ranking_type_map
from rankutils.statistical import ev_density_approximation

import ipdb as pdb

import matlab
import matlab.engine
matlab_engine = matlab.engine.start_matlab()


def run_extraction(dataset_choices, expconfig):

    np.set_printoptions(precision=3, linewidth=500, suppress=True)

    pathcfg = cfgloader("path_2.cfg")
    expcfg = cfgloader(expconfig)

    extr = None

    expname = expcfg.get('DEFAULT', 'expname')

    for dataset in dataset_choices:
        for r in dataset_choices[dataset]:

            dkey = "{0:s}_{1:03d}".format(dataset, r)
            rktpname = pathcfg[dkey]['rktpdir']

            print(". Running Extraction on: {0:s}.{1:s}".format(dataset, rktpname))
            print(". Experiment: ", expname)

            lblfpath = glob.glob(pathcfg[dkey]["label"] + "*" + rktpname + "*")[0]

            try:
                collmfpath = glob.glob(pathcfg[dkey]["collmatches"] + "*" + rktpname + "*db_matches*")[0]
            except (IndexError, KeyError):
                collmfpath = ''

            safe_create_dir(pathcfg[dkey]["feature"])

            collectionargs = dict(namelist_fpath=pathcfg[dkey]['namelist'],
                                  ditribution_fdir=pathcfg[dkey]['distribution'],
                                  collmatches_fpath=collmfpath,
                                  dkey=dkey)

            extr = Extractor(expcfg, **collectionargs)

            featname = expcfg.get('DEFAULT', 'features', fallback=expname)
            outfile = "{0:s}{1:s}.{2:s}".format(pathcfg[dkey]["feature"],
                                                dkey, featname)  # NPZ output file

            extr.extract(pathcfg[dkey]["rank"], lblfpath, outfile, matlab_engine=matlab_engine)

    matlab_engine.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset",
                        help="dataset to run labeling on. If 'all', runs on all datasets and descriptors.",
                        type=str,
                        choices=list(ranking_type_map.keys()) + ["all"])

    parser.add_argument("rktpnum",
                        help="Number linked to the ranking approach used for the dataset.",
                        type=int)

    parser.add_argument("expconfig", help="Path to experiment .cfg file",
                        type=str)

    args = parser.parse_args()

    dataset = args.dataset
    rktpnum = args.rktpnum
    expconfig = args.expconfig

    if args.dataset == "all":
        dataset_choices = ranking_type_map
    else:
        if args.rktpnum in ranking_type_map[args.dataset]:
            dataset_choices = dict()
            dataset_choices[args.dataset] = [args.rktpnum]
        elif args.rktpnum == -1:
            dataset_choices = dict()
            dataset_choices[args.dataset] = list(ranking_type_map[args.dataset])
        else:
            print("Unavailable raking-type number {0:d} for dataset {1:s}.".format(args.descnum, args.dataset))
            print("Choices are: ", ranking_type_map[args.dataset], "   Exiting\n---")

    run_extraction(dataset_choices, expconfig)
