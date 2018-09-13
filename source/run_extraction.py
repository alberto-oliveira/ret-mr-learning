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
from rankutils.mappings import descriptor_map
from rankutils.statistical import ev_density_approximation

import matlab
import matlab.engine
matlab_engine = matlab.engine.start_matlab()


def run_extraction(dataset_choices, expconfig):

    np.set_printoptions(precision=3, linewidth=500, suppress=True)

    pathcfg = cfgloader("path_2.cfg")
    expcfg = cfgloader(expconfig)

    extr = None

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:

            print(". Running Extraction on", dataset, " -- descriptor", descnum)
            print(". Experiment: ", expcfg['DEFAULT']['expname'])

            dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

            rkdir = pathcfg["rank"][dkey]
            fvdir = pathcfg["feature"][dkey]

            safe_create_dir(fvdir)

            if not extr:
                extr = Extractor(expcfg, pathcfg['namelists'][dkey], pathcfg['distribution'][dkey])
                #print("  << {0:s} >>".format(pathcfg['distribution'][dkey]))
                #print("  << Creating extractor >>")
            else:
                extr.update_namelist(pathcfg['namelists'][dkey])
                #print("  << {0:s} >>".format(pathcfg['distribution'][dkey]))
                extr.update_fit_params(pathcfg['distribution'][dkey])
                #print("  << Updating extractor >>")
                pass

            outfile = "{0:s}{1:s}.{2:s}".format(fvdir, dkey, expcfg['DEFAULT']['expname'])  # NPZ output file

            extr.extract(rkdir, outfile, matlab_engine=matlab_engine)

    matlab_engine.quit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset",
                        help="dataset to run labeling on. If 'all', runs on all datasets and descriptors.",
                        type=str,
                        choices=list(descriptor_map.keys()) + ["all"])

    parser.add_argument("descnum",
                        help="descriptor number. If the descriptor number does not exist for the dataset,"
                             "exits with error. If all is chosen for dataset, ignores and runs for all descriptors.",
                        type=int)

    parser.add_argument("expconfig", help="path to experiment .cfg file",
                        type=str)

    args = parser.parse_args()

    dataset = args.dataset
    descnum = args.descnum
    expconfig = args.expconfig

    if dataset == "all":
        dataset_choices = descriptor_map
    else:
        if descnum in descriptor_map[dataset]:
            dataset_choices = dict()
            dataset_choices[dataset] = [descnum]
        else:
            print("Unavailable descriptor number {0:d} for dataset {1:s}.".format(descnum, dataset))
            print("Choise are: ", descriptor_map[dataset], "   Exiting\n---")
            sys.exit(2)

    run_extraction(dataset_choices, expconfig)
