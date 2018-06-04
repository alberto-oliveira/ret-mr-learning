#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import shutil
import argparse
import glob

sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source")
from rankutils.cfgloader import *
from rankutils.mappings import descriptor_map

def clear_folders(dataset_choices, rootname):

    pathcfg = cfgloader("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source/path.cfg")

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:
            dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

            rootdir = pathcfg[rootname][dkey]
            rootlist = glob.glob(rootdir + "*")

            for r in rootlist:
                print("Removing: ", r)
                shutil.rmtree(r)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run labeling on. If 'all', runs on all datasets and descriptors.",
                        type=str,
                        choices=list(descriptor_map.keys())+["all"])

    parser.add_argument("descnum", help="descriptor number. If the descriptor number does not exist for the dataset,"
                        "exits with error. If all is chosen for dataset, ignores and runs for all descriptors.",
                        type=int)

    parser.add_argument("rootname", help="Name of the root folder to be cleared",
                        type=str,
                        choices=['rank', 'feature', 'label', 'output', 'result'])

    args = parser.parse_args()

    dataset = args.dataset
    descnum = args.descnum
    rootname = args.rootname

    if dataset == "all":
        dataset_choices = descriptor_map
    else:
        if descnum in descriptor_map[dataset]:
            dataset_choices = dict()
            dataset_choices[dataset] = [descnum]
        else:
            print("Unavailable descriptor number {0:d} for dataset {1:s}.".format(descnum, dataset))
            print("Choices are: ", descriptor_map[dataset], "   Exiting\n---")
            sys.exit(2)

    clear_folders(dataset_choices, rootname)
