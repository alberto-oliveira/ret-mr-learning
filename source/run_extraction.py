#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import glob

import numpy as np

from rankutils.extraction import *
from rankutils.rIO import *
from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, getbasename
from rankutils.mappings import descriptor_map

def run_extraction(dataset_choices, expconfig):

    pathcfg = cfgloader("path_2.cfg")
    dbparams = cfgloader("dbparams.cfg")
    expcfg = cfgloader(expconfig)
    extraction_list= create_extraction_list(expcfg)

    k = expcfg.getint('DEFAULT', 'topk')
    c_num = expcfg.getint('DEFAULT', 'c_number', fallback=-1)

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:

            print(". Running Extraction on", dataset, " -- descriptor", descnum)
            print(". Experiment: ", expcfg['DEFAULT']['expname'])

            dkey = "{0:s}_desc{1:d}".format(dataset, descnum)
            print("   -> Scoretype:", dbparams[dkey]['scoretype'])

            rkdir = pathcfg["rank"][dkey]
            fvdir = pathcfg["feature"][dkey]

            safe_create_dir(fvdir)

            rkflist = glob.glob(rkdir + "*.rk")
            rkflist.sort()

            outfeatures_list = [[] for r in range(0, k)]
            outfile = "{0:s}{1:s}.{2:s}".format(fvdir, dkey, expcfg['DEFAULT']['expname'])  # NPZ output file

            for rkfpath in rkflist:  # Iterates over rank files inside fold
                print("    |_", getbasename(rkfpath))

                rank = read_rank(rkfpath, dbparams[dkey]['scoretype'])


                ### Clustering ###
                # If there is a c_number parameter, then there are clustering based features
                # Cluster once per rank
                if c_num > 0:
                    centers = np.sort(jenks_breaks(rank[k:], c_num-1))

                    # Updates the parameters of clustering based features to include cluster
                    # centers
                    for featn, feattp, params in extraction_list:
                        if featn == 'cluster_diff':
                            params['c'] = c_num
                            params['centers'] = centers

                ##################

                ### Extraction ###
                # Per top-k extraction
                for r in range(0, k):  # Iterates over rank positions
                    features = extract_rank_features(rank, extraction_list, ci=r)
                    features = np.hstack(features)

                    outfeatures_list[r].append(features)

                ##################

            for r in range(0, k):
                outfeatures_list[r] = np.vstack(outfeatures_list[r])

            np.savez_compressed(outfile, *outfeatures_list)




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