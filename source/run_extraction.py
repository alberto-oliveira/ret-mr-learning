#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import glob

import numpy as np

from common.extraction import *
from common.rIO import *
from common.cfgloader import *
from common.utilities import safe_create_dir, getbasename
from common.mappings import descriptor_map

def run_extraction(dataset_choices, expconfig, fn, metric='score'):

    pathcfg = cfgloader("path_2.cfg")
    expcfg = cfgloader(expconfig)
    extractor_list = create_extraction_list(expcfg)

    k = expcfg['DEFAULT'].getint('topk')

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:

            print(". Running Extraction on", dataset, " -- descriptor", descnum)
            print(". Experiment: ", expcfg['DEFAULT']['expname'])

            dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

            rkdir = pathcfg["rank"][dkey]
            fvdir = pathcfg["feature"][dkey]

            safe_create_dir(fvdir)

            rkflist = glob.glob(rkdir + "*.rk")
            rkflist.sort()

            outarray_map = dict()

            feat_posit = [[] for _ in range(k)]
            outfile = "{0:s}{1:s}.{2:s}".format(fvdir, dkey, expcfg['DEFAULT']['expname'])  # NPZ output file

            feat_rank = []

            for rkfpath in rkflist:  # Iterates over rank files inside fold
                print("    |_", getbasename(rkfpath))

                aux = dict(positional=[], rank=[])

                rank = read_rank(rkfpath)
                for r in range(0, k):  # Iterates over rank positions
                    features = extract_rank_features(rank[metric], extractor_list, ci=r)

                    # Some features are per-rank, others are positional (that is, for each top-k positions)
                    # Although I recalculate per-rank feat. for each position, when concatenating for the top-k
                    # features I need to have them only once
                    for pos, tup in enumerate(extractor_list):
                        if tup[1] == 'rank' and not aux['rank']:
                            aux['rank'].append(features[pos])
                        elif tup[1] == 'pos':
                            aux['positional'].append(features[pos])

                    new_feat_posit = np.hstack(features)
                    if feat_posit[r]:
                        assert feat_posit[r][-1].shape == new_feat_posit.shape, \
                               "Inconsistent shapes between positional features for rank {0:d}. Previous" \
                               " was {1:s} and latest {2:s}"\
                               .format(r, str(feat_posit[r][-1].shape), str(new_feat_posit.shape))

                    feat_posit[r].append(new_feat_posit)

                if aux['positional']:
                    aux['positional'] = np.hstack(aux['positional'])
                if aux['rank']:
                    aux['rank'] = np.hstack(aux['rank'])

                feat_rank.append(np.hstack([aux['positional'], aux['rank']]).reshape(1, -1))

            for r in range(k):  # Iterates over rank position
                outarray_map['{0:d}'.format(r+1)] = np.vstack(feat_posit[r])

            outarray_map['r'] = np.vstack(feat_rank)

            np.savez_compressed(outfile, **outarray_map)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run labeling on. If 'all', runs on all datasets and descriptors.",
                        type=str,
                        choices=list(descriptor_map.keys())+["all"])

    parser.add_argument("descnum", help="descriptor number. If the descriptor number does not exist for the dataset,"
                        "exits with error. If all is chosen for dataset, ignores and runs for all descriptors.",
                        type=int)

    parser.add_argument("expconfig", help="path to experiment .cfg file",
                        type=str)

    parser.add_argument("--foldnum", "-f", help="Number of folds used. Default is 10.", type=int, default=10)

    args = parser.parse_args()

    dataset = args.dataset
    descnum = args.descnum
    expconfig = args.expconfig
    fn = args.foldnum

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

    run_extraction(dataset_choices, expconfig, fn)
