# /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import glob
import time

import numpy as np

from rankutils.features import *
from rankutils.rIO import *
from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, getbasename, get_index
from rankutils.mappings import descriptor_map
from rankutils.statistical import ev_density_approximation

import matlab
import matlab.engine
matlab_engine = matlab.engine.start_matlab()

extractor_map = dict(dct=rank_features_kDCT,
                     dct_shift=rank_features_shiftDCT,
                     deltaik=rank_features_deltaik,
                     deltaik_c=rank_features_circ_deltaik,
                     cluster_diff=rank_features_cluster_diff,
                     emd=rank_features_density_distance)


def create_extraction_list(cfg):
    extraction_list = []
    for sect in cfg.sections():
        if sect.startswith('feat'):

            featname = cfg.get(sect, 'name')

            params = dict()

            for key in cfg[sect]:

                if key not in cfg.defaults() and key != 'name':
                    try:
                        params[key] = cfg.getboolean(sect, key, fallback=False)
                    except ValueError:
                        try:
                            params[key] = cfg.getint(sect, key, fallback=-1)
                        except ValueError:
                            params[key] = cfg.get(sect, key, fallback='')

            extraction_list.append((featname, params))

    return extraction_list


def process_rank_features(rank, extraction_list, current_pos, **distparams):
    """

    Given a rank, extracts the desired features from it, returning a feature vector.

    :param rank: r-sized structured numpy array with two fields (<name>, <score>)
    :param extraction_list: dictionary of key:tuple, where key is the name of the feature, and tuple is the parameters
                     of the feature.
    :param current_pos: position of the rank that is currently being processed.
    :return: n-sized numpy array of features.
    """

    features = []

    for featn, params in extraction_list:

        extractor = extractor_map.get(featn)

        finalparams = dict(params)
        if featn == 'emd':
            finalparams.update(**distparams)

        if 'i' in params and params['i'] == -1:
            finalparams['i'] = current_pos

        if featn == 'emd':
            feat = extractor(**finalparams)
        else:
            feat = extractor(rank['score'], **finalparams)
        features.append(feat)

    return features


def run_extraction(dataset_choices, expconfig, interval):

    np.set_printoptions(precision=8, linewidth=500, suppress=True)

    pathcfg = cfgloader("path_2.cfg")
    expcfg = cfgloader(expconfig)
    extraction_list = create_extraction_list(expcfg)

    s, e = interval

    k = expcfg.getint('DEFAULT', 'topk')
    c_num = expcfg.getint('DEFAULT', 'c_number', fallback=-1)
    distrib = expcfg.get('DEFAULT', 'distribution', fallback='')
    bins = expcfg.getint('DEFAULT', 'density_bins', fallback=1000)
    tail_idx = expcfg.getint('DEFAULT', 'tail_idx', fallback=k)

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:

            print(". Running Extraction on", dataset, " -- descriptor", descnum)
            print(". Experiment: ", expcfg['DEFAULT']['expname'])

            dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

            rkdir = pathcfg["rank"][dkey]
            fvdir = pathcfg["feature"][dkey]

            safe_create_dir(fvdir)

            namelist = np.loadtxt(pathcfg['namelists'][dkey], usecols=0, dtype='U100')
            aux = glob.glob(pathcfg['distribution'][dkey] + "*{0:s}*".format(distrib.lower()))[0]
            dparams = np.load(aux)

            rkflist = glob.glob(rkdir + "*.rk")
            rkflist.sort()

            outfeatures_list = [[] for r in range(0, k)]
            outfile = "{0:s}{1:s}.{2:s}".format(fvdir, dkey, expcfg['DEFAULT']['expname'])  # NPZ output file

            if s >= e or s < 0:
                s = 0

            if e <= 0:
                e = len(rkflist)

            #c = 1
            #acc_t = 0
            for rkfpath in rkflist[s:e]:  # Iterates over rank files inside fold
                #ts = time.perf_counter()

                print("    |_", getbasename(rkfpath))

                rk_array = read_rank(rkfpath)

                ### Clustering ###
                # If there is a c_number parameter, then there are clustering based features
                # Cluster once per rank
                if c_num > 0:
                    centers = np.sort(jenks_breaks(rk_array['score'][k:], c_num - 1))

                    # Updates the parameters of clustering based features to include cluster
                    # centers
                    for featn, params in extraction_list:
                        if featn == 'cluster_diff':
                            params['c'] = c_num
                            params['centers'] = centers

                ##################

                ### Approximating Densities ###
                # If the distribution parameter is non-empty, then some feature will use approximated densities.
                # Pre-calculate the densities according to the density_bins parameter, and the interval from the
                # processed rank

                dmat = np.zeros((k, k), dtype=np.float64) - 1
                densities = []
                edges = []
                if distrib:

                    lower_bound = rk_array['score'][tail_idx]
                    upper_bound = rk_array['score'][-1]

                    pidx = get_index(namelist, rk_array['name'][0:k])
                    topparams = dparams[pidx, :].astype(np.float)

                    for param in topparams:

                        d = dict(name=distrib,
                                 shape=float(param[0]),
                                 scale=float(param[1]),
                                 loc=float(param[2]))
                        dens, edg = ev_density_approximation(d, lower_bound, upper_bound, bins,
                                                             input_engine=matlab_engine)
                        densities.append(dens)
                        edges.append(edg)



                ### Extraction ###
                # Per top-k extraction
                for r in range(0, k):  # Iterates over rank positions
                    features = process_rank_features(rk_array, extraction_list, current_pos=r, densities=densities,
                                                     edges=edges, distmat=dmat)
                    features = np.hstack(features)

                    #print("            {pos:02d}: ".format(pos=r), features)
                    outfeatures_list[r].append(features)

                #acc_t += time.perf_counter() - ts
                #print("            -> Avg. Time: {0:0.3f}".format(acc_t/c))
                #c += 1

                ##################

            for r in range(0, k):
                outfeatures_list[r] = np.vstack(outfeatures_list[r])

            np.savez_compressed(outfile, *outfeatures_list)

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

    parser.add_argument('--start', '-s', help='Optional starting index to process a subset of ranks. Default is 0',
                        type=int, default=0)

    parser.add_argument('--end', '-e', help='Optional ending index to process a subset of ranks. Default is -1, which '
                                            'processess until the end', type=int, default=-1)

    args = parser.parse_args()

    dataset = args.dataset
    descnum = args.descnum
    expconfig = args.expconfig
    s = args.start
    e = args.end

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

    run_extraction(dataset_choices, expconfig, (s, e))
