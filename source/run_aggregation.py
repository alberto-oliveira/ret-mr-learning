#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse
import glob

from rankutils.mappings import ranking_type_map
from rankutils.cfgloader import cfgloader
from rankutils.utilities import get_classname, get_query_classname, safe_create_dir
from rankutils.rIO import read_rank
from rankutils.aggregation import *

from sklearn.preprocessing import normalize

from tqdm import tqdm

#import ipdb as pdb

import numpy as np


aggregation_map = dict(places365=[1, 2, 3, 4, 5],
                       vggfaces=[1, 3, 4, 5],
                       imagenet=[1, 2, 3, 4])


def make_header(**kw):

    header_array = np.array((kw['dataset'], '+'.join(kw['keys']), kw['aggmethod'], kw['topn'], kw['k'],
                             kw['lfolder'], kw['phi'], kw['balance'], kw['abst']),
                            dtype=dict(names=('dataset', 'agg_keys', 'method', 'topn', 'k',
                                              'labelfolder', 'phi', 'balance', 'absent'),
                                       formats=('U20', 'U100', 'U20', np.int32, np.int32,
                                              'U50', np.float32, np.int8, np.int8)))

    return header_array


def get_keys(dataset, k_idx_str):

    if k_idx_str:
        keys = ["{0:s}_{1:03d}".format(dataset, int(k))
                for k in k_idx_str.split(',') if int(k) in ranking_type_map[dataset]]
    else:
        keys = ["{0:s}_{1:03d}".format(dataset, int(k)) for k in aggregation_map[dataset]]

    return keys


def get_labels(labelpath, n, foldsfpath=''):

    labels = np.load(labelpath)

    assert labels.shape[1] >= n, 'Number of columns in labels array is < than the required {0:d}'.format(n)

    labels = labels[:, 0:n]

    return labels


def reevaluate(ranknames, qclass, n):

    relabels = np.zeros((1, n), dtype=np.uint8)

    for i, name in enumerate(ranknames[0:n]):
        cname = get_classname(name)

        if cname == qclass:
            relabels[0, i] = 1

    return relabels


def aggregate(method, **kwargs):

    if method == 'combSUM':
        aggr_names, aggr_scores = aggr_combSUM(kwargs['names'], kwargs['scores'], weights=None)

    elif method == 'combSUM_pre':
        aggr_names, aggr_scores = aggr_combSUM_pre(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                   f=kwargs['f'], balance=kwargs['balance'], weights=None)

    elif method == 'combSUM_avgw':
        aggr_names, aggr_scores = aggr_combSUM_post_avgw(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                         f=kwargs['f'], weights=None)

    elif method == 'combSUM_majw':
        aggr_names, aggr_scores = aggr_combSUM_post_majw(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                         f=kwargs['f'], weights=None)

    elif method == 'combMNZ':
        aggr_names, aggr_scores = aggr_combMNZ(kwargs['names'], kwargs['scores'], weights=None)

    elif method == 'combMNZ_pre':
        aggr_names, aggr_scores = aggr_combMNZ_pre(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                   f=kwargs['f'], weights=None)

    elif method == 'combMIN':
        aggr_names, aggr_scores = aggr_combMIN(kwargs['names'], kwargs['scores'])

    elif method == 'combMIN_avgw':
        aggr_names, aggr_scores = aggr_combMIN_post_avgw(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                         f=kwargs['f'])

    elif method == 'combMIN_majw':
        aggr_names, aggr_scores = aggr_combMIN_post_majw(kwargs['names'], kwargs['scores'], kwargs['labels'],
                                                         f=kwargs['f'])

    elif method == 'combMEAN':
        aggr_names, aggr_scores = aggr_combMEAN(kwargs['names'], kwargs['scores'])

    elif method == 'combMEAN_post':
        aggr_names, aggr_scores = aggr_combMEAN_plus(kwargs['names'], kwargs['scores'], kwargs['labels'], f=kwargs['f'])

    elif method == 'combMEDIAN':
        aggr_names, aggr_scores = aggr_combMEDIAN(kwargs['names'], kwargs['scores'])

    elif method == 'combMINMAX':
        aggr_names, aggr_scores = aggr_combMINMAX(kwargs['names'], kwargs['scores'])

    return aggr_names, aggr_scores



def run_aggregation(dataset, aggconfig):

    np.set_printoptions(precision=4, linewidth=300, suppress=True)

    pathcfg = cfgloader("path_2.cfg")
    aggcfg = cfgloader(aggconfig)

    # required parameters
    aggmethod = aggcfg.get('DEFAULT', 'aggmethod')
    topn = aggcfg.getint('DEFAULT', 'topn')

    # non-required parameters
    k = aggcfg.getint('DEFAULT', 'k', fallback=0)
    keys = get_keys(dataset, aggcfg.get('DEFAULT', 'to_aggregate', fallback=''))
    f = aggcfg.getfloat('DEFAULT', 'f', fallback=100)/100
    balance = aggcfg.getint('DEFAULT', 'balance', fallback=0)
    abst = aggcfg.getint('DEFAULT', 'absent', fallback=-1)

    # non-required path parameters
    rootlabelpath = aggcfg.get('DEFAULT', 'rootlabelpath', fallback='')
    labelfolder = aggcfg.get('DEFAULT', 'labelfolder', fallback='')
    fulllabelpath = "{0:s}/{1:s}/{2:s}/".format(pathcfg.get('DEFAULT', 'agglbldir'), dataset, labelfolder)

    # output parameters
    outpath = aggcfg.get('DEFAULT', 'outpath')
    res_sufix = aggcfg.get('DEFAULT', 'res_sufix')

    safe_create_dir(os.path.dirname(outpath))
    respath = "{0:s}/{1:s}.{2:s}".format(outpath, dataset, res_sufix)

    print(aggmethod)
    print("labelfolder: ", labelfolder)
    print("fulllabelpath: ", fulllabelpath)
    print("topn = {0:d}".format(topn))
    print("k = {0:d}".format(k))
    print("keys:", keys)
    print("f = {0:0.2f}".format(f))
    print("balance = {0:d}".format(balance))
    print("absent = {0:d}".format(abst))
    print('Result file name:', os.path.basename(respath))

    header_array = make_header(dataset=dataset, keys=keys, aggmethod=aggmethod, topn=topn, k=k,
                               lfolder=os.path.basename(labelfolder), phi=f, balance=balance, abst=abst)

    assert k <= topn, "k[{0:d}] should have value <= topn[{1:d}]"

    filemap = dict()
    labelmap = dict()

    nsamples = 0
    for key in keys:

        ranktpname = pathcfg.get(key, 'rktpdir')

        rankfpaths = glob.glob(pathcfg.get(key, 'rank') + "*.rk")
        rankfpaths.sort()

        if nsamples == 0:
            nsamples = len(rankfpaths)

        filemap[key] = rankfpaths

        if rootlabelpath and labelfolder:
            labelmap[key] = get_labels(glob.glob(fulllabelpath + "/*{0:s}*".format(ranktpname))[0], k, '')
        else:
            labelmap[key] = np.zeros((nsamples, topn))

    agg_labels = []

    prev_size = -1
    for i in tqdm(range(nsamples), ncols=100, desc='Sample', total=nsamples):

        kwargs = dict(f=f, balance=balance)

        names = []
        scores = []
        labels = []

        exname = ''

        for key in keys:

            if exname == '':
                exname = os.path.basename(filemap[key][i])
                qclass = get_query_classname(exname)

            rank = read_rank(filemap[key][i], rows=topn)

            names.append(rank['name'])
            scores.append(1 - normalize(rank['score'].reshape(1, -1)).reshape(-1))
            labels.append(labelmap[key][i, 0:k])

        tbl_names, tbl_scores, tbl_labels = create_aggr_table(names, scores, labels, absent=abst)

        kwargs.update(dict(names=tbl_names, scores=tbl_scores, labels=tbl_labels))

        aggr_names, aggr_scores = aggregate(aggmethod, **kwargs)

        agg_labels.append(reevaluate(aggr_names, qclass, topn))


    np.savez(respath, header=header_array, labels=np.vstack(agg_labels))

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(ranking_type_map.keys()))

    parser.add_argument("aggconfig", help="path to aggregation config file.")
    parser.add_argument("-c", "--num_cores", help="cores to run parallel aggregations", type=int, default=1)

    args = parser.parse_args()

    if os.path.isfile(args.aggconfig):
        run_aggregation(args.dataset, args.aggconfig)
    elif os.path.isdir(args.aggconfig):

        cfgfiles = glob.glob(args.aggconfig + "/*.cfg")
        cfgfiles.sort()

        if args.num_cores == 1:
            for cfgf in cfgfiles:
                run_aggregation(args.dataset, cfgf)

        else:
            from joblib import Parallel, delayed
            from itertools import product

            inputs = [v for v in product([args.dataset], cfgfiles)]
            print(inputs)
            Parallel(n_jobs=args.num_cores)(delayed(run_aggregation)(*i) for i in inputs)

