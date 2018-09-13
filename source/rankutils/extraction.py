#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import numpy as np
import glob

import ipdb as pdb

from rankutils.utilities import getbasename, get_index
from rankutils.features import get_rank_feature
from rankutils.rIO import read_rank
from rankutils.statistical import ev_density_approximation, ev_fit
from rankutils.clustering import clustering_1d


class Extractor:

    def __init__(self, cfg, namefpath='', distfdir=''):

        self.expname = cfg.get('DEFAULT', 'expname')
        self.__topk = cfg.getint('DEFAULT', 'topk', fallback=10)
        self.__distribtype = cfg.get('parameters', 'distribution', fallback='GEV')
        self.__bins_approx = cfg.getint('parameters', 'bins_approximation', fallback=256)
        self.__tail_idx = cfg.getint('parameters', 'tail_idx', fallback=self.__topk)
        self.__cluster_num = cfg.getint('parameters', 'c_num', fallback=64)
        self.__clustering = cfg.get('parameters', 'clustering', fallback='jenks')
        self.__dct_type = cfg.getint('parameters', 'dct_type', fallback=2)
        self.__dct_range = cfg.getint('parameters', 'dct_range', fallback=20)
        self.__delta_range = cfg.getint('parameters', 'delta_range', fallback=20)

        self.__namelist = None
        self.__distfitparams = None

        if namefpath != '':
            self.__namelist = np.loadtxt(namefpath, usecols=0, dtype='U100')

        if distfdir != '':
            aux = glob.glob(distfdir + "*{0:s}*".format(self.__distribtype.lower()))[0]
            self.__distfitparams = np.load(aux)

        self.__cluster_check = False
        self.__density_check = False
        self.__fit_check = False

        self.__feature_queue = []

        for key in cfg['features']:

            if key not in cfg.defaults():
                featalias = cfg.get('features', key)

                print("-> ", featalias)
                self.__feature_queue.append(featalias)

                if featalias == 'cluster_diff' and not self.__cluster_check:
                    self.__cluster_check = True

                if (featalias == 'emd' or featalias == 'query_bhatt') and not self.__density_check:
                    self.__density_check = True

                if featalias == 'query_bhatt' and not self.__fit_check:
                    self.__fit_check = True

        if self.__density_check and self.__namelist is None:
            raise ValueError("Statistical (<emd>, <query_bhatt>) features requires a valid namelist. "
                             "Did you pass a valid namefpath argument?")

        if self.__density_check and self.__distfitparams is None:
            raise ValueError("Statistical (<emd>, <query_bhatt>) features requires a valid fit parameters array. "
                             "Did you pass a valid distfdir argument?")

    def update_namelist(self, namefpath):
        self.__namelist = np.loadtxt(namefpath, usecols=0, dtype='U100')
        return

    def update_fit_params(self, distfdir):
        aux = glob.glob(distfdir + "*{0:s}*".format(self.__distribtype.lower()))[0]
        self.__distfitparams = np.load(aux)
        return

    def extract(self, inputdir, outfile, matlab_engine=None):

        outfeatures_list = [[] for _ in range(0, self.__topk)]

        #pdb.set_trace()
        rkflist = glob.glob(inputdir + "*.rk")
        rkflist.sort()

        gfvargs = dict(scores=None,
                       i=-1,
                       topk=self.__topk,
                       dct_type=self.__dct_type,
                       dct_range=self.__dct_range,
                       delta_range=self.__delta_range,
                       cluster_num=self.__cluster_num,
                       clusters=None,
                       densities=None,
                       edges=None,
                       distmat=None,
                       q_density=None,
                       q_edges=None
                       )

        for rkfpath in rkflist[0:]:

                print("    |_", getbasename(rkfpath))

                rk_array = read_rank(rkfpath)
                gfvargs['scores'] = rk_array['score']

                ### Clustering ###
                if self.__cluster_check:
                    centers = np.sort(clustering_1d(self.__clustering, **dict(data=rk_array['score'][self.__topk:],
                                                                              c_num=self.__cluster_num)))

                    centers = np.sort(centers)[::-1]
                    gfvargs['centers'] = centers

                ##################

                ### Approximating Densities ###
                # If the distribution parameter is non-empty, then some feature will use approximated densities.
                # Pre-calculate the densities according to the density_bins parameter, and the interval from the
                # processed rank

                if self.__density_check:

                    gfvargs['distmat'] = np.zeros((self.__topk, self.__topk), dtype=np.float64) - 1
                    gfvargs['densities'] = []
                    gfvargs['edges'] = []

                    lower_bound = rk_array['score'][self.__tail_idx]
                    upper_bound = rk_array['score'][-1]

                    pidx = get_index(self.__namelist, rk_array['name'][0:self.__topk])

                    if pidx.size < self.__topk:
                        raise ValueError("Could not find top {0:d} ranked images in namelist. ".format(self.__topk))

                    topparams = self.__distfitparams[pidx, :].astype(np.float)

                    #print("       -> ", rk_array['name'][0], " : ", topparams[0])

                    for param in topparams:

                        d = dict(name=self.__distribtype,
                                 shape=float(param[0]),
                                 scale=float(param[1]),
                                 loc=float(param[2]))
                        dens, edg = ev_density_approximation(d, lower_bound, upper_bound, self.__bins_approx,
                                                             input_engine=matlab_engine)
                        gfvargs['densities'].append(dens)
                        gfvargs['edges'].append(edg)
                ##################

                if self.__fit_check:

                    lower_bound = rk_array['score'][self.__tail_idx]
                    upper_bound = rk_array['score'][-1]

                    q_tail = rk_array['score'][self.__tail_idx:]
                    qdist = ev_fit(q_tail, self.__distribtype, matlab_engine)

                    dens, edg = ev_density_approximation(qdist, lower_bound, upper_bound, self.__bins_approx,
                                                         input_engine=matlab_engine)
                    gfvargs['q_density'] = dens
                    gfvargs['q_edges'] = edg
                ##################

                ### Extraction ###
                # Per top-k extraction
                for r in range(0, self.__topk):  # Iterates over rank positions

                    gfvargs['i'] = r
                    features = []
                    for featalias in self.__feature_queue:
                        features.append(get_rank_feature(featalias, **gfvargs))
                    features = np.hstack(features)

                    outfeatures_list[r].append(features)

        for r in range(0, self.__topk):
            outfeatures_list[r] = np.vstack(outfeatures_list[r])

        np.savez_compressed(outfile, *outfeatures_list)

        return







