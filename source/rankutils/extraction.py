#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import numpy as np
import numpy.random as npran
import glob
import warnings

import ipdb as pdb
from tqdm import tqdm

from rankutils.utilities import getbasename, get_index
from rankutils.features import get_rank_feature
from rankutils.rIO import read_rank
from rankutils.statistical import ev_density_approximation, ev_fit
from rankutils.clustering import clustering_1d


def generate_rank_variations(rank, labels, nvar, vfactor):

    if vfactor < 0.05 or vfactor > 1.0:
        raise ValueError("vfactor should be between 0.3 and 1.0")

    variations = [(rank, labels)]

    for i in range(nvar):

        idx = np.arange(rank.size)
        npran.shuffle(idx)

        ci = int(np.floor(vfactor * rank.size))

        vidx = idx[0:ci]
        vidx.sort()

        variations.append((rank[vidx], labels[vidx]))

    return variations



class Extractor:

    def __init__(self, cfg, namefpath='', distfdir=''):

        # GT labels. Saved on a .npz file together with the feature vectors

        self.expname = cfg.get('DEFAULT', 'expname')
        self.__topk = cfg.getint('DEFAULT', 'topk', fallback=10)
        self.__distribtype = cfg.get('parameters', 'distribution', fallback='GEV')
        self.__bins_approx = cfg.getint('parameters', 'bins_approximation', fallback=256)
        self.__tail_idx = cfg.getint('parameters', 'tail_idx', fallback=self.__topk)
        self.__cluster_num = cfg.getint('parameters', 'c_num', fallback=64)
        self.__clustering = cfg.get('parameters', 'clustering', fallback='fixed')
        self.__dct_type = cfg.getint('parameters', 'dct_type', fallback=2)
        self.__dct_range = cfg.getint('parameters', 'dct_range', fallback=20)
        self.__delta_range = cfg.getint('parameters', 'delta_range', fallback=20)
        self.__abs_diff = cfg.getboolean('parameters', 'absolute_difference', fallback=False)
        self.__norm = cfg.getboolean('parameters', 'normalize', fallback=False)
        self.__nvar = cfg.getint('parameters', 'nvar', fallback=1)
        self.__varf = cfg.getfloat('parameters', 'varf', fallback=0.5)

        self.__namelist = None
        self.__distfitparams = None

        if namefpath != '':
            try:
                self.__namelist = np.loadtxt(namefpath, usecols=0, dtype='U100')
            except OSError:
                warnings.warn("Namelist file not found. Trying to run descriptors that use it will result in a crash", RuntimeWarning)

        if distfdir != '':
            try:
                aux = glob.glob(distfdir + "*{0:s}*".format(self.__distribtype.lower()))[0]
                self.__distfitparams = np.load(aux)
            except IndexError:
                warnings.warn("Distribution parameter file not found. Trying to run density-dependant extractors will result in a crash", RuntimeWarning)

        self.__cluster_check = False
        self.__density_check = False
        self.__fit_check = False

        self.__feature_queue = []
        self.__fv_dim = 0

        for key in cfg['features']:

            if key not in cfg.defaults():
                featalias = cfg.get('features', key)

                #print("-> ", featalias)
                self.__feature_queue.append(featalias)

                if featalias == 'cluster_diff' and not self.__cluster_check:
                    self.__cluster_check = True
                    self.__fv_dim += self.__cluster_num

                if (featalias == 'emd' or featalias == 'query_bhatt') and not self.__density_check:
                    self.__density_check = True
                    self.__fv_dim += self.__bins_approx

                if featalias == 'query_bhatt' and not self.__fit_check:
                    self.__fit_check = True
                    self.__fv_dim += self.__bins_approx

                if featalias == 'deltaik_c':
                    self.__fv_dim += self.__delta_range

                if featalias == 'dct_shift':
                    self.__fv_dim += self.__dct_range


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
        try:
            aux = glob.glob(distfdir + "*{0:s}*".format(self.__distribtype.lower()))[0]
            self.__distfitparams = np.load(aux)
        except IndexError:
            warnings.warn("Distribution parameter file not found. Trying to run density-dependant extractors"
                          "will result in a crash", RuntimeWarning)
        return

    def extract(self, inputdir, labelfpath, outfile, matlab_engine=None):

        labels = np.load(labelfpath)


        #pdb.set_trace()
        rkflist = glob.glob(inputdir + "*.rk")
        rkflist.sort()

        assert len(rkflist) == labels.shape[0], "Inconsistent number of ranks and label rows!"

        gfvargs = dict(scores=None,
                       i=-1,
                       topk=self.__topk,
                       dct_type=self.__dct_type,
                       dct_range=self.__dct_range,
                       delta_range=self.__delta_range,
                       cluster_num=self.__cluster_num,
                       abs=self.__abs_diff,
                       norm=self.__norm,
                       clusters=None,
                       densities=None,
                       edges=None,
                       distmat=None,
                       q_density=None,
                       q_edges=None
                       )

        total_samples = len(rkflist)

        pos_vectors = [[] for _ in range(0, self.__topk)]

        # Last dimension is just to keep the labels and features dimensions the same
        outlabels = np.zeros((self.__topk, total_samples, self.__nvar, 1), dtype=np.uint8)
        outfeatures = np.zeros((self.__topk, total_samples, self.__nvar, self.__fv_dim), dtype=np.float64)

        for i in tqdm(range(total_samples), ncols=75, desc='Rank File', total=total_samples):

            #print("    |_", getbasename(rkfpath), end='')

            # We are only interested in labels for the topk
            rklabels = labels[i]

            rk_array = read_rank(rkflist[i])

            # Experimental. Generate random variations of ranked scores + labels by removing a random sample of
            # elements from the rank.
            variations = generate_rank_variations(rk_array['score'], rklabels, self.__nvar, self.__varf)

            sample_vectors = [[] for _ in range(0, self.__nvar)]
            for v in tqdm(range(self.__nvar), ncols=75, desc='  .Variation', total=self.__nvar):

                rkscores, rklabels = variations[v]

                gfvargs['scores'] = rkscores

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
                    fvector = []
                    for featalias in self.__feature_queue:
                        fvector.append(get_rank_feature(featalias, **gfvargs))
                    fvector = np.hstack(fvector).reshape(1, -1)

                    assert fvector.size == self.__fv_dim, "Inconsistent feature vector size <{0:03d}> with " \
                                                              "precomputed feature vector dimension <{1:03d}> ".format(fvector.size, self.__fv_dim)

                    outlabels[r, i, v, 0] = rklabels[r]
                    outfeatures[r, i, v] = fvector


        #pdb.set_trace()
        np.savez_compressed(outfile, features=outfeatures, labels=outlabels)

        return







