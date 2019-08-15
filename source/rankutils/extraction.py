#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import numpy as np
import numpy.random as npran
import glob
import warnings

from tqdm import tqdm

from rankutils.utilities import getbasename, get_index
from rankutils.features import get_rank_feature
from rankutils.rIO import read_rank
from rankutils.statistical import ev_density_approximation, ev_fit, ktau_matrix
from rankutils.clustering import clustering_1d


def load_namelist(fpath):

    dt = dict(names=('name', 'numfeat', 'cid'), formats=('U100', np.int32, np.int32))
    namelist = np.loadtxt(fpath, dtype=dt)

    return namelist


def generate_rank_variations(rank, labels, nvar, vfactor):

    if vfactor < 0.05 or vfactor > 1.0:
        raise ValueError("vfactor should be between 0.3 and 1.0")

    variations = [(rank['name'], rank['score'], labels)]

    for i in range(nvar-1):

        idx = np.arange(rank.size)
        npran.shuffle(idx)

        ci = int(np.floor(vfactor * rank.size))

        vidx = idx[0:ci]
        vidx.sort()

        variations.append((rank['name'][vidx], rank['score'][vidx], labels[vidx]))

    return variations


def get_name_in_coll(rkfpath, key):

    if key == 'vggfaces_002':
        aux = os.path.basename(rkfpath).split('_', 1)[1]
        aux = aux.rsplit('_', 1)[0]
        aux += '.png'
    else:
        aux = os.path.basename(rkfpath).split('_', 1)[1]
        aux = aux.rsplit('.', 1)[0]

    return aux



class Extractor:

    def __init__(self, cfg, **collecionargs):

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
        self.__num_intv = cfg.getint('parameters', 'num_intervals', fallback=100)
        self.__contextual_k = cfg.getint('parameters', 'contextual_k', fallback=300)
        self.__norm = cfg.getboolean('parameters', 'normalize', fallback=False)
        self.__nvar = cfg.getint('parameters', 'nvar', fallback=1)
        self.__varf = cfg.getfloat('parameters', 'varf', fallback=0.5)
        self.__get_previous = cfg.getboolean('parameters', 'get_previous', fallback=False)
        self.__get_next = cfg.getboolean('parameters', 'get_previous', fallback=False)

        self.__namelist = None
        self.__distfitparams = None
        self.__collmatches = None
        self.__collscores = None

        self.__dkey = collecionargs['dkey']

        if 'namelist_fpath' in collecionargs:
            try:
                self.__namelist = load_namelist(collecionargs['namelist_fpath'])
            except OSError:
                warnings.warn("Namelist file not found. Trying to run descriptors that use it will"
                              " result in a crash", RuntimeWarning)

        if 'ditribution_fdir' in collecionargs:
            try:
                aux = glob.glob(collecionargs['ditribution_fdir'] + "*{0:s}*".format(self.__distribtype.lower()))[0]
                self.__distfitparams = np.load(aux)
            except IndexError:
                warnings.warn("Distribution parameter file not found. Trying to run density-dependant extractors will"
                              " result in a crash", RuntimeWarning)

        self.__cluster_check = False
        self.__density_check = False
        self.__fit_check = False
        self.__contextual_check = False
        self.__correlation_check = False
        self.__feature_queue = []
        self.__fv_dim = 0

        for key in cfg['features']:

            if key not in cfg.defaults():
                featalias = cfg.get('features', key)

                #print("-> ", featalias)
                self.__feature_queue.append(featalias)

                if featalias == 'raw_scores':
                    self.__fv_dim += 1

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

                if featalias == 'rank_jacc':
                    self.__contextual_check = True
                    self.__fv_dim += self.__num_intv

                if featalias == 'accum_jacc':
                    self.__contextual_check = True
                    self.__fv_dim += self.__num_intv

                if featalias == 'cid_jacc':
                    self.__contextual_check = True
                    self.__fv_dim += self.__num_intv

                if featalias == 'cid_freq_diff':
                    self.__contextual_check = True
                    # When counting the frequencies of CID, the size of the feature vector is the number of different
                    # CIDs. Because CIDs are indexed from 0, that number is the maximum CID value + 1
                    self.__fv_dim += np.max(self.__namelist['cid']) + 1

                if featalias == 'ktau_top':
                    self.__contextual_check = True
                    self.__correlation_check = True
                    self.__fv_dim += self.__topk

                if featalias == 'seq_bhatt':
                    self.__contextual_check = True
                    self.__fv_dim += self.__bins_approx

        if 'collmatches_fpath' in collecionargs and self.__contextual_check:
            try:
                self.__collmatches = np.load(collecionargs['collmatches_fpath'])[:, 0:self.__contextual_k]
            except OSError:
                warnings.warn("Collection matching indices file not found. Trying to run descriptors that use it will"
                              " result in a crash", RuntimeWarning)

        if 'collscores_fpath' in collecionargs and self.__contextual_check:
            try:
                self.__collscores = np.load(collecionargs['collscores_fpath'])[:, 0:self.__contextual_k]
            except OSError:
                warnings.warn("Collection matching indices file not found. Trying to run descriptors that use it will"
                              " result in a crash", RuntimeWarning)

        if self.__contextual_check and self.__namelist is None:
            raise ValueError("Contextual features requires a valid namelist. "
                             "Did you pass a valid namelist file path argument?")

        if self.__contextual_check and self.__collmatches is None:
            raise ValueError("Contextual features requires a valid collection matching indices file. "
                             "Did you pass a valid collection matches file path argument?")

        if self.__density_check and self.__namelist is None:
            raise ValueError("Statistical features requires a valid namelist. "
                             "Did you pass a valid namelist file path argument?")

        if self.__density_check and self.__distfitparams is None:
            raise ValueError("Statistical features requires a valid fit parameters array. "
                             "Did you pass a valid distribution file dir argument?")

    def update_namelist(self, namefpath):
        try:
            self.__namelist = load_namelist(namefpath)
        except OSError:
            warnings.warn("Namelist file not found.", RuntimeWarning)
        return

    def update_fit_params(self, distfdir):
        try:
            aux = glob.glob(distfdir + "*{0:s}*".format(self.__distribtype.lower()))[0]
            self.__distfitparams = np.load(aux)
        except IndexError:
            warnings.warn("Distribution parameter file not found. Trying to run density-dependant extractors"
                          "will result in a crash", RuntimeWarning)
        return

    def update_collection_matches(self, collmfpath):

        if self.__contextual_check:
            try:
                self.__collmatches = np.load(collmfpath)
            except OSError:
                warnings.warn("Collection matches file not found. Trying to run density-dependant extractors"
                              "will result in a crash", RuntimeWarning)

        return

    def update_collection_matches(self, collsfpath):

        if self.__contextual_check:
            try:
                self.__collscores = np.load(collsfpath)
            except OSError:
                warnings.warn("Collection scores file not found. Trying to run density-dependant extractors"
                              "will result in a crash", RuntimeWarning)

        return

    def extract(self, inputdir, labelfpath, outfile, matlab_engine=None):

        labels = np.load(labelfpath)

        rkflist = glob.glob(inputdir + "*.rk")
        rkflist.sort()

        assert len(rkflist) == labels.shape[0], "Inconsistent number of ranks <{0:d}> and label rows <{1:d}>!"\
                                                .format(len(rkflist), labels.shape[0])

        gfvargs = dict(scores=None,
                       i=-1,
                       topk=self.__topk,
                       dct_type=self.__dct_type,
                       dct_range=self.__dct_range,
                       delta_range=self.__delta_range,
                       cluster_num=self.__cluster_num,
                       abs=self.__abs_diff,
                       num_intv=self.__num_intv,
                       norm=self.__norm,
                       clusters=None,
                       densities=None,
                       edges=None,
                       distmat=None,
                       q_density=None,
                       q_edges=None,
                       bins=self.__bins_approx,
                       coll_matches=self.__collmatches,
                       coll_scores=self.__collscores)

        total_samples = len(rkflist)

        pos_vectors = [[] for _ in range(0, self.__topk)]

        # Last dimension is just to keep the labels and features dimensions the same
        outlabels = np.zeros((self.__topk, total_samples, self.__nvar, 1), dtype=np.uint8)

        t = 1
        if self.__get_next:
            t += 1
        if self.__get_previous:
            t += 1
        outfeatures = np.zeros((self.__topk, total_samples, self.__nvar, self.__fv_dim * t), dtype=np.float64)

        for i in tqdm(range(total_samples), ncols=75, desc='Rank File', total=total_samples):

            #print("    |_", getbasename(rkfpath), end='')

            # We are only interested in labels for the topk
            rklabels = labels[i]

            rk_array = read_rank(rkflist[i])

            # Experimental. Generate random variations of ranked scores + labels by removing a random sample of
            # elements from the rank.
            variations = generate_rank_variations(rk_array, rklabels, self.__nvar, self.__varf)

            sample_vectors = [[] for _ in range(0, self.__nvar)]
            for v in tqdm(range(self.__nvar), ncols=75, desc='  .Variation', total=self.__nvar):

                rknames, rkscores, rklabels = variations[v]

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

                    pidx = get_index(self.__namelist['name'], rk_array['name'][0:self.__topk])

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

                if self.__contextual_check:

                    aux = get_name_in_coll(rkflist[i], self.__dkey)

                    try:
                        gfvargs['query_idx'] = np.argwhere(self.__namelist['name'] == aux)[0, 0]
                    except:
                        pass
                        #pdb.set_trace()

                    gfvargs['topk_idx'] = np.zeros(self.__topk, dtype=np.int32) - 1
                    for r in range(0, self.__topk):
                        gfvargs['topk_idx'][r] = np.argwhere(self.__namelist['name'] == rknames[r])[0, 0]

                    gfvargs['cid_list'] = self.__namelist['cid']

                if self.__correlation_check:
                    gfvargs['corr_mat'] = ktau_matrix(gfvargs['coll_matches'], gfvargs['topk_idx'])
                    #pdb.set_trace()

                ### Extraction ###
                # Per top-k extraction
                for r in range(0, self.__topk):  # Iterates over rank positions

                    gfvargs['i'] = r
                    fvector = []
                    for featalias in self.__feature_queue:
                        fvector.append(get_rank_feature(featalias, **gfvargs))
                    fvector = np.hstack(fvector).reshape(1, -1)

                    assert fvector.size == self.__fv_dim, "Inconsistent feature vector size <{0:d}> with " \
                                                              "precomputed feature vector dimension <{1:d}> ".format(fvector.size, self.__fv_dim)

                    outlabels[r, i, v, 0] = rklabels[r]
                    outfeatures[r, i, v, 0:self.__fv_dim] = fvector

                if self.__get_next or self.__get_previous:
                    for r in range(0, self.__topk):
                        catfeat = [outfeatures[r, i, v, 0:self.__fv_dim]]

                        if self.__get_previous:
                            if r > 0:
                                catfeat.append(outfeatures[r - 1, i, v, 0:self.__fv_dim])
                            else:
                                catfeat.append(np.zeros(self.__fv_dim, dtype=np.float64))

                        if self.__get_next:
                            if r < self.__topk-1:
                                catfeat.append(outfeatures[r + 1, i, v, 0:self.__fv_dim])
                            else:
                                catfeat.append(np.zeros(self.__fv_dim, dtype=np.float64))

                        outfeatures[r, i, v] = np.hstack(catfeat)


                #pdb.set_trace()
                #if rklabels[0] == 0:
                #    pdb.set_trace()

        #pdb.set_trace()
        np.savez_compressed(outfile, features=outfeatures, labels=outlabels)

        return







