#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as fft

from sklearn.preprocessing import normalize

from jenkspy import jenks_breaks

#import ipdb as pdb


def get_rank_feature(featalias, **ka):

    if featalias == 'raw_scores':
        return  rank_features_raw_scores(ka['scores'], ka['i'])
    elif featalias == 'dct':
        return rank_features_kDCT(ka['scores'], ka['topk'], ka['dct_type'], ka['norm'])
    elif featalias == 'dct_shift':
        return rank_features_shiftDCT(ka['scores'], ka['i'], ka['dct_range'], ka['dct_type'], ka['norm'])
    elif featalias == 'deltaik':
        return rank_features_deltaik(ka['scores'], ka['i'], ka['delta_range']), ka['norm']
    elif featalias == 'deltaik_c':
        return rank_features_circ_deltaik(ka['scores'], ka['i'], ka['delta_range'], ka['abs'], ka['norm'])
    elif featalias == 'cluster_diff':
        return rank_features_cluster_diff(ka['scores'], ka['i'], ka['topk'], ka['cluster_num'], ka['centers'],
                                          ka['norm'])
    elif featalias == 'emd':
        return rank_features_density_distance(ka['densities'], ka['edges'], ka['i'], ka['distmat'], ka['norm'])
    elif featalias == 'query_bhatt':
        return rank_features_density_distance_from_query(ka['q_density'], ka['q_edges'],
                                                         ka['densities'], ka['edges'], ka['i'], ka['norm'])
    elif featalias == 'rank_jacc':
        pidx = ka['topk_idx'][ka['i']]
        return rank_features_interval_jaccard(ka['coll_matches'], ka['query_idx'], pidx, ka['num_intv'], ka['norm'])
    elif featalias == 'accum_jacc':
        pidx = ka['topk_idx'][ka['i']]
        return rank_features_accum_jaccard(ka['coll_matches'], ka['query_idx'], pidx, ka['num_intv'], ka['norm'])
    elif featalias == 'cid_jacc':
        pidx = ka['topk_idx'][ka['i']]
        return rank_features_interval_cid_jaccard(ka['coll_matches'], ka['cid_list'], ka['query_idx'], pidx,
                                                  ka['num_intv'], ka['norm'])
    elif featalias == 'cid_freq_diff':
        i = ka['topk_idx'][ka['i']]
        return rank_features_cid_frequency_diff(ka['coll_matches'], ka['cid_list'], ka['query_idx'], i, ka['norm'])

    elif featalias == 'ktau_top':
        return rank_features_topk_correlation(ka['corr_mat'], ka['i'])

    else:
        raise ValueError("<{0:s}> is not a valid feature alias. ".format(featalias))


def rank_features_kDCT(scores, k, dct_type, norm=False):
    """
    Extracts DCT features from the top-k ranked scores from an input rank. The 'notop' flags points if top-k is from
    position 1 to k (notop = False) or from position 2 to k+1 (notop = True).

    :param scores: numpy array comprising ranked scores.
    :param k: number of top positions considered.
    :param dct_type: type of dct as per scipy.
    :return: array with DCT coefficients.
    """

    #print(k, dct_type, notop, sep=" --- ")

    topk = scores[:k]

    kdct = fft.dct(topk, dct_type)

    if norm:
        return normalize(kdct.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return kdct


def rank_features_shiftDCT(scores, i, k, dct_type, norm=False):
    """
    Extracts DCT features from the i-th to (i+k)-th position of the scores.

    :param scores: numpy array comprising ranked scores.
    :param i: initial position
    :param k: number of positions
    :param dct_type: type of dct as per scipy.
    :return: array with DCT coefficients.
    """

    #print(k, dct_type, notop, sep=" --- ")

    kdct = fft.dct(scores[i:i + k], dct_type)

    if norm:
        return normalize(kdct.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return kdct


def rank_features_deltaik(scores, i, k, norm=False):
    """
    Extracts the Delta i-k features from a scores. Delta_i-k is defined as <(si - si+1), (si - si+2), ..., (si - sk)>, for
    ranked scores {s1, s2, ..., si, ..., sk, ..., sn}


    :param scores: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining final position k
    :return: numpy array of features
    """

    featlist = []

    if k >= scores.shape[0]:
        k = scores.shape[0] - 1

    for p in range(i+1, k+1):

        diff = scores[i] - scores[p]
        featlist.append(diff)

    fv = np.array(featlist)

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_circ_deltaik(scores, i, k, abs=False, norm=False):
    """
    Extracts the circular Delta i-k features from a scores. circ_Delta_i-k_ is defined as <(si - s1), (si - s2), ...,
    (si - si-1), (si - si+1), ..., (si - sk)>, for ranked scores {s1, s2, ..., si, ..., sk, ..., sn}

    :param scores: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining final position k. Should be >= than i
    :return: numpy array of features
    """

    featlist = []

    assert k >= i, "The final position k:<{0:d}> should be >= the anchor position i:<{1:d}>".format(k, i)

    if k >= scores.shape[0]:
        k = scores.shape[0] - 1

    for p in range(0, k+1):

        if p != i:
            diff = scores[i] - scores[p]
            if abs:
                diff = np.abs(diff)
            featlist.append(diff)

    fv = np.array(featlist)

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_cluster_diff(scores, i, k, c, centers=[], norm=False):
    """
    Extracts the Cluster difference features from a rank. Cluster difference features for position i are defined as:
    <|si - m0|, |si - m1|, |si - m2|, ..., |si - mc|>, where {m0, m1, ..., mc} are m clusters found by jenks-breaks
    optimization on the scores of the tail, defined as {sk+1, sk+2, ...., sn}


    :param scores: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining initial position of the tail k
    :param c: number of clusters to be found by jenks breaks optimization
    :return: numpy array of features
    """

    if centers == []:
        centers = np.sort(np.array(jenks_breaks(scores[k:].reshape(-1), c - 1)))

    fv = np.array(np.abs(scores[i] - centers))

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_raw_scores(scores, i):

    return scores[i]


def rank_features_density_distance(densities, edges, i, distmat=np.array([]), norm=False):

    from rankutils.statistical import EMD

    t = len(densities)
    i_density = densities[i]
    i_edges = edges[i]

    if distmat.size != t*t:
        distmat = np.zeros((t, t), dtype=np.float64) - 1

    for j in range(t):
        if j != i:

            j_density = densities[j]
            j_edges = edges[j]

            if distmat[j, i] != -1:
                distmat[i, j] = distmat[j, i]
            else:
                distmat[i, j] = EMD(i_density, i_edges, j_density, j_edges)
                distmat[j, i] = distmat[i, j]

    fv = np.hstack([distmat[i, 0:i], distmat[i, i+1:]])

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_density_distance_from_query(q_density, q_edges, r_densities, r_edges, i, norm=False):

    from rankutils.statistical import Bhattacharyya_coefficients

    curr_density = r_densities[i]
    curr_edges = r_edges[i]

    fv = Bhattacharyya_coefficients(q_density, q_edges, curr_density, curr_edges)

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_interval_jaccard(collmatches, qidx, pidx, n, norm=False):

    if not np.all(collmatches[pidx] != -1) or not np.all(collmatches[qidx] != -1):
        pdb.set_trace()

    q_split = np.array_split(collmatches[qidx], n)
    p_split = np.array_split(collmatches[pidx], n)

    fv = np.zeros(n, dtype=np.float64)

    for i in range(n):

        intersection = np.intersect1d(q_split[i], p_split[i])
        union = np.union1d(q_split[i], p_split[i])

        fv[i] = intersection.size/union.size

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_accum_jaccard(collmatches, qidx, pidx, n, norm=False):

    q_split = np.array_split(collmatches[qidx], n)
    p_split = np.array_split(collmatches[pidx], n)

    fv = np.zeros(n, dtype=np.float64)

    accum_p = q_split[0]
    accum_q = p_split[0]

    intersection = np.intersect1d(accum_q, accum_p)
    union = np.union1d(accum_q, accum_p)

    fv[0] = intersection.size / union.size

    for i in range(1, n):

        accum_q = np.hstack([accum_q, q_split[i]])
        accum_p = np.hstack([accum_p, p_split[i]])

        intersection = np.intersect1d(accum_q, accum_p)
        union = np.union1d(accum_q, accum_p)

        fv[i] = intersection.size/union.size

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_interval_cid_jaccard(collmatches, cid, qidx, pidx, n, norm=False):

    # For some idx 'qidx', gets the index in the dataset of the top-l, then gets the CID of those top-l
    q_split_cid = np.array_split(cid[collmatches[qidx]], n)
    # For some idx 'pidx', gets the index in the dataset of the top-l, then gets the CID of those top-l
    p_split_cid = np.array_split(cid[collmatches[pidx]], n)

    fv = np.zeros(n, dtype=np.float64)

    for i in range(n):

        intersection = np.intersect1d(q_split_cid[i], p_split_cid[i])
        union = np.union1d(q_split_cid[i], p_split_cid[i])

        fv[i] = intersection.size/union.size

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_cid_frequency_diff(collmatches, cid, qidx, i, norm=False):

    i_topn_cid = cid[collmatches[i]]
    q_topn_cid = cid[collmatches[qidx]]
    nc = np.max(cid) + 1

    q_freq = np.bincount(q_topn_cid, minlength=nc)
    i_freq = np.bincount(i_topn_cid, minlength=nc)

    fv = np.abs(q_freq - i_freq)

    if norm:
        return normalize(fv.reshape(1, -1), norm='l2').reshape(-1)
    else:
        return fv


def rank_features_topk_correlation(corrmat, i):

    return corrmat[i]


