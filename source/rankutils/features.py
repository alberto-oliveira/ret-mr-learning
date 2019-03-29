#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as fft

from sklearn.preprocessing import normalize

from jenkspy import jenks_breaks

#import ipdb as pdb


def get_rank_feature(featalias, **ka):

    if featalias == 'dct':
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
    else:
        raise ValueError("<{0:s}> is not a valid feature alias. ".format(featalias))


def rank_features_kDCT(scores, k, dct_type, norm=False):
    """
    Extracts DCT features from the top-k ranked scores from an input rank. The 'notop' flags points if top-k is from
    position 1 to k (notop = Fslse) or from position 2 to k+1 (notop = True).

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

