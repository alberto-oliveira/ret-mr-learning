#-*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as fft

from jenkspy import jenks_breaks


def rank_features_kDCT(rank, k, dct_type, notop=False):
    """
    Extracts DCT features from the top-k ranked scores from an input rank. The 'notop' flags points if top-k is from
    position 1 to k (notop = Fslse) or from position 2 to k+1 (notop = True).

    :param rank: numpy array comprising ranked scores.
    :param k: number of top positions considered.
    :param dct_type: type of dct as per scipy.
    :param notop: if False, top-k is from position 1 to k, if True top-k is from position 2 to k+1. Default is False.
    :return: array with DCT coefficients.
    """

    #print(k, dct_type, notop, sep=" --- ")

    if notop:
        topk = rank[1:k+1]
    else:
        topk = rank[:k]

    kdct = fft.dct(topk, dct_type)

    return kdct


def rank_features_shiftDCT(rank, i, k, dct_type):
    """
    Extracts DCT features from the i-th to (i+k)-th position of the rank.

    :param rank: numpy array comprising ranked scores.
    :param i: initial position
    :param k: number of positions
    :param dct_type: type of dct as per scipy.
    :return: array with DCT coefficients.
    """

    #print(k, dct_type, notop, sep=" --- ")

    kdct = fft.dct(rank[i:i+k], dct_type)

    return kdct


def rank_features_deltaik(rank, i, k):
    """
    Extracts the Delta i-k features from a rank. Delta_i-k is defined as <(si - si+1), (si - si+2), ..., (si - sk)>, for
    ranked scores {s1, s2, ..., si, ..., sk, ..., sn}


    :param rank: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining final position k
    :return: numpy array of features
    """

    fv = []

    if k >= rank.shape[0]:
        k = rank.shape[0]-1

    for p in range(i+1, k+1):

        diff = rank[i] - rank[p]
        fv.append(diff)

    return np.array(fv)


def rank_features_circ_deltaik(rank, i, k):
    """
    Extracts the circular Delta i-k features from a rank. circ_Delta_i-k_ is defined as <(si - s1), (si - s2), ...,
    (si - si-1), (si - si+1), ..., (si - sk)>, for ranked scores {s1, s2, ..., si, ..., sk, ..., sn}

    :param rank: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining final position k
    :return: numpy array of features
    """

    fv = []

    if k >= rank.shape[0]:
        k = rank.shape[0]-1

    for p in range(0, k+1):

        if p != i:
            diff = rank[i] - rank[p]
            fv.append(diff)

    return np.array(fv)


def rank_features_cluster_diff(rank, i, k, c, centers=[]):
    """
    Extracts the Cluster difference features from a rank. Cluster difference features for position i are defined as:
    <|si - m0|, |si - m1|, |si - m2|, ..., |si - mc|>, where {m0, m1, ..., mc} are m clusters found by jenks-breaks
    optimization on the scores of the tail, defined as {sk+1, sk+2, ...., sn}


    :param rank: numpy array with ranked scores
    :param i: integer defining anchor position i
    :param k: integer defining initial position of the tail k
    :param c: number of clusters to be found by jenks breaks optimization
    :return: numpy array of features
    """

    if centers == []:
        centers = np.sort(np.array(jenks_breaks(rank[k:].reshape(-1), c - 1)))

    fv = np.abs(rank[i] - centers)

    return np.array(fv)