#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

from rankutils.rIO import read_relfile
from rankutils.utilities import getbasename


def create_labeling_array(lbllist):
    """
    Creates structured numpy array of labels, from a list of label tuples.

    :param lbllist: list of label tuples
    :return: structured numpy array os labels
    """

    # Does not count the name of the file itself, only the k labels
    k = len(lbllist[0]) - 1

    nmtuple = tuple(['file'] + [str(x) for x in range(k)])
    fmttuple = tuple(['U100'] + [np.int32 for x in range(k)])

    dt = dict(names=nmtuple, formats=fmttuple)

    lblarray = np.array(lbllist, dtype=dt)

    return lblarray


def write_labeling_file(outfile, lblarray, k):
    """
    Writes the labeling file. Each row is the basename of a rankfile, followed by a sequence of k integers in {0, 1},
    in which the i-th integer is 1 if the rank is accepted when M = i. In practice, this means that, in positions 2 to
    k+1 (remember, the first position does not count for top-k, as per our formulation) there are at least M = i
    relevant or predicted relevant objects.

    :param outfile: path of the output file.
    :param lblarray: structured numpy array containing rankfile names and acceptance checks.
    :param k: number of top-k positions to consider.
    :return:
    """



    fmtstr = "%-50s"
    for i in range(k):
        fmtstr += " %d"

    np.savetxt(outfile, lblarray, fmt=fmtstr)

def pp_labeling(relfile, k, notop=False):
    """
    Labels rank as accepted/rejected according to the relevance of the top-k ranked objects, considering a threshold
    M that varies from 1 to k. Can label either groundtruth files, which will generate a groundtruth accept labels, or
    relevance prediction files, which will generate performance prediction labels. NEVER counts the rank-1 position of
    the rank, meaning that top-k -> positions 2 to k+1

    :param relfile: relevance file for rank. Can be either a groundtruth file or a predicted relevance file.
    :param k: number of positions of interest. Ignores the rank-1 position.
    :return: numpy vector of 1s and 0s, in which the ith position is 1 if in positions 2 to k+1 there are at least
             i+1 relevant objects, and 0 otherwise. For example, if position 3 of the output vector is 1, then the
             number of relevant objects in positions 2 to k+1 is AT LEAST 4.
    """

    relarray = read_relfile(relfile)

    if notop:
        pk = np.sum(relarray['rel'][1:k+1])
    else:
        pk = np.sum(relarray['rel'][0:k])

    outtp = np.array([int(pk >= x) for x in range(1, k+1)], dtype=np.uint8)

    return outtp

def rp_labeling(relfile, k, notop=False):
    """
    Synthetizes a relevance file into a row vector of relevants. Remembering, the raw relevance files contains n
    rows of pairs <collection_object> <R>, in which n is the number of objects returned for the rank,
    <collection_object> is the name of the object in the collection, and R in {0, 1} is 1 if it is relevant to the
    query, and 0 otherwise.

    :param relfile: relevance file for rank. Can be either a groundtruth file or a predicted relevance file.
    :param k: number of positions of interest. Ignores the rank-1 position.
    :param notop: if true, does not consider the rank-1 position to generate the labels.
    :return: numpy array of shape (1, k), where the i-th column is the relevance of the i+2th element of the rank.
    """

    relarray = read_relfile(relfile)

    if notop:
        outarray = relarray['rel'][1:k+1].astype(np.uint8)
    else:
        outarray = relarray['rel'][0:k].astype(np.uint8)

    return outarray.reshape((1, -1))
