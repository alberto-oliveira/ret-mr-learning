#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np


def create_aggr_table(rnamelist, rscorelist, rlabellist):

    from functools import reduce

    nranks = len(rnamelist)
    nrows = rnamelist[0].size  # By construction, this value should be consistent between all ranks

    aggr_names = reduce(np.union1d, rnamelist)

    # if the object at position i is not present in Rank_j,
    # then aggr_scores[i, j] = -1
    aggr_scores = np.zeros((aggr_names.size, nranks), dtype=np.float64) - 1

    # Labels for aggregation are: 'relevant' = 1, 'non-relevant' = -1
    # and 'not-present' = 0
    aggr_labels = np.zeros((aggr_names.size, nranks), dtype=np.int32)

    for i in range(nranks):

        names = rnamelist[i]
        scores = rscorelist[i]
        labels = rlabellist[i]

        for k in range(nrows):

            n = names[k]
            s = scores[k]
            l = labels[k]

            pos = np.argwhere(aggr_names == n)
            aggr_scores[pos, i] = s

            if l == 1:
                aggr_labels[pos, i] = 1
            else:
                aggr_labels[pos, i] = -1

    return aggr_names, aggr_scores, aggr_labels


def aggr_combSUM(aggr_names, aggr_scores, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.array(aggr_scores)
    np.place(aggr_scores_z, aggr_scores_z == -1, 0)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    sum_scores = np.sum(aggr_scores_z, axis=1) * weights
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combSUM_plus(aggr_names, aggr_scores, aggr_labels, f=0.0, balance=0, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.array(aggr_scores)
    np.place(aggr_scores_z, aggr_scores_z == -1, 0)


    # For the labels, we use an auxiliary version of the aggr_labels table, which
    # employs three types of balance measures:
    #  0 - Positive and Negative predictions are accounted for
    # +1 - Only Positiuve predictions are accounted for
    # -1 - Only Negative predictions are accounted for
    if balance != 0:
        aggr_labels_ = aggr_labels.copy()
        np.place(aggr_labels_, aggr_labels != balance, 0)
    else:
        aggr_labels_ = aggr_labels

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    label_weights = (f * aggr_labels_) + 1

    sum_scores = np.sum(aggr_scores_z * label_weights, axis=1) * weights
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combMIN(aggr_names, aggr_scores):

    # For the min, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for Inf
    aggr_scores_inf = np.array(aggr_scores)
    np.place(aggr_scores_inf, aggr_scores_inf == -1, np.Inf)

    min_scores = np.min(aggr_scores_inf, axis=1)
    sidx = np.argsort(min_scores)[::-1]

    return aggr_names[sidx], min_scores[sidx]


def aggr_combMAX(aggr_names, aggr_scores):

    max_scores = np.max(aggr_scores, axis=1)
    sidx = np.argsort(max_scores)[::-1]

    return aggr_names[sidx], max_scores[sidx]


def aggr_combMNZ(aggr_names, aggr_scores, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.array(aggr_scores)
    np.place(aggr_scores_z, aggr_scores_z == -1, 0)

    p = (aggr_scores != -1).astype(np.int32)
    sum_p = np.sum(p, axis=1)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    sum_scores = np.sum(aggr_scores_z, axis=1) * sum_p * weights
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]




