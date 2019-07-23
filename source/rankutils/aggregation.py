#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
#import ipdb as pdb


def labeled_rerank(labels, final_pos=1, order=np.array([])):

    if order.size == 0:
        order = np.arange(labels.shape[0])

    i = labels.shape[0] - 1

    while i > final_pos:

        label_a = labels[i]
        label_b = labels[i-1]

        if label_a == 1 and label_b == -1:
            order[i-1], order[i] = order[i], order[i-1]
            i -= 2
        else:
            i -= 1

    return order


def create_aggr_table(rnamelist, rscorelist, rlabellist, absent=0):

    from functools import reduce

    nranks = len(rnamelist)
    nrows = rnamelist[0].size  # By construction, this value should be consistent between all ranks

    aggr_names = reduce(np.union1d, rnamelist)

    # if the object at position i is not present in Rank_j,
    # then aggr_scores[i, j] = -1
    aggr_scores = np.zeros((aggr_names.size, nranks), dtype=np.float64) - 1

    # There are four possible cases for aggregation labels:
    # (1) Predicted relevant -- label = 1
    # (2) Predicted non-relevant -- label = -1
    # (3) Present in rank, but not predicted -- label = 0
    # (4) Not present in rank -- label = args('absent')
    aggr_labels = np.zeros((aggr_names.size, nranks), dtype=np.float64) + absent

    aggr_standings = np.zeros((aggr_names.size, nranks), dtype=np.int32)

    for i in range(nranks):

        names = rnamelist[i]
        scores = rscorelist[i]
        labels = rlabellist[i]

        for k in range(nrows):

            n = names[k]
            s = scores[k]
            try:
                l = labels[k]
            except IndexError:
                l = np.NaN

            pos = np.argwhere(aggr_names == n)
            aggr_scores[pos, i] = s

            if labels.dtype == np.uint8:
                if l == 1:
                    aggr_labels[pos, i] = 1
                elif l == 0:
                    aggr_labels[pos, i] = -1
                elif np.isnan(l):
                    aggr_labels[pos, i] = 0
            else:
                if np.isnan(l):
                    aggr_labels[pos, i] = 0.5
                else:
                    aggr_labels[pos, i] = l

            aggr_standings[pos, i] += (nrows-k)

    shuffle_idx = np.arange(aggr_names.shape[0])
    np.random.shuffle(shuffle_idx)

    return aggr_names[shuffle_idx], aggr_scores[shuffle_idx], aggr_labels[shuffle_idx], aggr_standings[shuffle_idx]


def aggr_combSUM(aggr_names, aggr_scores, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    sum_scores = np.array(np.ma.sum(aggr_scores_z, axis=1)) * weights
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combSUM_pre(aggr_names, aggr_scores, aggr_labels, f=0.0, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    aggr_labels_ = np.ma.array(aggr_labels, mask=aggr_scores_.mask)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    # Pre weighting
    if aggr_labels_.dtype == np.uint8:
        label_weights = (f * aggr_labels_) + 1
    else:
        label_weights = f*(aggr_labels_ - 0.5) + 1

    sum_scores = np.array(np.ma.sum(aggr_scores_ * label_weights, axis=1) * weights)
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combSUM_post_avgw(aggr_names, aggr_scores, aggr_labels, f=0.0):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    # Post weighting
    label_weights = (f * np.mean(aggr_labels, axis=1)) + 1

    #pdb.set_trace()
    sum_scores = np.array(np.ma.sum(aggr_scores_z, axis=1) * label_weights)
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combSUM_post_majw(aggr_names, aggr_scores, aggr_labels, f=0.0):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    aggr_labels_ = np.ma.array(aggr_labels, mask=aggr_scores_z.mask)

    #pdb.set_trace()

    # Post weighting
    majority_votes = np.array(np.clip(np.ma.sum(aggr_labels_, axis=1), -1, 1))

    label_weights = (f * majority_votes) + 1

    sum_scores = np.array(np.ma.sum(aggr_scores_z, axis=1) * label_weights)
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combMIN(aggr_names, aggr_scores):

    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    min_scores = np.ma.min(aggr_scores_, axis=1)
    sidx = np.argsort(min_scores)[::-1]

    return aggr_names[sidx], min_scores[sidx]


def aggr_combMIN_post_avgw(aggr_names, aggr_scores, aggr_labels, f=0.0):

    # For the min, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for Inf
    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    label_weights = (f * np.mean(aggr_labels, axis=1)) + 1

    min_scores = np.array(np.ma.min(aggr_scores_, axis=1) * label_weights)
    sidx = np.argsort(min_scores)[::-1]

    return aggr_names[sidx], min_scores[sidx]


def aggr_combMIN_post_majw(aggr_names, aggr_scores, aggr_labels, f=0.0):

    # For the min, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for Inf
    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)
    aggr_labels_ = np.ma.array(aggr_labels, mask=aggr_scores_.mask)

    label_weights = (f * np.array(np.clip(np.ma.sum(aggr_labels_, axis=1), -1, 1))) + 1

    min_scores = np.array(np.ma.min(aggr_scores_, axis=1) * label_weights)
    sidx = np.argsort(min_scores)[::-1]

    return aggr_names[sidx], min_scores[sidx]


def aggr_combMAX(aggr_names, aggr_scores):

    max_scores = np.max(aggr_scores, axis=1)
    sidx = np.argsort(max_scores)[::-1]

    return aggr_names[sidx], max_scores[sidx]


def aggr_combMINMAX(aggr_names, aggr_scores):

    aggr_scores_mask = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    max_scores = np.array(np.ma.max(aggr_scores_mask, axis=1))
    min_scores = np.array(np.ma.min(aggr_scores_mask, axis=1))

    minmax_scores = (max_scores + min_scores)/2
    sidx = np.argsort(max_scores)[::-1]

    return aggr_names[sidx], minmax_scores[sidx]


def aggr_combMNZ(aggr_names, aggr_scores, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    p = (aggr_scores != -1).astype(np.int32)
    sum_p = np.sum(p, axis=1)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    sum_scores = np.array(np.ma.sum(aggr_scores_z, axis=1) * sum_p * weights)
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_combMNZ_pre(aggr_names, aggr_scores, aggr_labels, f=0.0, weights=None):

    # For the sum, we use an auxiliary version of the aggr_scores table which
    # replaces the -1s for 0s
    aggr_scores_z = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    p = (aggr_scores != -1).astype(np.int32)
    sum_p = np.sum(p, axis=1)

    if not weights:
        weights = np.ones(aggr_scores.shape[0], dtype=np.int32)

    # Pre weighting
    label_weights = (f * aggr_labels) + 1

    sum_scores = np.array(np.ma.sum(aggr_scores_z * label_weights, axis=1) * sum_p * weights)
    sidx = np.argsort(sum_scores)[::-1]

    return aggr_names[sidx], sum_scores[sidx]


def aggr_bordaCount(aggr_names, aggr_scores, aggr_standings):

    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    # Use the mean score to break ties
    mean_scores = np.array(np.ma.mean(aggr_scores_, axis=1))

    borda_scores = np.sum(aggr_standings, axis=1)

    final_scores = borda_scores + mean_scores
    sidx = np.argsort(final_scores)[::-1]

    #pdb.set_trace()

    return aggr_names[sidx], final_scores[sidx]


def aggr_bordaCount_post_majw(aggr_names, aggr_scores, aggr_standings, aggr_labels, f=1.0):

    aggr_scores_ = np.ma.array(aggr_scores, mask=aggr_scores == -1)
    aggr_labels_ = np.ma.array(aggr_labels, mask=aggr_scores_.mask)

    majority_votes = np.array(np.clip(np.ma.sum(aggr_labels_, axis=1), -1, 1))

    label_weights = (f * majority_votes) + 1

    # Use the mean score to break ties
    mean_scores = np.array(np.ma.mean(aggr_scores_, axis=1))

    borda_scores = np.sum(aggr_standings, axis=1)

    final_scores = (borda_scores + mean_scores) * label_weights
    sidx = np.argsort(final_scores)[::-1]

    return aggr_names[sidx], final_scores[sidx]











def aggr_combMEAN(aggr_names, aggr_scores):

    aggr_scores_mask = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    mean_scores = np.array(np.ma.mean(aggr_scores_mask, axis=1))
    sidx = np.argsort(mean_scores)[::-1]

    return aggr_names[sidx], mean_scores[sidx]


def aggr_combMEAN_plus(aggr_names, aggr_scores, aggr_labels, f=0.0):

    aggr_scores_mask = np.ma.array(aggr_scores, mask=aggr_scores==-1)

    label_weights = (f * np.mean(aggr_labels, axis=1)) + 1

    mean_scores = np.array(np.ma.mean(aggr_scores_mask, axis=1)) * label_weights
    sidx = np.argsort(mean_scores)[::-1]

    return aggr_names[sidx], mean_scores[sidx]


def aggr_combMEDIAN(aggr_names, aggr_scores):

    aggr_scores_mask = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    median_scores = np.array(np.ma.median(aggr_scores_mask, axis=1))
    sidx = np.argsort(median_scores)[::-1]

    return aggr_names[sidx], median_scores[sidx]


def aggr_combMEDIAN_plus(aggr_names, aggr_scores, aggr_labels, f=0.0):

    aggr_scores_mask = np.ma.array(aggr_scores, mask=aggr_scores == -1)

    label_weights = (f * np.mean(aggr_labels, axis=1)) + 1

    median_scores = np.array(np.ma.median(aggr_scores_mask, axis=1)) * label_weights
    sidx = np.argsort(median_scores)[::-1]

    return aggr_names[sidx], median_scores[sidx]