#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from jenkspy import jenks_breaks

#import ipdb as pdb


def clustering_1d(clalias, **ka):

    assert ka['data'].ndim == 1 or ka['data'].shape[0] == 1 or ka['data'].shape[1] == 1, \
           "Input data must be 1d. It is {0:s}".format(str(ka['data'].shape))

    if ka['data'].ndim > 1:
        data = ka['data'].reshape(-1)
    else:
        data = ka['data']

    if clalias == 'jenks':
        return jenks_breaks(data, ka['c_num'] - 1)
    elif clalias == 'random':
        return random_clustering(data, ka['c_num'])
    elif clalias == 'fixed':
        return fixed_points_clustering(data, ka['c_num'])
    else:
        raise ValueError("<{0:s}> is not a valid 1d clustering alias. ".format(clalias))


def head_tail_break(data, t=0.4):

    assert data.ndim == 1, "Data should be 1-Dimensional."

    breaks = []

    if t >= 1:
        t = 1.0
    elif t < 0.1:
        t = 0.1

    mean = np.mean(data)
    head_idx = np.argwhere(data >= mean)

    head = data[head_idx]

    if head.size/data.size <= t:
        aux = head_tail_break(head.reshape(-1), t)
        breaks = aux + [mean]

    return breaks


def head_tail_clustering(data, t=0.4):

    mean_breaks = head_tail_break(data, t)

    clustering = np.zeros((data.size), dtype=np.float32)
    cluster_centers = np.zeros((len(mean_breaks) + 1), dtype=np.float32)

    for b in mean_breaks:
        clustering[data >= b] += 1

    for c in np.arange(len(mean_breaks), -1, -1):
        i = len(mean_breaks) - c
        cidx = np.argwhere(clustering == c)

        cluster_centers[i] = np.mean(data[cidx])

    return clustering, cluster_centers


def diff_break_clustering(data, dev_factor=0, min_clusters=-1):

    diffs = np.abs((data[0:-1] - data[1:]).reshape(-1))

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    breaks = np.argwhere(diffs >= (mean_diff + dev_factor*std_diff)).reshape(-1) + 1

    if breaks.size == 0:
        return data

    cluster_centers = np.zeros((breaks.size+1), dtype=np.float32)

    for i in range(0, breaks.size + 1):

        if i == 0:
            vals = data[0:breaks[i]]
        elif i == breaks.size:
            vals = data[breaks[i-1]:]
        else:
            vals = data[breaks[i-1]:breaks[i]]

        cluster_centers[i] = np.median(vals)

    return cluster_centers


def random_clustering(data, c):

    import numpy.random as rnd
    rnd.seed(93311)

    ri = rnd.randint(0, data.size, c)
    return data[ri]


def fixed_points_clustering(data, c):

    #pdb.set_trace()
    if c > data.size:
        raise ValueError("Number of clusters <{0:d}> must be less than total data size <{1:d}>".format(c, data.size))

    step = int(np.floor(data.size / (c-1)))
    idx = np.arange(0, step*c, step)

    if idx[-1] >= data.size:
        idx[-1] = data.size-1

    return data[idx]
