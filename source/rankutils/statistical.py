#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

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





