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


def EMD(densa, edgesa, densb, edgesb):

    from scipy.stats import wasserstein_distance

    bsa = np.abs((edgesa[1] - edgesa[0]))
    bsb = np.abs((edgesb[1] - edgesb[0]))

    pa = bsa * densa
    pb = bsb * densb

    return wasserstein_distance(pa, pb)


def Bhattacharyya_coefficients(densa, edgesa, densb, edgesb):

    bsa = np.abs((edgesa[1] - edgesa[0]))
    bsb = np.abs((edgesb[1] - edgesb[0]))

    pa = bsa*densa
    pb = bsb*densb

    return np.sqrt(pa*pb)


def ev_density_approximation(d, lowerb, upperb, bins, input_engine=None):

    import matlab
    if not input_engine:
        import matlab.engine
        matlab_engine = matlab.engine.start_matlab()
    else:
        matlab_engine = input_engine

    edges = np.linspace(lowerb, upperb, bins + 1).reshape(-1)
    aux = matlab.double(edges[:-1].tolist())

    if d['name'] == 'WBL':
        dens = np.array(matlab_engine.wblpdf(aux, d['scale'], d['shape']), dtype=np.float64).reshape(-1)

    elif d['name'] == 'GEV':
        dens = np.array(matlab_engine.gevpdf(aux, d['shape'], d['scale'], d['loc']), dtype=np.float64).reshape(-1)

    if not input_engine:
        matlab_engine.quit()

    return dens, edges


def ev_fit(data, disttype, input_engine=None):

    import matlab
    if not input_engine:
        import matlab.engine
        matlab_engine = matlab.engine.start_matlab()
    else:
        matlab_engine = input_engine

    if isinstance(data, matlab.double):
        data_m = data
    elif isinstance(data, np.ndarray):
        data_m = matlab.double(data[data != 0].reshape(-1).tolist())
    elif isinstance(data, (tuple, list)):
        data_m = matlab.double(data)
    else:
        raise TypeError("ev_fit() arg 1 must be a matlab.double, numpy.ndarray, list, or of floats instance.")

    distb = dict(name=disttype)

    if disttype == 'WBL':
        estpar, _ = matlab_engine.wblfit(data_m, nargout=2)
        distb['scale'] = estpar[0][0]
        distb['shape'] = estpar[0][1]
        distb['loc'] = 0

    elif disttype == 'GEV':
        opt = matlab_engine.statset('Display', 'off', 'MaxIter', 2000.0, 'MaxFunEval', 15000.0)
        estpar, _ = matlab_engine.gevfit(data_m, [], opt, nargout=2)
        distb['scale'] = estpar[0][1]
        distb['shape'] = estpar[0][0]
        distb['loc'] = estpar[0][2]

    if not input_engine:
        matlab_engine.quit()

    return distb





