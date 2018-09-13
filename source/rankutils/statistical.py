#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np


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


"""
https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
"""
def MAD_outlier(data, t=3.5):
    """
    Returns a boolean array with True if data are outliers and False
    otherwise.

    Parameters:
    -----------
        data : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    median = np.median(data, axis=0)
    diff = np.sum((data - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > t


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


def ev_values(d, n, bounds, input_engine=None):

    import matlab
    if not input_engine:
        import matlab.engine
        matlab_engine = matlab.engine.start_matlab()
    else:
        matlab_engine = input_engine

    lb, ub = bounds
    x = matlab.double(np.linspace(0, np.floor(ub * 1.5), n).reshape(-1).tolist())
    x_cdf = matlab.double(np.linspace(lb, ub, n).reshape(-1).tolist())
    x_inv = matlab.double(np.linspace(0, 1, n, endpoint=False).reshape(-1).tolist())

    d['x'] = np.array(x).reshape(-1)
    d['xcdf'] = np.array(x_cdf).reshape(-1)
    d['xinv'] = np.array(x_inv).reshape(-1)

    if d['name'] == 'WBL':
        d['pdf'] = np.array(matlab_engine.wblpdf(x, d['scale'], d['shape']), dtype=np.float64).reshape(-1)
        d['cdf'] = np.array(matlab_engine.wblcdf(x_cdf, d['scale'], d['shape']), dtype=np.float64).reshape(-1)
        d['inv'] = np.array(matlab_engine.wblinv(x_inv, d['scale'], d['shape']), dtype=np.float64).reshape(-1)

    elif d['name'] == 'GEV':
        d['pdf'] = np.array(matlab_engine.gevpdf(x, d['shape'], d['scale'], d['loc']), dtype=np.float64).reshape(-1)
        d['cdf'] = np.array(matlab_engine.gevcdf(x_cdf, d['shape'], d['scale'], d['loc']), dtype=np.float64).reshape(-1)
        d['inv'] = np.array(matlab_engine.gevinv(x_inv, d['shape'], d['scale'], d['loc']), dtype=np.float64).reshape(-1)

    return


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





