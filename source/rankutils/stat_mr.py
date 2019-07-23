#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import random
import warnings

from rankutils.utilities import ProgressBar

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import matlab.engine

#import ipdb as pdb

np.random.seed(93311)


def unwrap_self(arg, **kwarg):
    return StatMR.statistical_fixed(*arg, **kwarg)


class StatMR(BaseEstimator):

    __opt_metrics_map = dict(ACC=accuracy_score, F1=f1_score, MCC=matthews_corrcoef)
    __matlab_engine = matlab.engine.start_matlab()
    __matlab_engine.warning('off', 'all', nargout=0)
    __gev_opt = __matlab_engine.statset('Display', 'off', 'MaxIter', 2000.0, 'MaxFunEval', 15000.0)

    from matlab.engine import MatlabExecutionError

    def __init__(self, dist_name='WBL', k=10, delta=0.99999999, opt_metric='MCC', method='fixed', verbose=False):

        self.k = k
        self.delta = delta
        self.opt_metric = opt_metric
        self.v = verbose

        self._F = 0.0
        self._Z = 0.0

        self._opt_val = -np.inf

        self.__min_tail_sz = 4

        self.__eval_metric = StatMR.__opt_metrics_map.get(self.opt_metric, None)

        if dist_name == 'WBL' or dist_name == 'GEV':
            self.__dist_name = dist_name
        else:
            raise ValueError('Unsupported distribution type <{type:s}> - options are \'WBL\'(weibull) and '
                             '\'GEV\'(generalized extreme value)'.format(type=dist_name))

        if method == 'fixed':
            self.__method_name = method
            self._method_funct = self.statistical_fixed
        elif method == 'mixt':
            self.__method_name = method
            self._method_funct = self.statistical_mixture
        else:
            raise ValueError('Undefined weibull method <{0:s}> - options are: '.format(method), {'fixed', 'mixt'})

    @property
    def opt_val(self):
        return self._opt_val

    @property
    def method_funct(self):
        return self.__method_name

    @method_funct.setter
    def method_funct(self, method):
        if method == 'fixed':
            self.__method_name = method
            self._method_funct = self.statistical_fixed
        elif method == 'mixt':
            self.__method_name = method
            self._method_funct = self.statistical_mixture
        else:
            raise TypeError('Undefined weibull method <{0:s}> - options are: '.format(method), {'fixed', 'mixt'})

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, F):
        if F <= 0.0:
            self._F = 0.1
        elif F > 1.0:
            self._F = 1.0
        else:
            self._F = F

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        if Z <= 0.0:
            self._Z = 0.1
        elif Z > 1.0:
            self._Z = 1.0
        else:
            self._Z = Z

    def fit(self, X, y, f_val=(0.0, 0.75, 0.2), z_val=(0.4, 1.1, 0.2)):
        """
        Fits the values of F and Z, using X and y.

        For each (F, Z) pair defined in the ranges of f_val and z_val, fits for every rank contained in X a weibull
        distribution to its tail, which is then used to derive the cutoff score according to the value of dehttps://www.okcupid.com/homelta.
        Such cutoff is used to predict the relevance of the top k scores in X, which are in turn compared to their
        true relevance in y. The prediction is evaluated according to optimization metric optMetric, and the best
        values of F and Z kept. Alternatively, accepts single values for F and Z, if they were precomputed. If single
        valuea are provided for both F and Z, no fitting is required.

        :param X: (n, m) array of n ranks, each with m scores. Each rank may contain repeated scores. For accurate
                   predictions, the number of unique samples need to be reasonably greater than k.
        :param y: (n, k) array of {0, 1} labels, describing the relevances of the top k elements of each rank in X.
                   The labels in y should be preprocessed according to the usage or not of the top score when predicting.
        :param f_val: either a triple (fmin, fmax, fstep) of floats or a single float value f. If a triple, will perform
                      the fitting of F according to the range set by the triple. If a single value, sets F = f.
        :param z_val: either a triple (zmin, zmax, zstep) of floats or a single float value z. If a triple, will perform
                      the fitting of Z according to the range set by the triple. if a single value, sets Z = z.
        :return:
        """

        # Splits the training into target and tail. Target are the scores to be predicted
        # while tail is the tail used to fit the weibull distribution. Note that, the tail
        # will still be post processed.

        assert y.shape[1] == self.k, "The number of labels for each sample should be equal to the value of k set."

        if isinstance(f_val, tuple):
            f_range = np.arange(*f_val)
        elif isinstance(f_val, list):
            f_range = np.array(f_val)
        elif isinstance(f_val, float):
            f_range = np.array([f_val])
            self._F = f_val
        else:
            raise TypeError('f_val should either be a tuple, list, or float.')

        if isinstance(z_val, tuple):
            z_range = np.arange(*z_val)
        elif isinstance(z_val, list):
            z_range = np.array(z_val)
        elif isinstance(z_val, float):
            z_range = np.array([z_val])
            self._Z = z_val
        else:
            raise TypeError('z_val should either be a tuple, list, or float.')

        if self._F == 0.0 or self._Z == 0.0:

            if X.size == 0 or y.size == 0:
                raise ValueError("Neither X or y can be empty when fitting.")

            if self.v: print('--- Fitting for f in', f_range, 'and z in', z_range, file=sys.stdout, flush=True)
            self.__find_best_params(X, y, f_range, z_range)

        return

    def predict(self, X):

        targetX = X[:, 0:self.k]
        remainderX = X[:, self.k:]

        nsamples = remainderX.shape[0]

        t_values = np.zeros((nsamples, 1), dtype=np.float32)
        predicted = []

        for i in tqdm(range(nsamples), desc='     -> Predicting ', total=nsamples, ncols=100):

            tail = remainderX[i, :]
            target = targetX[i, :]

            t = self._method_funct(tail, self._F, self._Z)
            t_values[i] = t

            p = (target >= t).astype(np.uint8)
            predicted.append(p)

        predicted = np.vstack(predicted)

        return predicted, t_values

    def get_tail(self, data, f, z):

        tail = np.unique(data)[::-1]
        tail = tail[tail != -1]

        #tail = data[data != -1]

        n = tail.size

        if n < self.__min_tail_sz:
            if self.v:
                #pdb.set_trace()
                print("# of unique scores is less than the minimum tail size of {0:d}".format(self.__min_tail_sz))
            return np.array([]), 0, 0

        start_i = int(np.floor(f*n))                   # starting index of the tail
        tl = int(np.floor(z*tail[start_i:].size))      # tail distribution length

        end_i = start_i + tl                           # ending index of the distribution

        # While the tail distribution does not hit minimum size, increase its size by one on each end that can be
        # increased. If no minimum tail size is hit, return None as tail
        while tl < self.__min_tail_sz and (start_i > 0 or end_i < n):
            if start_i-1 >= 0:
                start_i -= 1
                tl += 1

            if end_i+1 <= n:
                end_i += 1
                tl += 1

        if tl < self.__min_tail_sz:
            if self.v:
                print("Could not extend the tail to reach minimum tail size of {0:d}".format(self.__min_tail_sz))
            return np.array([]), 0, 0

        tail = tail[start_i:end_i]

        if np.min(tail) == 0:
            tail[tail == 0] += 0.00000001

        return tail, start_i, end_i

    def statistical_fixed(self, data, f, z):

        #pdb.set_trace()

        tail, sidx, eidx = self.get_tail(data, f, z)

        if tail.size == 0:
            if self.v:
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1

        dparams = StatMR.ev_estim_matlab(tail, self.__dist_name)

        try:
            t = StatMR.ev_quant_matlab(self.__dist_name, self.delta, **dparams)
        except MatlabExecutionError:
            t = np.inf

        return t

    def statistical_mixture(self, data, f, z):

        n_iter = 15
        inc_f = self.__min_tail_sz
        dec_f = self.__min_tail_sz

        _, sidx, eidx = self.get_tail(data, f, z)

        if eidx - sidx == 0:
            if self.v:
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1

        validtail = np.unique(data)[::-1]
        validtail = validtail[validtail != -1]

        a = sidx - inc_f
        if a < 0:
            a = 0

        b = eidx + inc_f
        if b >= validtail.size:
            b = validtail.size

        validtail = validtail[a:b]

        if validtail.size < self.__min_tail_sz:
            if self.v:
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1

        start_rg = np.arange(0, 2*inc_f + 1, dtype=np.int32)
        end_rg = np.arange(validtail.size-2*inc_f, validtail.size+1, dtype=np.int32)

        idx_pairs = np.array(np.meshgrid(start_rg, end_rg)).T.reshape(-1, 2)
        np.random.shuffle(idx_pairs)
        wblgen = []

        it = 0
        while n_iter >= 0 and it < idx_pairs.shape[0]:
            sp, ep = idx_pairs[it]

            # Skips any index pair which do not generate a tail with minimum size
            if ep - sp >= self.__min_tail_sz:

                tail = validtail[sp:ep]

                if np.min(tail) == 0:
                    tail[tail == 0] += 0.00000001

                dparams = StatMR.ev_estim_matlab(tail, self.__dist_name)
                wblgen.append(StatMR.ev_gen_matlab(self.__dist_name, 50, **dparams))
                n_iter -= 1

            it += 1

        wblgen = np.array(wblgen).reshape(-1)
        try:
            if np.min(wblgen) == 0:
                wblgen[np.flatnonzero(wblgen == 0)] += 0.00000001
        except ValueError:
            pdb.set_trace()

        dparams = StatMR.ev_estim_matlab(wblgen, self.__dist_name)
        try:
            t = StatMR.ev_quant_matlab(self.__dist_name, self.delta, **dparams)
        except MatlabExecutionError:
            t = np.inf

        return t

    def __find_best_params(self, X, y, f_range, z_range):

        bestf = f_range[0]
        bestz = z_range[0]

        targetX = X[:, 0:self.k]
        remainderX = X[:, self.k:]

        icount = 0

        np.set_printoptions(linewidth=400)
        if self.v:
            print("  -> Total iterations:", f_range.shape[0]*z_range.shape[0])
        for f in f_range:
            for z in z_range:

                if z <= f:
                    continue

                if self.v:
                    print("    |_ Iteration #{2:d}: f = {0:0.2f} and z = {1:0.2f}".format(f, z, icount),
                          file=sys.stdout, flush=True)

                nsamples = remainderX.shape[0]

                t_values = np.zeros((nsamples, 1), dtype=np.float32)

                for i in tqdm(range(nsamples), total=nsamples, desc="      - Training: ", ncols=100):
                    t = self._method_funct(remainderX[i, :], f, z)
                    t_values[i] = t

                t_values = np.array(t_values, dtype=np.float32).reshape(-1, 1)

                predicted = (targetX >= t_values).astype(np.uint8)
                label = y[:, 1:]

                np.seterr('ignore')
                m = self.__eval_metric(label.reshape(-1), predicted[:, 1:].reshape(-1))
                np.seterr('warn')

                if self.v:
                    print("       -> M = {0:0.3f}  |  Best = {1:0.3f}".format(m, self._opt_val), end="",
                          file=sys.stdout, flush=True)

                if m >= self._opt_val:
                    if self.v:
                        print("    -- F: {0:0.2f} -> {1:0.2f} | Z: {2:0.2f} -> {3:0.2f}".format(bestf, f, bestz, z),
                              file=sys.stdout, flush=True)
                    bestf = f
                    bestz = z
                    self._opt_val = m

                else:
                    print()
                icount += 1

        self._F = bestf
        self._Z = bestz

        return

    @staticmethod
    def ev_quant_matlab(dist_type, p, scale, shape, loc):

        if dist_type == 'WBL':
            t = StatMR.__matlab_engine.wblinv(p, scale, shape)
            sys.stderr.flush()

            return t

        elif dist_type == 'GEV':
            t = StatMR.__matlab_engine.gevinv(p, shape, scale, loc)
            sys.stderr.flush()

            return t

        else:
            raise ValueError('Unsupported distribution <{0:s}>'.format(dist_type))

    @staticmethod
    def ev_gen_matlab(dist_type, n, scale, shape, loc):

        if dist_type == 'WBL':
            samples = StatMR.__matlab_engine.wblrnd(scale, shape, n, 1)
            sys.stderr.flush()

            return samples

        elif dist_type == 'GEV':
            samples = StatMR.__matlab_engine.gevrnd(shape, scale, loc, n, 1)
            sys.stderr.flush()

            return samples

        else:
            raise ValueError('Unsupported distribution <{0:s}>'.format(dist_type))

    @staticmethod
    def ev_estim_matlab(data, dist_type):

        data_m = matlab.double(data.reshape(-1).tolist())

        dparams = dict()

        if dist_type == 'WBL':
            estpar, _ = StatMR.__matlab_engine.wblfit(data_m, nargout=2)
            sys.stderr.flush()

            dparams['scale'] = estpar[0][0]
            dparams['shape'] = estpar[0][1]
            dparams['loc'] = 0

        elif dist_type == 'GEV':
            #pdb.set_trace()
            estpar, _ = StatMR.__matlab_engine.gevfit(data_m, [], StatMR.__gev_opt, nargout=2)
            sys.stderr.flush()
            sys.stdout.flush()

            dparams['scale'] = estpar[0][1]
            dparams['shape'] = estpar[0][0]
            dparams['loc'] = estpar[0][2]


        return dparams






""" Placeholder """
def weibull_fixed_tail(scores, k, notop=True):

    if notop:
        i = random.randint(1, k)
    else:
        i = random.randin(0, k-1)

    return scores[i]

""" Placeholder """
def weibull_mixt_tail(scores, k, notop=True):

    if notop:
        i = random.randint(1, k)
    else:
        i = random.randin(0, k-1)

    return scores[i]