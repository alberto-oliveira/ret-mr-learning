#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import random
import warnings

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

import matlab.engine
from matlab.engine import MatlabExecutionError

from time import perf_counter

import ipdb as pdb

np.random.seed(93311)

def unwrap_self(arg, **kwarg):
    return WeibullMR_M.weibull_fixed(*arg, **kwarg)

class WeibullMR_M(BaseEstimator):

    __opt_metrics_map = dict(ACC=accuracy_score, F1=f1_score, MCC=matthews_corrcoef)
    __matlab_engine = matlab.engine.start_matlab()

    def __init__(self, k=10, delta=0.99999999, opt_metric='MCC', method='fixed', notop=True, verbose=False):

        self.k = k
        self.delta = delta
        self.opt_metric = opt_metric
        self.notop = notop
        self.v = verbose

        self._F = 0.0
        self._Z = 0.0

        self._opt_val = -np.inf

        self.__min_tail_sz = 4

        self.__eval_metric = WeibullMR_M.__opt_metrics_map.get(self.opt_metric, None)

        if method == 'fixed':
            self.__method_name = method
            self._wmethod = self.weibull_fixed
        elif method == 'mixt':
            self.__method_name = method
            self._wmethod = self.weibull_mixture
            #self.__wmethod = self.weibull_fixed
        else:
            raise TypeError('Undefined weibull method <{0:s}> - options are: ', set(['fixed', 'mixt']))

    @property
    def opt_val(self):
        return self._opt_val

    @property
    def wmethod(self):
        return self.__method_name

    @wmethod.setter
    def wmethod(self, method):
        if method == 'fixed':
            self.__method_name = method
            self._wmethod = self.weibull_fixed
        elif method == 'mixt':
            self.__method_name = method
            self._wmethod = self.weibull_mixture
        else:
            raise TypeError('Undefined weibull method <{0:s}> - options are: '.format(method), set(['fixed', 'mixt']))

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
        distribution to its tail, which is then used to derive the cutoff score according to the value of delta.
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

        if type(f_val) == tuple:
            f_range = np.arange(*f_val)
        elif type(f_val) == float:
            f_range = np.array([f_val])
            self._F = f_val
        else:
            raise TypeError('f_val should either be a float or a tuple.')

        if type(z_val) == tuple:
            z_range = np.arange(*z_val)
        elif type(z_val) == float:
            z_range = np.array([z_val])
            self._Z = z_val
        else:
            raise TypeError('z_val should either be a float or a tuple.')

        if self._F == 0.0 or self._Z == 0.0:

            if X.size == 0 or y.size == 0:
                raise ValueError("Neither X or y can be empty when fitting.")

            if self.v: print('--- Fitting for f in', f_range, 'and z in', z_range, file=sys.stdout, flush=True)
            self.__find_best_params(X, y, f_range, z_range)

        return

    def predict(self, X):

        if self.notop:
            targetX = X[:, 1:self.k + 1]
            tailX = X[:, self.k + 1:]
        else:
            targetX = X[:, 0:self.k]
            tailX = X[:, self.k:]

        #print(targetX)

        t_values = []
        predicted = []
        nsamples = tailX.shape[0]
        for i in range(nsamples):

            tail = tailX[i, :]
            target = targetX[i, :]

            t, _, _ = self._wmethod(tail, self._F, self._Z)
            t_values.append(t)

            p = (target >= t).astype(np.uint8)
            predicted.append(p)

        predicted = np.vstack(predicted)

        return predicted, t_values


    def weibull_fixed(self, tail, f, z):

        #pdb.set_trace()
        tail_u = np.unique(tail)[::-1]
        tail_u = tail_u[tail_u != -1]   # Full tail

        #tail_u = np.fliplr(np.sort(tail[tail != -1].reshape(1, -1))).reshape(-1)

        tail_sz = tail_u.shape[0]       # Full tail size

        if tail_sz < self.__min_tail_sz:
            if self.v:
                print("# of unique scores is less than the minimum tail size of {0:d}".format(self.__min_tail_sz))
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1, 0.0, 0.0

        sidx = int(np.floor(f*tail_sz))                     # starting index of the distribution
        tlen = int(np.floor(z*(tail_u[sidx:].shape[0])))    # tail distribution length

        eidx = sidx + tlen                                  # ending index of the distribution

        # While the tail distribution does not hit minimum size, increase its size by one on each end that can be
        # increased. If no minimum tail size is hit, returns t = -1 as a dummy prediction that no top score is
        # relevant.
        while tlen < self.__min_tail_sz and (sidx > 0 or eidx < tail_sz):
            if sidx-1 >= 0:
                sidx -= 1
                tlen += 1
            if eidx+1 <= tail_sz:
                eidx += 1
                tlen += 1

        if tlen < self.__min_tail_sz:
            if self.v:
                print("Could not extend the tail to reach minimum tail size of {0:d}".format(self.__min_tail_sz))
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1, 0.0, 0.0

        tail_dist = tail_u[sidx:eidx]

        tail_dist[tail_dist == 0] += 0.00001

        tws = perf_counter()
        scl, shp = WeibullMR_M.weibull_estim_matlab(tail_dist)
        twe = perf_counter()

        tqs = perf_counter()
        try:
            t = WeibullMR_M.weibull_quant_matlab(scl, shp, self.delta)
        except MatlabExecutionError:
            t = np.inf
        tqe = perf_counter()

        #if self.v:
            #print("     -> Tail Distribution Size:", tail_dist.shape,
                  #"| Time: ({0:0.3f} + {1:0.3f})s".format((twe-tws), (tqe-tqs)))

        return t, (twe-tws), (tqe-tqs)

    def weibull_mixture(self, tail, f, z):

        n_iter = 30
        inc_f = self.__min_tail_sz
        dec_f = self.__min_tail_sz

        #pdb.set_trace()
        tail_u = np.unique(tail)[::-1]
        tail_u = tail_u[tail_u != -1]   # Full tail
        #tail_u = tail[tail != -1]
        tail_sz = tail_u.shape[0]       # Full tail size

        if tail_sz < self.__min_tail_sz:
            if self.v:
                print("# of unique scores is less than the minimum tail size of {0:d}".format(self.__min_tail_sz))
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1, 0.0, 0.0

        sidx = int(np.floor(f*tail_sz))                     # starting index of the distribution
        tlen = int(np.floor(z*(tail_u[sidx:].shape[0])))    # tail distribution length

        eidx = sidx + tlen                                  # ending index of the distribution

        # While the tail distribution does not hit minimum size, increase its size by one on each end that can be
        # increased. If no minimum tail size is hit, returns t = -1 as a dummy prediction that no top score is
        # relevant.
        while tlen < self.__min_tail_sz and (sidx > 0 or eidx < tail_sz):
            if sidx-1 >= 0:
                sidx -= 1
                tlen += 1
            if eidx+1 <= tail_sz:
                eidx += 1
                tlen += 1

        if tlen < self.__min_tail_sz:
            if self.v:
                print("Could not extend the tail to reach minimum tail size of {0:d}".format(self.__min_tail_sz))
                print("  -> Setting t = -1: All predictions non-relevant. ")
            return -1, 0.0, 0.0


        # The start of the tail increases by subtracting from sidx (starting index) and decreases by adding to sidx.
        # The maximum increase is inc_f, and maximum decrease is dec_f.
        a = sidx-inc_f
        if a < 0:
            a = 0

        b = sidx+dec_f
        if b > tail_sz:
            b = tail_sz

        start_rg = np.arange(a, b+1, dtype=np.int32)


        # The end of the tail increases by adding to eidx (ending index) and decreases by subtracting from eidx.
        # The maximum increase is inc_f. The maximum decrease is dec_f.
        a = eidx-dec_f
        if a < 0:
            a = 0

        b = eidx+inc_f
        if b > tail_sz:
            b = tail_sz

        end_rg = np.arange(a, b+1, dtype=np.int32)

        idx_pairs = np.array(np.meshgrid(start_rg, end_rg)).T.reshape(-1, 2)
        np.random.shuffle(idx_pairs)
        wblgen = []

        tws = perf_counter()
        it = 0
        while n_iter >= 0 and it < idx_pairs.shape[0]:
            sp, ep = idx_pairs[it]

            # Skips any index pair which do not generate a tail with minimum size
            if ep - sp >= self.__min_tail_sz:

                tail_dist = tail_u[sp:ep]

                tail_dist[tail_dist == 0] += 0.00001

                scl, shp = WeibullMR_M.weibull_estim_matlab(tail_dist)
                wblgen.append(WeibullMR_M.weibull_gen_matlab(scl, shp, 50))
                n_iter -= 1

            it += 1
        twe = perf_counter()

        scl, shp = WeibullMR_M.weibull_estim_matlab(np.array(wblgen).reshape(-1))
        tqs = perf_counter()
        try:
            t = WeibullMR_M.weibull_quant_matlab(scl, shp, self.delta)
        except MatlabExecutionError:
            t = np.inf
        tqe = perf_counter()

        return t, (twe-tws), (tqe-tqs)

    def __find_best_params(self, X, y, f_range, z_range):

        bestf = f_range[0]
        bestz = z_range[0]

        if self.notop:
            targetX = X[:, 1:self.k + 1]
            tailX = X[:, self.k + 1:]
        else:
            targetX = X[:, 0:self.k]
            tailX = X[:, self.k:]

        icount = 0

        np.set_printoptions(linewidth=400)
        if self.v:
            print("-> Total iterations:", f_range.shape[0]*z_range.shape[0])
        for f in f_range:
            for z in z_range:

                ts = perf_counter()
                if self.v: print("  |_ Iteration #{2:d}: f = {0:0.2f} and z = {1:0.2f}".format(f, z, icount),
                                 file=sys.stdout, flush=True)

                nsamples = tailX.shape[0]

                #print('N CORES: ', ncores)

                #results = Parallel(n_jobs=-1)(delayed(unwrap_self)
                          #(params) for params in zip([self]*nsamples, tailX, [f]*nsamples, [z]*nsamples))


                results = []

                for i in range(nsamples):
                    tail = tailX[i, :]

                    #ts = perf_counter()
                    t, tw, tq = self._wmethod(tail, f, z)
                    #te = perf_counter()
                    #pdb.set_trace()

                    results.append([t, tw, tq])

                results = np.array(results, dtype=np.float32)
                t_values = results[:, 0].reshape(-1, 1)
                time_w = np.sum(results[:, 1])
                time_q = np.sum(results[:, 2])

                predicted = (targetX >= t_values).astype(np.uint8).reshape(-1)

                label = y.reshape(-1)

                np.seterr('ignore')
                m = self.__eval_metric(label, predicted)
                np.seterr('warn')

                te = perf_counter()
                if self.v:
                    print("     -> # samples: {2:d} | Elapsed: {0:0.3f}s | Avg. Wbl. Time {1:0.3f}s |"
                          " Avg. Qtl. Time {3:0.3f}s".format(te-ts, time_w/nsamples, nsamples, time_q/nsamples),
                          file=sys.stdout, flush=True)
                    print("     -> M = {0:0.3f}  |  Best = {1:0.3f}".format(m, self._opt_val), file=sys.stdout, flush=True)

                if m >= self._opt_val:
                    if self.v:
                        print("     -- Updating F: {0:0.2f} -> {1:0.2f}".format(bestf, f), file=sys.stdout, flush=True)
                        print("     -- Updating Z: {0:0.2f} -> {1:0.2f}".format(bestz, z), file=sys.stdout, flush=True)
                    bestf = f
                    bestz = z
                    self._opt_val = m

                if self.v:
                    print("  --\n")
                icount += 1

        self._F = bestf
        self._Z = bestz

        return

    @staticmethod
    def weibull_quant_matlab(scl, shp, p):

        t = WeibullMR_M.__matlab_engine.wblinv(p, scl, shp)
        sys.stderr.flush()

        return t

    @staticmethod
    def weibull_gen_matlab(scl, shp, n):

        wblr = WeibullMR_M.__matlab_engine.wblrnd(scl, shp, n, 1)

        sys.stderr.flush()
        arr = np.array(wblr, dtype=np.float64).reshape(-1)
        return arr

    @staticmethod
    def weibull_estim_matlab(data):

        data_m = matlab.double(data.reshape(-1).tolist())
        #print(data_m)
        estpar, _ = WeibullMR_M.__matlab_engine.wblfit(data_m, nargout=2)
        sys.stderr.flush()

        scale = estpar[0][0]
        shape = estpar[0][1]

        return scale, shape






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