#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import errno
import glob
import time
#import ipdb as pdb

import numpy as np

from rankutils.rIO import read_rank, rk_dtype

completedir = lambda d: d if d[-1] == '/' else d + "/"
getbasename = lambda f: os.path.splitext(os.path.basename(f))[0]


class ProgressBar:

    def __init__(self, n, updt_every, char_length=50, fill='█'):

        self.__n = n
        self.__char_length = char_length
        self.__c = fill
        self.__updt_last = 0.0

        if updt_every <= 0.01:
            self.__updt = 0.01
        elif updt_every > 0.5:
            self.__updt = 0.5
        else:
            self.__updt = updt_every

    @property
    def updt(self):
        return self.__updt

    @updt.setter
    def updt(self, u):
        if u <= 0.01:
            self.__updt = 0.01
        elif u > 0.5:
            self.__updt = 0.5
        else:
            self.__updt = u

    def start(self):

        self.__updt_last = 0.0
        prefix = "Progress |"
        not_filled = '-' * self.__char_length
        suffix = '| {complete:0.1%} Complete'.format(complete=self.__updt_last)
        print("{0:s}{1:s}{2:s}".format(prefix, not_filled, suffix), end='\r')

    def update(self, i):

        at = (i+1)/self.__n

        if at >= 1.0 or (at - self.__updt_last) >= self.__updt:
            nchars = int(at * self.__char_length)

            prefix = "Progress |"
            filled = self.__c * nchars
            not_filled = '-' * (self.__char_length - nchars)
            suffix = '| {complete:0.1%} Complete'.format(complete=at)

            print("{0:s}{1:s}{2:s}{3:s}".format(prefix, filled, not_filled, suffix), end='\r')
            self.__updt_last = at

        if i == self.__n:
            print('\n')


def safe_create_dir(dir):
    """ Safely creates dir, checking if it already exists.

    Creates any parent directory necessary. Raises exception
    if it fails to create dir for any reason other than the
    directory already existing.

    :param dir: of the directory to be created
    """


    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def merge_kfolds_rounds(respath, folds):

    resflist = glob.glob(completedir(respath) + "*.npy")
    resflist.sort()

    round_res = []

    nf = folds.shape[1]

    for j in range(nf):

        fold = folds[:, j]

        zero_res = np.load(resflist[j*2])
        one_res = np.load(resflist[j*2 + 1])

        zero_idx = np.argwhere(fold.reshape(-1) == 0).reshape(-1)
        one_idx = np.argwhere(fold.reshape(-1) == 1).reshape(-1)

        npred = zero_res.shape[1]

        merged_res = np.zeros((zero_res.shape[0] + one_res.shape[0], npred), dtype=np.uint8)
        merged_res[zero_idx] = zero_res
        merged_res[one_idx] = one_res
        round_res.append(merged_res)

    return round_res




def ndarray_bin_to_int(arr):

    if arr.ndim == 2:
        arr2d = arr
    elif arr.ndim == 1:
        arr2d = arr.reshape(1, -1)
    else:
        raise ValueError("Input array should be either 1d or 2d. It is {0:d}d".format(arr.ndim))

    nbi = arr2d.shape[1]

    # If the number of bits is not divisible per 8, it has to pad the array with 0s
    # I manually pad it, so I can pad in the left
    if nbi % 8 != 8:
        p = int(np.ceil(nbi / 8) * 8 - nbi)
        arr2d = np.pad(arr2d, pad_width=((0, 0), (p, 0)), mode='constant', constant_values=0)

    packn = np.packbits(arr2d, axis=1).astype(np.uint32)
    if packn.ndim == 1:
        packn = packn.reshape(1, -1)

    nBy = packn.shape[1]  # number of columns is the number of Bytes

    for x in range(0, nBy-1, 1):
        np.left_shift(packn[:, x:x+1], (nBy-1-x)*8, packn[:, x:x+1])

    return np.sum(packn, axis=1).astype(np.int32)


def makeDlabel(d):
    if d['name'] == 'WBL':
        return '{0:s} : shape={1:0.2f}, scale={2:0.2f}'.format(d['name'], d['shape'], d['scale'])

    elif d['name'] == 'GEV':
        return '{0:s} : shape={1:0.2f}, scale={2:0.2f}, loc={3:0.2f}'.format(d['name'], d['shape'], d['scale'],
                                                                             d['loc'])


def get_classname(name):
    parts = name.split("_")
    i = 0

    for i in range(len(parts)):
        if parts[i].isdigit():
            break

    return "_".join(parts[:i])


def get_query_classname(qname):
    suffix = qname.split("_", 1)[1]
    return get_classname(suffix)


# Generates a modification of a set of binary labels such that f% of the bits are
# shifted to the other state. Bits are randomly chosen, in an uniform manner
def gen_mod(l, f):
    l_ = l.copy().reshape(-1)
    i = np.random.permutation(l.size)
    total = int(np.floor(f * l.size))
    i = i[:total]
    l_[i] = (l_[i] - 1) / 255

    return l_.reshape(l.shape)


def read_and_convert(rkfpath, limit=np.newaxis, scale=False, convert=False):

    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler((1, 2))

    fullrank = read_rank(rkfpath)

    rk = fullrank['score']
    namelist = fullrank['name']

    rk = rk[0:limit]

    if scale:
        rk = mms.fit_transform(rk.reshape(-1, 1)).reshape(-1)

    if convert:
        rk = np.max(rk) - rk

    lowerb = np.min(rk)
    upperb = np.max(rk)

    return rk, namelist, (lowerb, upperb)


def preprocess_ranks(dir, maxsz=1000):

    rkpathlist = glob.glob(dir + "/*.rk")
    rkpathlist.sort()

    rklist = []
    ts = time.perf_counter()
    for fpath in rkpathlist:

        rk = read_rank(fpath, 'score')
        if rk[0] < rk[-1]:
            rk = np.max(rk) - rk

        if rk.shape[0] > maxsz:
            rklist.append(rk)
            maxsz = rk.shape[0]
        else:
            rklist.append(np.pad(rk, (0, maxsz-rk.shape[0]), 'constant', constant_values=-1))

    rkarr = np.vstack(rklist)
    te = time.perf_counter()

    del rklist

    print("Preprocess -- Elapsed: {0:0.3f}s".format(te-ts))

    return rkarr


def get_index(array_a, array_b):
    """
    Get the positions of array_b elements in array_a. Both need to be numpy.ndarrays and have the same size.

    :param array_a: numpy.ndarray where the search is performed.
    :param array_b: numpy.ndarray with the elements to be searched for
    :return:
    """

    found = []

    for val in array_b:
        try:
            found.append(np.flatnonzero(array_a == val)[0])
        except IndexError:
            pass

    return np.array(found, dtype=np.intp)


def get_img_name(imgpath):

    basename = os.path.basename(imgpath)
    parts = basename.split("_", 2)
    imname = parts[2]

    return imname


def name_from_rankfile(rkpath):

    basename = os.path.splitext(os.path.basename(rkpath))[0]
    parts = basename.split("_", 1)
    basename = parts[1]

    return basename
