#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import traceback
import errno
import glob
import time

import ipdb as pdb

import numpy as np

from rankutils.rIO import read_rank, rk_dtype

completedir = lambda d: d if d[-1] == '/' else d + "/"
getbasename = lambda f: os.path.splitext(os.path.basename(f))[0]

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

def check_experiment(expdir):
    """
    Checks if determined experiment output directories exist.

    :param expdir: experiment directory in /output
    :return: 0 if neither relevance prediction or performance prediction exists, 1 if only relevance prediction exists,
             2 if only performance prediction exists, and 3 if both exists.
    """

    expdir = completedir(expdir)
    check = 0

    rpcheck = os.path.isdir(isexpdir + "rel-prediction")
    ppcheck = os.path.isdir(isexpdir + "perf-prediction")

    if rpcheck and ppcheck:
        check = 3
    elif rpcheck:
        check = 1
    elif ppcheck:
        check = 2
    else:
        check = 0


    return check


def get_rank_colname(rank):

    if np.all(rank['score'] == 1):
        colname = 'normd'
    else:
        colname = 'score'

    return colname

def preprocess_ranks(dir, colname='votes', maxsz=1000):

    if colname not in rk_dtype['names']:
        raise ValueError('attribute \'colname\' must be one of:', rk_dtype['names'])

    rkpathlist = glob.glob(dir + "/*.rk")
    rkpathlist.sort()

    rklist = []
    ts = time.perf_counter()
    for fpath in rkpathlist:

        rk = read_rank(fpath, colname)

        if rk.shape[0] > maxsz:
            rklist.append(rk)
            maxsz = rk.shape[0]
        else:
            rklist.append(np.pad(rk, (0, maxsz-rk.shape[0]), 'constant', constant_values=-1))

    rkarr = np.vstack(rklist)
    te = time.perf_counter()

    del rklist

    print("Elapsed: {0:0.3f}s".format(te-ts))

    return rkarr


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
