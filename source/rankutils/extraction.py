#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import numpy as np

from rankutils.features import *

extractor_map = dict(dct=rank_features_kDCT,
                     dct_shift=rank_features_shiftDCT,
                     deltaik=rank_features_deltaik,
                     deltaik_c=rank_features_circ_deltaik)


def extract_rank_features(rank, extraction_list, ci):
    """

    Given a rank, extracts the desired features from it, returning a feature vector.

    :param rank: r-sized numpy array of ranked scores.
    :param extraction_list: dictionary of key:tuple, where key is the name of the feature, and tuple is the parameters
                     of the feature.
    :param ci: position of the rank that is currently being processed.
    :return: n-sized numpy array of features.
    """

    features = []

    for featn, feattp, params in extraction_list:

        extractor = extractor_map.get(featn, None)
        if extractor is not None:
            try:
                if 'i' in params and params['i'] == -1:
                    params['i'] = ci
                    feat = extractor(rank, **params)
                    params['i'] = -1
                else:
                    feat = extractor(rank, **params)
                features.append(feat)
            except TypeError as tpe:
                print("Failure executing extractor <{0:s}> ({1:s}) with parameters ".format(featn, feattp), params)
                print(type(tpe).__name__, tpe.args)
                print("-----")
        else:
            print("Extractor named {0:s} does not exist!".format(featn), "\n-----")

    return features


def create_extraction_list(feat_cfg):

    extraction_list = []
    for sect in feat_cfg.sections():
        if sect.startswith('feat'):

            sect_map = dict()
            extraction_list.append((feat_cfg[sect]['name'],
                                    feat_cfg[sect]['type'],
                                    sect_map))

            for key in feat_cfg[sect]:

                if key == 'i' or key == 'k' or key == "dct_type":
                    try:
                        sect_map[key] = feat_cfg[sect].getint(key)
                    except ValueError as ve:
                        sect_map[key] = -1
                if key == 'notop':
                    sect_map[key] = feat_cfg[sect].getboolean(key)

    return extraction_list


