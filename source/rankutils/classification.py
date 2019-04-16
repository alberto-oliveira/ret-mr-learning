#/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

classifier_map = dict(log=LogisticRegression,
                      linearsvm=LinearSVC,
                      rbfsvm=SVC,
                      rfor=RandomForestClassifier)

import ipdb as pdb


def fpe_err_handler(type, flag):
    print("     * Floating point error {0:s}, with flag {1:d}".format(type, flag))


def get_classifier(cname):

    cclass = classifier_map.get(cname, None)

    try:
        clf = cclass(class_weight='balanced', gamma='scale')

    except TypeError as tpe:
        print("Failure initializing classifier <{0:s}>.".format(cname))
        print(type(tpe).__name__, tpe.args)
        print("-----")
        clf = None

    return clf


def get_training_from_pack(features_pack, train_idx, scale=True):

    if not isinstance(features_pack, np.lib.npyio.NpzFile):
        raise ValueError("<features_pack> argument should be NpzFile")

    all_feat = []
    train_feat = []
    for k in features_pack.keys():
        pos_feat = features_pack[k]
        all_feat.append(pos_feat)
        train_feat.append(pos_feat[train_idx])

    all_feat = np.vstack(all_feat)
    train_feat = np.vstack(train_feat)

    if scale:
        SCL = MinMaxScaler((-1, 1))
        SCL.fit(all_feat)
        train_feat = SCL.transform(train_feat)
        return train_feat, SCL

    return train_feat, None


def validation_acc(features, labels, cname):

    import warnings

    CLF = get_classifier(cname)

    skf = StratifiedKFold(n_splits=3, random_state=93311)

    valid_acc = np.zeros(3)
    i = 0

    for train_idx, test_idx in skf.split(features, labels):

        TRAIN_X = features[train_idx, :]
        TRAIN_y = labels[train_idx]

        TEST_X = features[test_idx, :]
        TEST_y = labels[test_idx]

        CLF.fit(TRAIN_X, TRAIN_y)
        PRED_y = CLF.predict(TEST_X)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            valid_acc[i] = balanced_accuracy_score(TEST_y, PRED_y)

        i += 1

    return valid_acc


def run_two_set_classification(features, labels, foldidx, cname, get_valid_acc=False):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    v_acc = []

    f0_idx, f1_idx = foldidx
    n0 = f0_idx.size
    n1 = f1_idx.size

    N = features.shape[0]
    V = features.shape[1]

    predicted = []

    CLF = get_classifier(cname)

    # Train is fold 1 and Test is fold 0
    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(features[f1_idx, :].reshape(n1*V, -1))
    TEST_X = SCL.transform(features[f0_idx, 0])

    TRAIN_y = labels[f1_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if np.unique(TRAIN_y).size == 1 and not isinstance(CLF, RandomForestClassifier):

        CLF_one = OneClassSVM(gamma='scale')
        CLF_one.fit(TRAIN_X)

        PRED_y = CLF_one.predict(TEST_X)

        inl_idx = np.flatnonzero(PRED_y == 1)
        outl_idx = np.flatnonzero(PRED_y == -1)
        if TRAIN_y[0] == 0:
            PRED_y[inl_idx] = 0
            PRED_y[outl_idx] = 1
        else:
            PRED_y[inl_idx] = 1
            PRED_y[outl_idx] = 0

        predicted.append(PRED_y.reshape((-1, 1)))

    else:

        if get_valid_acc:
            v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

        CLF.fit(TRAIN_X, TRAIN_y)

        PRED_y = CLF.predict(TEST_X)
        predicted.append(PRED_y.reshape((-1, 1)))


    # Train is fold 0 and Test is fold 1
    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(features[f0_idx, :].reshape(n0*V, -1))
    TEST_X = SCL.transform(features[f1_idx, 0])

    TRAIN_y = labels[f0_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if np.unique(TRAIN_y).size == 1 and not isinstance(CLF, RandomForestClassifier):

        CLF_one = OneClassSVM(gamma='scale')
        CLF_one.fit(TRAIN_X)

        PRED_y = CLF_one.predict(TEST_X)

        inl_idx = np.flatnonzero(PRED_y == 1)
        outl_idx = np.flatnonzero(PRED_y == -1)
        if TRAIN_y[0] == 0:
            PRED_y[inl_idx] = 0
            PRED_y[outl_idx] = 1
        else:
            PRED_y[inl_idx] = 1
            PRED_y[outl_idx] = 0

        predicted.append(PRED_y.reshape((-1, 1)))

    else:

        if get_valid_acc:
            v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

        CLF.fit(TRAIN_X, TRAIN_y)

        PRED_y = CLF.predict(TEST_X)
        predicted.append(PRED_y.reshape((-1, 1)))


    # Predicted is a tuple of predicted labels: the first are the labels for fold 0 as test,
    # while the second are the labels for fold 1 as test.
    if get_valid_acc:
        return predicted, v_acc
    else:
        return predicted


# Considers sets of non-variate features/labels (2 dimensions)
def run_classification(features, labels, foldidx, cname, get_valid_acc):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    v_acc = []

    train_idx, test_idx = foldidx

    CLF = get_classifier(cname)

    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(features[train_idx])
    TEST_X = SCL.transform(features[test_idx])

    TRAIN_y = labels[train_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if get_valid_acc:
        v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

    CLF.fit(TRAIN_X, TRAIN_y)

    PRED_y = CLF.predict(TEST_X)

    if get_valid_acc:
        return PRED_y, v_acc
    else:
        return PRED_y
