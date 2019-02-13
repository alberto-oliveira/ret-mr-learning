#/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale

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


def run_single_two_set_classification(features_pack, labels_mat, foldidx, cname, scale=True):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    f0_idx, f1_idx = foldidx

    ### FOLD 0 is TESTING. FOLD 1 is TRAINING
    TRAIN_X, scaler = get_training_from_pack(features_pack, f1_idx)
    TRAIN_y = labels_mat[f1_idx].transpose().reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    assert np.unique(TRAIN_y).size > 1, "Only one class in training labels"

    CLF = get_classifier(cname)
    CLF.fit(TRAIN_X, TRAIN_y)

    p0 = []

    for k in features_pack.keys():

        if scaler:
            TEST_X = scaler.transform(features_pack[k][f0_idx])
        else:
            TEST_X = features_pack[k][f0_idx]

        PRED_y = CLF.predict(TEST_X)

        p0.append(PRED_y.reshape((-1, 1)))

    p0 = np.hstack(p0)

    ### FOLD 1 is TESTING. FOLD 0 is TRAINING
    TRAIN_X, scaler = get_training_from_pack(features_pack, f0_idx)
    TRAIN_y = labels_mat[f0_idx].transpose().reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
        "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    assert np.unique(TRAIN_y).size > 1, "Only one class in training labels"

    CLF = get_classifier(cname)
    CLF.fit(TRAIN_X, TRAIN_y)

    p1 = []

    for k in features_pack.keys():

        if scaler:
            TEST_X = scaler.transform(features_pack[k][f1_idx])
        else:
            TEST_X = features_pack[k][f1_idx]

        PRED_y = CLF.predict(TEST_X)

        p1.append(PRED_y.reshape((-1, 1)))

    p1 = np.hstack(p1)

    return p0, p1


def run_two_set_classification(features, labels, foldidx, cname, scale=True):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

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

        CLF.fit(TRAIN_X, TRAIN_y)

        PRED_y = CLF.predict(TEST_X)
        predicted.append(PRED_y.reshape((-1, 1)))


    # Predicted is a tuple of predicted labels: the first are the labels for fold 0 as test,
    # while the second are the labels for fold 1 as test.
    return predicted

"""
def run_two_set_classification(features, labels, foldidx, cname, scale=True):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    f0_idx, f1_idx = foldidx

    predicted = []

    # Scaling between -1 and 1 if desired. Good for SVM.
    if scale:
        SCL = MinMaxScaler((-1, 1))
        aux = np.vstack(features)
        SCL.fit(aux)
        features_s = SCL.transform(features)
        del aux
    else:
        features_s = features

    CLF = get_classifier(cname)


    # Train is fold 1 and Test is fold 0
    TRAIN_X = features_s[f1_idx, :]
    TRAIN_y = labels[f1_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], "Inconsistent shape of features {0:s} and labels {1:s}" \
                                                 .format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    TEST_X = features_s[f0_idx, :]

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

        CLF.fit(TRAIN_X, TRAIN_y)

        PRED_y = CLF.predict(TEST_X)
        predicted.append(PRED_y.reshape((-1, 1)))


    # Train is fold 0 and Test is fold 1
    TRAIN_X = features_s[f0_idx, :]
    TRAIN_y = labels[f0_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], "Inconsistent shape of features {0:s} and labels {1:s}" \
                                                 .format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    TEST_X = features_s[f1_idx, :]

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

        CLF.fit(TRAIN_X, TRAIN_y)

        PRED_y = CLF.predict(TEST_X)
        predicted.append(PRED_y.reshape((-1, 1)))


    # Predicted is a tuple of predicted labels: the first are the labels for fold 0 as test,
    # while the second are the labels for fold 1 as test.
    return predicted


def load_features(indir, key):
    features = []
    featfiles = glob.glob(indir + "*.npz")
    featfiles.sort()
    for ff in featfiles:
        with np.load(ff) as foldfeat:
            features.append(foldfeat[key])
    return features

def load_labels(indir):
    labels =[]
    lblfiles = glob.glob(indir + "*.npy")
    lblfiles.sort()
    for lf in lblfiles:
        labels.append(np.load(lf))

    return labels
"""