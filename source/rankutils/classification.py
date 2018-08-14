#/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

classifier_map = dict(log=LogisticRegression,
                      linearsvm=LinearSVC,
                      svm=SVC,
                      rfor=RandomForestClassifier)

def fpe_err_handler(type, flag):
    print("     * Floating point error {0:s}, with flag {1:d}".format(type, flag))

def get_classifier(cname):

    cclass = classifier_map.get(cname, None)

    try:
        clf = cclass(class_weight='balanced')

    except TypeError as tpe:
        print("Failure initializing classifier <{0:s}>.".format(cname))
        print(type(tpe).__name__, tpe.args)
        print("-----")
        clf = None

    return clf

def run_classification(features, labels, cname, scale=False, M=0):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    fold_num = len(labels)
    predicted = []

    features_s = []

    # Scaling between -1 and 1 if desired. Good for SVM.
    if scale:
        SCL = MinMaxScaler((-1, 1))
        aux = np.vstack(features)
        SCL.fit(aux)
        del aux
        for feat in features:
            features_s.append(SCL.transform(feat))
    else:
        features_s = features

    CLF = get_classifier(cname)

    for f in range(fold_num):
        #print("        # Test Fold:", f)

        TRAIN_X = np.vstack(features_s[0:f] + features_s[f+1:])
        TRAIN_y = np.vstack(labels[0:f] + labels[f+1:])  # get only labels refering to M val
        print(TRAIN_y.shape)
        TRAIN_y = TRAIN_y[:, M]

        TEST_X = features_s[f]

        assert TRAIN_X.shape[0] == TRAIN_y.shape[0], "Inconsistent shape of features {0:s} and labels {1:s}"\
               .format(str(TRAIN_X.shape), str(TRAIN_y.shape))

        #try:
        CLF.fit(TRAIN_X, TRAIN_y)
        PRED_y = CLF.predict(TEST_X)

        predicted.append(PRED_y.reshape((-1, 1)))
        #except ValueError as ver:
            #print("        @ M = {0:d} -> Could not FIT <{1:s}> classifier. ".format(M+1, cname))
            #print("          -> Train Labels:", TRAIN_y.reshape(-1).astype(np.uint8))
            #predicted.append(np.ones(labels[f][:, M].shape)*-1)

    return predicted

def run_two_set_classification(features, labels, foldidx, cname, scale=False):

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    f0_idx, f1_idx = foldidx

    predicted = []

    # Scaling between 1 and 2 if desired. Good for SVM.
    if scale:
        SCL = MinMaxScaler((1, 2))
        aux = np.vstack(features)
        SCL.fit(aux)
        features_s = SCL.transform(features)
        del aux
    else:
        features_s = features

    CLF = get_classifier(cname)


    # Train is fold 1 and Test is fold 0
    TRAIN_X = features_s[f1_idx, :]
    TRAIN_y = labels[f1_idx]

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], "Inconsistent shape of features {0:s} and labels {1:s}" \
                                                 .format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    TEST_X = features_s[f0_idx, :]

    CLF.fit(TRAIN_X, TRAIN_y)
    PRED_y = CLF.predict(TEST_X)
    predicted.append(PRED_y.reshape((-1, 1)))


    # Train is fold 0 and Test is fold 1
    TRAIN_X = features_s[f0_idx, :]
    TRAIN_y = labels[f0_idx]

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], "Inconsistent shape of features {0:s} and labels {1:s}" \
                                                 .format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    TEST_X = features_s[f1_idx, :]
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