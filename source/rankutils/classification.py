#/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

from pystruct.learners import NSlackSSVM

np.random.seed(93311)

classifier_map = dict(log=LogisticRegression,
                      svc=SVC,
                      rfor=RandomForestClassifier)

grid = [
  #{'C': [1, 10], 'kernel': ['linear']},
  {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}
 ]


def fpe_err_handler(type, flag):
    print("     * Floating point error {0:s}, with flag {1:d}".format(type, flag))


def get_classifier(cname):

    cclass = classifier_map.get(cname, None)

    try:
        clf = cclass(class_weight='balanced', probability=False, kernel='rbf', gamma='scale')

    except TypeError as tpe:
        print("Failure initializing classifier <{0:s}>.".format(cname))
        print(type(tpe).__name__, tpe.args)
        print("-----")
        clf = None

    return clf


def grid_search(estimator, data_x, data_y):

    #print("\n----BEGIN----\n")
    shf = StratifiedShuffleSplit(1, 0.2, 0.8, random_state=93311)

    gs = GridSearchCV(estimator, param_grid=grid, scoring='balanced_accuracy', cv=shf, refit=True, verbose=0, n_jobs=7)
    gs.fit(data_x, data_y)

    #print("\n----END----\n")
    return gs.best_estimator_


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


# Considers sets of non-variate features/labels (2 dimensions)
def run_positional_classification(features, labels, foldidx, cname, get_valid_acc, get_prob):

    get_prob = True

    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    v_acc = []

    train_idx, test_idx = foldidx

    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(features[train_idx])
    TEST_X = SCL.transform(features[test_idx])

    TRAIN_y = labels[train_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if get_valid_acc:
        v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

    if np.unique(TRAIN_y).size == 1:
        one_class = True
        CLF = OneClassSVM(gamma='scale')
        CLF.fit(TRAIN_X, TRAIN_y)
    else:
        one_class = False
        CLF = get_classifier(cname)
        #pdb.set_trace()

        if TRAIN_X.shape[0] >= 100 and (TRAIN_y == 0).sum() > 2 and (TRAIN_y == 1).sum() > 2:
            CLF = grid_search(CLF, TRAIN_X, TRAIN_y)
        else:
            CLF.fit(TRAIN_X, TRAIN_y)

    if not one_class:                             # Not one class classifier and output is labels
        PRED_y = CLF.predict(TEST_X)
        try:
            PROB_y = CLF.predict_proba(TEST_X)[:, 1]
        except AttributeError:
            PROB_y = PRED_y.copy()
    else:                                         # One class classifier and output is labels
        PRED_y = CLF.predict(TEST_X)
        inl_idx = np.flatnonzero(PRED_y == 1)
        outl_idx = np.flatnonzero(PRED_y == -1)
        if TRAIN_y[0] == 0:
            PRED_y[inl_idx] = 0
            PRED_y[outl_idx] = 1
        else:
            PRED_y[inl_idx] = 1
            PRED_y[outl_idx] = 0
        PROB_y = PRED_y.copy()

    if get_valid_acc:
        return PRED_y, PROB_y, v_acc
    else:
        return PRED_y, PROB_y

# Considers sets of non-variate but multipositional features/labels
# Features are k x n x d dimensional
# Labels are k x n x 1 dimensional
def run_single_classification(features, labels, foldidx, cname, get_valid_acc, get_prob):
    #pdb.set_trace()
    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    k, _, d = features.shape

    v_acc = []

    train_idx, test_idx = foldidx

    CLF = get_classifier(cname)

    TRAIN_X = features[1:, train_idx]

    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(TRAIN_X.reshape(-1, d))
    TRAIN_y = labels[1:, train_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if get_valid_acc:
        v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

    #pdb.set_trace()

    if TRAIN_X.shape[0] > 20000:
        sample_idx = np.random.choice(np.arange(TRAIN_X.shape[0]), 20000, replace=False)
        CLF = grid_search(CLF, TRAIN_X[sample_idx], TRAIN_y[sample_idx])
    else:
        CLF = grid_search(CLF, TRAIN_X, TRAIN_y)

    pred_list = []
    prob_list = []

    for i in range(k):
        TEST_X = SCL.transform(features[i, test_idx])

        pred_list.append(CLF.predict(TEST_X))
        try:
            prob_list.append(CLF.predict_proba(TEST_X)[:, 1])
        except AttributeError:
            prob_list.append(pred_list[-1].copy())

    if get_valid_acc:
        return pred_list, prob_list, v_acc
    else:
        return pred_list, prob_list


def run_block_classification(features, labels, foldidx, cname, bs, be, get_valid_acc):
    #pdb.set_trace()
    np.set_printoptions(precision=2, linewidth=300)
    np.seterr(all='call')
    np.seterrcall(fpe_err_handler)

    if bs == 0:
        bs_ = 1
    else:
        bs_ = bs

    k, _, d = features.shape

    v_acc = []

    train_idx, test_idx = foldidx

    CLF = get_classifier(cname)

    TRAIN_X = features[bs_:be, train_idx]

    SCL = StandardScaler()
    TRAIN_X = SCL.fit_transform(TRAIN_X.reshape(-1, d))
    TRAIN_y = labels[bs_:be, train_idx].reshape(-1)

    assert TRAIN_X.shape[0] == TRAIN_y.shape[0], \
           "Inconsistent shape of features {0:s} and labels {1:s}".format(str(TRAIN_X.shape), str(TRAIN_y.shape))

    if get_valid_acc:
        v_acc.append(validation_acc(TRAIN_X, TRAIN_y, cname))

    CLF = grid_search(CLF, TRAIN_X, TRAIN_y)
    #CLF.fit(TRAIN_X, TRAIN_y)

    pred_list = []
    prob_list = []

    for i in range(bs, be):
        if i >= k:
            break

        TEST_X = SCL.transform(features[i, test_idx])

        pred_list.append(CLF.predict(TEST_X))
        try:
            prob_list.append(CLF.predict_proba(TEST_X)[:, 1])
        except AttributeError:
            prob_list.append(pred_list[-1].copy())

    if get_valid_acc:
        return pred_list, prob_list, v_acc
    else:
        return pred_list, prob_list


def run_strc_classification(sequences, labels, foldidx, seq_size, cname, mname):

    from pystruct.models import ChainCRF, MultiLabelClf
    from pystruct.learners import NSlackSSVM, OneSlackSSVM, FrankWolfeSSVM, StructuredPerceptron

    #import ipdb as pdb

    n, k, d = sequences.shape

    train_idx, test_idx = foldidx
    SCL = StandardScaler()

    TRAIN_X = sequences[train_idx]
    TRAIN_y = labels[train_idx]

    TRAIN_X = SCL.fit_transform(TRAIN_X.reshape(-1, d))
    TRAIN_X = TRAIN_X.reshape(-1, seq_size, d)

    TRAIN_y = TRAIN_y.reshape(-1, seq_size)

    if mname == 'crf':
        model = ChainCRF(directed=False)

    if cname == '1slack':
        sclf = OneSlackSSVM(model=model, C=1, max_iter=1500, verbose=0, n_jobs=4)
    elif cname == 'nslack':
        sclf = NSlackSSVM(model=model, max_iter=250, verbose=0)
    elif cname == 'sperc_2':
        sclf = StructuredPerceptron(model=model, max_iter=100)

    #pdb.set_trace()
    sclf.fit(TRAIN_X, TRAIN_y)

    TEST_X = sequences[test_idx]
    test_size = TEST_X.shape[0]
    TEST_y = labels[train_idx]

    TEST_X = SCL.transform(TEST_X.reshape(-1, d))
    TEST_X = TEST_X.reshape(-1, seq_size, d)

    PRED_y = sclf.predict(TEST_X)
    PRED_y = np.array(PRED_y)
    PRED_y = PRED_y.reshape(test_size, k)

    return PRED_y, PRED_y


def run_sequence_labeling(sequences, labels, foldidx, seq_size, cname):

    from seqlearn.perceptron import StructuredPerceptron
    #from seqlearn.hmm import MultinomialHMM

    #import ipdb as pdb

    n, k, d = sequences.shape

    train_idx, test_idx = foldidx
    SCL = StandardScaler()

    TRAIN_X = sequences[train_idx]
    TRAIN_y = labels[train_idx]

    TRAIN_X = SCL.fit_transform(TRAIN_X.reshape(-1, d))
    TRAIN_y = TRAIN_y.reshape(-1)
    lengths_train = np.ones(int(TRAIN_X.shape[0]/seq_size), dtype=np.uint8)*seq_size

    if cname == "sperc":
        clf = StructuredPerceptron(verbose=0, lr_exponent=1.0, max_iter=1000)
    if cname == "hmm":
        #clf = MultinomialHMM()
        raise TypeError("hmm unsupported")
    clf.fit(TRAIN_X, TRAIN_y, lengths_train)

    TEST_X = sequences[test_idx]

    TEST_X = SCL.transform(TEST_X.reshape(-1, d))
    lengths_test = np.ones(int(TEST_X.shape[0]/seq_size), dtype=np.uint8)*seq_size

    PRED_y = clf.predict(TEST_X, lengths=lengths_test)

    PRED_y = PRED_y.reshape(test_idx.size, k)

    return PRED_y, PRED_y



