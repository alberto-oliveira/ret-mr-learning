#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
sys.path.append("../source/")

import numpy as np

import ipdb as pdb

from pystruct.models import ChainCRF, BinaryClf, MultiLabelClf
from pystruct.learners import NSlackSSVM

from sklearn.metrics import balanced_accuracy_score

from rankutils.utilities import reshape_features

fv = np.load(sys.argv[1])
fold = np.load(sys.argv[2])

features, labels = reshape_features(fv)

#pdb.set_trace()

fidx = fold[0, :, 0]

X_train, X_test = features[fidx == 0], features[fidx == 1]
y_train, y_test = labels[fidx == 0], labels[fidx == 1]

model = ChainCRF()
#model = BinaryClf(X_train.shape[0])
model = MultiLabelClf()
ssvm = NSlackSSVM(model=model, max_iter=500)

X_train_l = [row for row in X_train]
y_train_l = [row for row in y_train]

X_test_l = [row for row in X_test]

ssvm.fit(X_train, y_train)

y_pred_l = ssvm.predict(X_test_l)

y_pred = np.array(y_pred_l)
#for pred in y_pred:
#    print(pred)

#print(y_pred)

assert y_pred.shape == y_test.shape

for i in range(y_pred.shape[1]):

    print("bACC [{0:d}] = {1:0.3f}".format(i, balanced_accuracy_score(y_test[:, i], y_pred[:, i])))


