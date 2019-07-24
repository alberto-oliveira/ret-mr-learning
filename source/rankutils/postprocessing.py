#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import cv2

from collections import OrderedDict

from rankutils.utilities import ndarray_bin_to_int
#import ipdb as pdb


### METHOD USING INTEGER FORMS OF LABEL ARRAYS
def labeling_frequencies_(labels, freq=False):

    int_labels = ndarray_bin_to_int(labels)

    cts = np.bincount(int_labels)
    if freq:
        cts = cts/np.sum(cts)

    nzi = np.flatnonzero(cts)

    return nzi, cts[nzi]


def label_frequency_mod_(predicted_labels, train_labels, k=3, mod_f=1.5):

    #pdb.set_trace()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    mod_predicted_labels = predicted_labels.copy()

    idx, freqs = labeling_frequencies(train_labels, True)
    unique_labels = np.array([list(np.binary_repr(x, width=train_labels.shape[1])) for x in idx], dtype=np.uint8)

    matches = bf_matcher.knnMatch(predicted_labels, unique_labels, k=k)

    for i, label_matches in enumerate(matches):

        try:
            # The label has a nonzero frequency in the training labels
            if label_matches[0].distance == 0:

                lfreq = freqs[label_matches[0].trainIdx]

                midx = [m.trainIdx for m in label_matches[1:]]
                dists = [m.distance for m in label_matches[1:]]

                mfreqs = freqs[midx]

                p = np.argmax(mfreqs)
                maxfreq = mfreqs[p]
                maxidx = midx[p]
                maxdist = dists[p]

                if maxfreq/lfreq >= mod_f and maxdist <= train_labels.shape[1] // 2:
                    mod_predicted_labels[i] = unique_labels[maxidx]

            else:

                midx = [m.trainIdx for m in label_matches]
                dists = [m.distance for m in label_matches]
                mfreqs = freqs[midx]

                p = np.argmax(mfreqs)
                maxidx = midx[p]
                maxdist = dists[p]

                if maxdist <= train_labels.shape[1] // 2:
                    mod_predicted_labels[i] = unique_labels[maxidx]
        except IndexError:
            print("at ", i)

    return mod_predicted_labels

#######

def labeling_frequencies(labels, proportion=False):

    frequencies = OrderedDict()
    unique = []
    total = labels.shape[0]

    for l in labels:

        if l.tostring() not in frequencies:
            frequencies[l.tostring()] = 1
            unique.append(l)
        else:
            frequencies[l.tostring()] += 1

    if proportion:
        for l in frequencies:
            frequencies[l] = frequencies[l]/total

    return frequencies, np.vstack(unique)


def label_frequency_mod(predicted_labels, train_labels, k=2, mod_f=1.5):

    #pdb.set_trace()

    assert predicted_labels.shape[1] == train_labels.shape[1], "Non-consistent shapes between predicted and train " \
                                                               "labels"
    bitn = train_labels.shape[1]
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    mod_predicted_labels = predicted_labels.copy()

    freqs, unique_labels = labeling_frequencies(train_labels, True)

    matches = bf_matcher.knnMatch(predicted_labels, unique_labels, k=k)

    for i, label_matches in enumerate(matches):

        try:
            plabel = predicted_labels[i]

            if plabel.tostring() in freqs:
                pfreq = freqs[plabel.tostring()]

                mlabel = unique_labels[label_matches[1].trainIdx]  # index [1] is used because [0] is the query label
                mfreq = freqs[mlabel.tostring()]

                if label_matches[1].distance <= (bitn // 2) and (mfreq / pfreq) >= mod_f:
                    mod_predicted_labels[i] = mlabel
                    #print("{0:04d} {1:s} -> {2:s} : {3:0.1f}".format(i, str(plabel), str(mlabel), label_matches[1].distance))

            else:
                mlabel = unique_labels[label_matches[0].trainIdx]

                mod_predicted_labels[i] = mlabel
                #rint("{0:04d} {1:s} -> {2:s} : {3:0.1f}".format(i, str(plabel), str(mlabel), label_matches[0].distance))

        except IndexError:
            print("at ", i)

    return mod_predicted_labels






