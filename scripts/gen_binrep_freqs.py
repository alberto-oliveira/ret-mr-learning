#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import numpy as np

sys.path.append("../source")
from rankutils.utilities import ndarray_bin_to_int


def get_print_sumfreqs(labels, k, mincount=0):

    m = 0
    l = 0

    labels_ = labels[:, 0:k]
    labels_sum = np.sum(labels_, axis=1, dtype=np.int32)
    count = np.bincount(labels_sum)
    ars = np.argsort(count)[::-1]

    total = labels.shape[0]

    for v in ars:
        if count[v] == 0:
            break

        print("{0:02d} : {1:05d} : {2:03.2%}".format(v, count[v], float(count[v])/total))
        if count[v] >= mincount:
            m += count[v]
        else:
            l += count[v]

    return m, l

def get_print_freqs(labels, k, sw='', mincount=0, top=np.newaxis):

    m = 0
    l = 0

    labels_ = labels[:, 0:k]
    labelsint = ndarray_bin_to_int(labels_)
    count = np.bincount(labelsint)
    ars = np.argsort(count)[::-1]

    total = labels.shape[0]

    for v in ars[:top]:
        if count[v] == 0:
            break

        binrep = np.binary_repr(v, width=k)
        if sw != '':
            if binrep.startswith(sw):
                print("{0:s} : {1:05d} : {2:03.2%}".format(binrep, count[v], float(count[v])/total))
                if count[v] >= mincount:
                    m += 1
                else:
                    l += 1
        else:
            print("{0:s} : {1:05d} : {2:03.2%}".format(binrep, count[v], float(count[v])/total))
            if count[v] >= mincount:
                m += count[v]
            else:
                l += count[v]

    return m, l


if __name__ == "__main__":
    labels = np.load(sys.argv[1])
    k = int(sys.argv[2])

    outfname = os.path.basename(sys.argv[1]).rsplit('.', 1)[0] + ".freqs"

    labels_ = labels[:, 0:k]
    labelsint = ndarray_bin_to_int(labels_)
    count = np.bincount(labelsint)
    ars = np.argsort(count)[::-1]

    with open(outfname, 'w') as outf:

        for v in ars:
            if count[v] == 0:
                break

            outf.write("{0:s}:{1:d}\n".format(np.binary_repr(v, width=k), count[v]))
