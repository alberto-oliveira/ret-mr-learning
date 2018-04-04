#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
#np.random.seed(93311)

from sklearn.metrics.pairwise import cosine_distances

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import matlab.engine

import ipdb as pdb

mlab = matlab.engine.start_matlab()
entropy = importr("entropy")

def compute_kl_div(dists_a, dists_b):

    divs = np.zeros((dists_a.shape[0], dists_b.shape[0]), dtype=np.float32)
    for i, da in enumerate(dists_a):
        da_r = robjects.FloatVector(da.reshape(-1).tolist())

        for j, db in enumerate(dists_b):
            db_r = robjects.FloatVector(db.reshape(-1).tolist())
            kld = entropy.KL_plugin(da_r, db_r)

            divs[i, j] = kld[0]

    return divs


def gen_wbl_hists(wbl_par, nsamples):

    aux = []
    for par in wbl_par:
        wblr = mlab.wblrnd(float(par[0]), float(par[1]), nsamples, 1)
        hist, _ = np.histogram(np.array(wblr, dtype=np.float64).reshape(1, -1), bins=50, density=True)
        aux.append(hist)

    arr = np.vstack(aux)
    sys.stderr.flush()
    return arr

def gen_wbl_pdf(wbl_par, vals):

    aux = []
    for par in wbl_par:
        wblpdf = mlab.wblpdf(matlab.double(vals.tolist()), float(par[0]), float(par[1]))
        aux.append(np.array(wblpdf, dtype=np.float64).reshape(1, -1))

    arr = np.vstack(aux)
    sys.stderr.flush()
    return arr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number", help="# of comparative examples. Default is 100.", default=100, type=int)
    parser.add_argument("-s", "--shape", help="Shape of base distribution.", default=3.0, type=float)
    parser.add_argument("-c", "--scale", help="Scale of base distribution.", default=2.0, type=float)
    parser.add_argument("-p", "--plot", help="Plot distributions.", action="store_true")

    args = parser.parse_args()

    scl = args.scale
    shp = args.shape
    p = args.plot
    rsize = 10000

    #b_par = 5*np.random.random_sample((1, 2))
    b_par = np.array([[scl, shp]])
    b_hists = gen_wbl_hists(b_par, rsize)
    b_pdf = gen_wbl_pdf(b_par, np.arange(0, 1000, 0.1))

    c_par = 10*np.random.random_sample((args.number, 2))
    #aux_shp = 10*np.random.random_sample((args.number, 2))
    #c_par = np.hstack([aux_scl.reshape(-1, 1), aux_shp.reshape(-1, 1)])
    #print(c_par.shape)
    c_hists = gen_wbl_hists(c_par, rsize)
    c_pdf = gen_wbl_pdf(c_par, np.arange(0, 1000, 0.1))

    cos_dists = cosine_distances(b_hists, c_hists).reshape(-1)
    kl_divs = compute_kl_div(b_pdf, c_pdf).reshape(-1)

    order_cos = np.argsort(cos_dists)
    order_kl = np.argsort(kl_divs)

    print("scale: {0:05.3f} and shape: {1:05.3f})\n".format(b_par[0, 0], b_par[0, 1]))
    print(",,Cosine,,,,KL-Div")
    print(",pos,value,scale,shape,,pos,value,scale,shape")
    for i in range(c_par.shape[0]):
        pos_cosi = order_cos[i]
        pos_kl = order_kl[i]

        par_cosi = c_par[pos_cosi]
        val_cosi = cos_dists[pos_cosi]

        par_kl = c_par[pos_kl]
        val_kl = kl_divs[pos_kl]

        print(",{0:04d}, {6:0.7f}, {1:05.3f}, {2:05.3f},,{3:04d}, {7:0.7f}, {4:5.3f}, {5:5.3f}"
              .format(pos_cosi, par_cosi[0], par_cosi[1], pos_kl, par_kl[0], par_kl[1], val_cosi, val_kl))

    print("--")

    if p:
        plt.plot(np.arange(0, 10, 0.01), b_dist, label="reference")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), bbox_transform=plt.gcf().transFigure,
                   fancybox=True, shadow=True)
        plt.show()

