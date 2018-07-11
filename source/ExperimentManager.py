#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import glob
import pickle

from collections import OrderedDict

from rankutils.labeling import *
from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, preprocess_ranks
from rankutils.mappings import descriptor_map, baseline_map
from rankutils.classification import *

import time

import ipdb as pdb

import numpy as np
np.random.seed(93311)

class ExperimentManager:

    def __init__(self, pathcfg="path.cfg", dbparamscfg="dbparams.cfg"):

        self.__pathcfg = cfgloader(pathcfg)
        self.__dbparams = cfgloader(dbparamscfg)

        self.__expmap = OrderedDict()

    @property
    def pathcfg(self):
        return self.__pathcfg

    @property
    def dbparams(self):
        return self.__dbparams

    @property
    def evalcfg(self):
        return self.__evalcfg

    @property
    def expmap(self):
        return self.__expmap

    def set_experiment_map(self, dataset_pairs=[], set=''):

        if set == 'all':
            self.__expmap = OrderedDict(descriptor_map)
        elif set == 'none':
            self.__expmap = OrderedDict()
        else:
            self.__expmap = OrderedDict()
            for dataset, descnum in dataset_pairs:
                if descnum not in descriptor_map[dataset]:
                    raise ValueError("Unavailable descriptor number {0:d} for dataset {1:s}."
                                     " Available: {2:s}".format(descnum, dataset, str(descriptor_map[dataset])))
                else:
                    if dataset not in self.__expmap:
                        self.__expmap[dataset] = [descnum]
                    else:
                        self.__expmap[dataset].append(descnum)

    def add_to_experiment_map(self, dataset, descnum):

        if descnum not in descriptor_map[dataset]:
            raise ValueError("Unavailable descriptor number {0:d} for dataset {1:s}."
                             " Available: {2:s}".format(descnum, dataset, str(descriptor_map[dataset])))
        else:
            if dataset in self.__expmap:
                if descnum not in self.__expmap:
                    self.__expmap[dataset].append(descnum)
            else:
                self.__expmap[dataset] = [descnum]

    def list_available_experiments(self):

        for dataset in descriptor_map:
            print("{0:<15s} ->".format("[" + dataset + "]"), descriptor_map[dataset])

        print("\n---\n")

        return


    def run_baselines(self, bslname, k, bsl):

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print(". Creating Baselines for ", dataset, " -- descriptor", descnum)
                print(". Baseline output name: ", bslname)

                rkdir = self.__pathcfg['rank'][dkey]
                fold_idx = np.load(glob.glob(rkdir + "*folds.npy")[0])

                outdir = self.__pathcfg['output'][dkey] + "{0:s}/rel-prediction/".format(bslname)
                safe_create_dir(outdir)

                n, rounds = fold_idx.shape
                print("   -> Total files: ", n)
                print("   -> # of rounds: ", rounds)

                for r in range(rounds):

                    # Test fold is 0 for round r
                    fs = np.sum(fold_idx[:, r] == 0)  # Number of examples indexed for fold 0
                    outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_bsl_irp.npy".format(outdir, dkey, r, k)

                    params = (fs, k, self.__dbparams[dkey].getfloat('p10'))
                    if bsl == 'maxn':
                        labels = np.load(glob.glob(self.__pathcfg["label"][dkey] + "*irp*")[0])
                        fidx = np.argwhere(fold_idx[:, r] == 0)
                        bslarray = baseline_map[bsl](labels[fidx])

                    elif bsl == 'ranp':
                        bslarray = baseline_map[bsl](*params)

                    else:
                        bslarray = baseline_map[bsl](*params[0:2])

                    np.save(outfile, bslarray)

                    # Test fold is 1 for round r
                    fs = np.sum(fold_idx[:, r] == 1)  # Number of examples indexed for fold 1
                    outfile = "{0:s}{1:s}_r{2:03d}_001_top{3:d}_bsl_irp.npy".format(outdir, dkey, r, k)

                    # Shape from test_idx is the number of examples in that fold
                    params = (fs, k, self.__dbparams[dkey].getfloat('p10'))
                    if bsl == 'maxn':
                        labels = np.load(glob.glob(self.__pathcfg["label"][dkey] + "*irp*")[0])
                        fidx = np.argwhere(fold_idx[:, r] == 1)
                        bslarray = baseline_map[bsl](labels[fidx])

                    elif bsl == 'ranp':
                        bslarray = baseline_map[bsl](*params)

                    else:
                        bslarray = baseline_map[bsl](*params[0:2])

                    np.save(outfile, bslarray)

    # Deprecated
    def run_label_conversion(self, l, h):

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:

                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print("Converting labels from ", dataset, " -- descriptor", descnum)

                indir = self.__pathcfg['label'][dkey]

                if(os.path.isdir(indir)):
                    inflist = glob.glob(indir + "*irp*.npy")

                    outdir = self.__pathcfg['label'][dkey]
                    safe_create_dir(outdir)

                    for infile in inflist:

                        inarr = np.load(infile)
                        inarr_sum = np.sum(inarr, 1)

                        aux = []

                        for i in range(l, h + 1):
                            aux.append((inarr_sum >= i).reshape(-1, 1).astype(np.uint8))

                        outarr = np.hstack(aux)

                        parts = (os.path.basename(infile)).rsplit(".", 2)
                        outfile = outdir + parts[0] + ".rpp_lbls.npy"

                        print("    |_", os.path.basename(infile), "->", os.path.basename(outfile))

                        np.save(outfile, outarr)
                else:
                    print(" -> input directory <{0:s}> does not exist!".format(indir))

                print("---")


    def run_irp_to_rpp_conversion(self, expfoldername, l, h):

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print("Running IRP to RPP conversion on", dataset, " -- descriptor", descnum)
                print(" -> exp:", expfoldername)

                indir = self.__pathcfg['output'][dkey] + expfoldername + "/rel-prediction/"

                if(os.path.isdir(indir)):
                    inflist = glob.glob(indir + "*.npy")

                    outdir = self.__pathcfg['output'][dkey] + expfoldername + "/perf-prediction/"
                    safe_create_dir(outdir)

                    for infile in inflist:

                        inarr = np.load(infile)
                        inarr_sum = np.sum(inarr, 1)

                        aux = []

                        for i in range(l, h+1):
                            aux.append((inarr_sum >= i).reshape(-1, 1).astype(np.uint8))

                        outarr = np.hstack(aux)

                        parts = (os.path.basename(infile)).rsplit("_", 1)
                        outfile = outdir + parts[0] + "_rpp.npy"

                        print("    |_", os.path.basename(infile), "->", os.path.basename(outfile))

                        np.save(outfile, outarr)
                else:
                    print(" -> input directory <{0:s}> does not exist!".format(indir))

                print("---")


    def run_weibull_mr(self, expconfig, sampling=-1.0, backend="matlab"):

        #from ranku.weibull_r import WeibullMR_R
        from rankutils.weibull_m import WeibullMR_M

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        method = expcfg['IRP']['method']
        opt = expcfg['IRP']['optimization']
        notop = expcfg['DEFAULT'].getboolean('notop')
        k = expcfg['DEFAULT'].getint('topk')

        if method == 'fixed':
            fran = (0.1, 0.85, 0.1)
            zran = (0.2, 1.1, 0.2)
        elif method == 'mixt':
            fran = (0.2, 0.85, 0.2)
            zran = (0.25, 1.1, 0.25)

        for dataset in self.__expmap:
                for descnum in self.__expmap[dataset]:
                    dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                    print(". Running <Weibull MR - IRP> for ", dataset, " -- descriptor", descnum)
                    print(". Experiment name: ", expname)

                    fold_idx = np.load(glob.glob(self.__pathcfg['rank'][dkey] + "*folds.npy")[0])
                    n, rounds = fold_idx.shape

                    # Loads and preprocesses the ranks. The result is an NxM matrix, where N is the
                    # number of ranks and M is an arbitrary maximum size for the processed ranks.
                    # M is used to homogenize the size of the ranks
                    ranks = preprocess_ranks(self.__pathcfg['rank'][dkey], self.__dbparams[dkey]['scoretype'],
                                             maxsz=8000)
                    labels = np.load(glob.glob(self.__pathcfg['label'][dkey] + '*irp*')[0])

                    # Consistency checkings
                    assert n == ranks.shape[0], "Inconsistent number of indexes <{0:d}> and rank files <{1:d}>."\
                                                .format(n, len(ranks.shape[0]))

                    assert n == labels.shape[0], "Inconsistent number of indexes <{0:d}> and labels <{1:d}>."\
                                                 .format(n, labels.shape[0])

                    assert labels.shape[1] == k, "Inconsistent number of labels <{0:d}> and k <{1:d}>."\
                                                 .format(labels.shape[1], k)

                    outdir = self.__pathcfg['output'][dkey] + "{0:s}/rel-prediction/".format(expname)
                    safe_create_dir(outdir)

                    for r in range(rounds):
                        print("  -> Starting round #:", r)

                        # Getting round r indexes for fold 0 and fold 1
                        idx_0 = np.argwhere(fold_idx[:, r] == 0).reshape(-1)
                        idx_1 = np.argwhere(fold_idx[:, r] == 1).reshape(-1)

                        # Train is fold 1 and Test is fold 0
                        TEST_X = ranks[idx_0, :]
                        wblpath = "{0:s}weibull-{2:s}_r{1:03d}_000.wbl".format(outdir, r, method)

                        # Let's try to open the saved classifier file. If not possible, we've got to retrain it.
                        try:
                            with open(wblpath, 'rb') as inpf:
                                wbl = pickle.load(inpf)

                            te = -1.0
                            ts = 0.0

                        except FileNotFoundError:
                            wbl = WeibullMR_M(k=k, method=method, opt_metric=opt, notop=False, verbose=True)
                            TRAIN_X = ranks[idx_1, :]
                            TRAIN_y = labels[idx_1, :]

                            if sampling > 0.0 and TRAIN_X.shape[0] >= 1000:
                                sample_i = np.arange(0, TRAIN_X.shape[0])
                                np.random.shuffle(sample_i)
                                sample_i = sample_i[:np.int(sampling*TRAIN_X.shape[0])]

                                TRAIN_X = TRAIN_X[sample_i, :]
                                TRAIN_y = TRAIN_y[sample_i, :]

                            #pdb.set_trace()
                            ts = time.perf_counter()
                            wbl.fit(TRAIN_X, TRAIN_y, f_val=fran, z_val=zran)
                            te = time.perf_counter()

                        print("     -> With [1] as training (F:{0:0.2f}, Z:{1:0.2f}): M = {2:0.3f}"
                              .format(wbl.F, wbl.Z, wbl.opt_val))
                        print("     -> Elapsed: {0:0.3f}s".format(te-ts))

                        predicted, _ = wbl.predict(TEST_X)

                        outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                        print("     ->", os.path.basename(outfile), "...", end="", flush=True)

                        np.save(outfile, predicted)

                        # Let's save this predictor
                        with open(wblpath, 'wb') as outf:
                            pickle.dump(wbl, outf)
                        print(" Done!\n")

                        # Train is fold 0 and Test is fold 1
                        TEST_X = ranks[idx_1, :]
                        wblpath = "{0:s}weibull-{2:s}_r{1:03d}_001.wbl".format(outdir, r, method)

                        # Let's try to open the saved classifier file. If not possible, we've got to retrain it.
                        try:
                            with open(wblpath, 'rb') as inpf:
                                wbl = pickle.load(inpf)

                            te = -1.0
                            ts = 0.0

                        except FileNotFoundError:
                            wbl = WeibullMR_M(k=k, method=method, opt_metric=opt, notop=False, verbose=True)
                            TRAIN_X = ranks[idx_0, :]
                            TRAIN_y = labels[idx_0, :]

                            # SAMPLING SHOULD GO HERE #

                            if sampling > 0.0 and TRAIN_X.shape[0] >= 1000:
                                sample_i = np.arange(0, TRAIN_X.shape[0])
                                np.random.shuffle(sample_i)
                                sample_i = sample_i[:np.int(sampling*TRAIN_X.shape[0])]

                                TRAIN_X = TRAIN_X[sample_i, :]
                                TRAIN_y = TRAIN_y[sample_i, :]

                            #pdb.set_trace()
                            ts = time.perf_counter()
                            wbl.fit(TRAIN_X, TRAIN_y, f_val=fran, z_val=zran)
                            te = time.perf_counter()

                        print("     -> With [0] as training (F:{0:0.2f}, Z:{1:0.2f}): M = {2:0.3f}"
                              .format(wbl.F, wbl.Z, wbl.opt_val))
                        print("     -> Elapsed: {0:0.3f}s".format(te - ts))

                        predicted, _ = wbl.predict(TEST_X)

                        outfile = "{0:s}{1:s}_r{2:03d}_001_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                        print("     ->", os.path.basename(outfile), "...", end="", flush=True)

                        np.save(outfile, predicted)

                        # Let's save this predictor
                        with open(wblpath, 'wb') as outf:
                            pickle.dump(wbl, outf)
                        print(" Done!\n")




    def run_learning_mr(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print(". Running <Learning MR - IRP> on", dataset, " -- descriptor", descnum)
                print(". Experiment name: ", expname)

                fold_idx = np.load(glob.glob(self.__pathcfg['rank'][dkey] + "*folds.npy")[0])
                n, rounds = fold_idx.shape

                features = np.load(glob.glob(self.__pathcfg['feature'][dkey] + "*{0:s}*".format(expname))[0])
                labels = np.load(glob.glob(self.__pathcfg['label'][dkey] + "*irp*")[0])

                outdir = self.__pathcfg['output'][dkey] + "{0:s}/rel-prediction/".format(expname)
                safe_create_dir(outdir)

                for r in range(rounds):
                    print("  -> Starting round #:", r)
                    idx_0 = np.argwhere(fold_idx[:, r] == 0).reshape(-1)
                    idx_1 = np.argwhere(fold_idx[:, r] == 1).reshape(-1)

                    # Contains the per-rank predictions, considering both folds once as train/test
                    predicted = [[], []]

                    for m in range(0, k):
                        print("      -> Classifying Rank -", m+1)

                        # savez_compressed+load names arrays as arr_0, arr_1, etc.
                        r_features = features["arr_{0:d}".format(m)]
                        r_labels = labels[:, m]

                        # run classification already performs proper fold division. It suffices that a list is passed
                        # whereby each position contains the features according to the fold
                        r_predicted = run_two_set_classification(r_features, r_labels, [idx_0, idx_1], cname, False)

                        # Fold 0 is test
                        predicted[0].append(r_predicted[0])

                        # Fold 1 is test
                        predicted[1].append(r_predicted[1])

                    # We close the round by stacking the per-rank-position predictions, and saving the output files
                    predicted[0] = np.hstack(predicted[0])
                    predicted[1] = np.hstack(predicted[1])

                    # Sanity check: is the total number of predictions the same as the total number of labels?
                    assert (predicted[0].shape[0] + predicted[1].shape[0]) == labels.shape[0],\
                           "Inconsistent number of predicted labels <{0:d} + {1:d} = {2:d}> and labels <{3:d}>"\
                           .format(predicted[0].shape[0], predicted[1].shape[0],
                                   (predicted[0].shape[0] + predicted[1].shape[0]), labels.shape[0])

                    # Phew, all done. Let's save the predictions. The naming convention is has rXXX_YYY where XXX is the
                    # number of the round, and YYY is either 000 (when fold 0 was test) or 001 (when fold 1 was test)
                    outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    np.save(outfile, predicted[0])

                    outfile = "{0:s}{1:s}_r{2:03d}_001_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    np.save(outfile, predicted[1])


        return







