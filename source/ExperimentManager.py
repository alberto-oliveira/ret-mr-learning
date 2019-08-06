#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import glob
import pickle
import shutil

import warnings

from collections import OrderedDict
from tqdm import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from rankutils.labeling import *
from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, preprocess_ranks
from rankutils.mappings import ranking_type_map, baseline_map
from rankutils.classification import *
from rankutils.postprocessing import label_frequency_mod

import time

import ipdb as pdb

import numpy as np

bseed = 93311

np.random.seed(bseed)

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

    def set_experiment_map(self, dataset_pairs=None, set=''):

        if set == 'all':
            self.__expmap = OrderedDict(ranking_type_map)
        elif set == 'none':
            self.__expmap = OrderedDict()
        else:
            if isinstance(dataset_pairs, dict):
                self.__expmap = OrderedDict(dataset_pairs)

            elif isinstance(dataset_pairs, list):
                self.__expmap = OrderedDict()
                for dataset, rktpnum in dataset_pairs:
                    if rktpnum not in ranking_type_map[dataset]:
                        raise ValueError("Unavailable ranking-type number {0:d} for dataset {1:s}."
                                         " Available: {2:s}".format(rktpnum, dataset, str(ranking_type_map[dataset])))
                    else:
                        if dataset not in self.__expmap:
                            self.__expmap[dataset] = [rktpnum]
                        else:
                            self.__expmap[dataset].append(rktpnum)

    def add_to_experiment_map(self, dataset, rktpnum):

        if rktpnum not in ranking_type_map[dataset]:
            raise ValueError("Unavailable ranking-type number {0:d} for dataset {1:s}."
                             " Available: {2:s}".format(rktpnum, dataset, str(ranking_type_map[dataset])))
        else:
            if dataset in self.__expmap:
                if rktpnum not in self.__expmap:
                    self.__expmap[dataset].append(rktpnum)
            else:
                self.__expmap[dataset] = [rktpnum]

    def list_available_experiments(self):

        for dataset in ranking_type_map:
            print("{0:<15s} ->".format("[" + dataset + "]"), ranking_type_map[dataset])

        print("\n---\n")

        return

    def run_baselines(self, bsl, k, outfolder):

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Creating Baselines for ", dataset, " -- ", rktpname)
                print(". Baseline output name: ", outfolder)

                rkdir = self.__pathcfg[dkey]['rank']
                fold_idx = np.load(glob.glob(rkdir + "*folds.npy")[0])

                outdir = self.__pathcfg[dkey]['output'] + "{0:s}/".format(outfolder)
                safe_create_dir(outdir)

                n, rounds = fold_idx.shape
                print("   -> Total files: ", n)
                print("   -> # of rounds: ", rounds)

                for r in range(rounds):

                    # Test fold is 0 for round r
                    fs = np.sum(fold_idx[:, r] == 0)  # Number of examples indexed for fold 0
                    outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_bsl_irp.npy".format(outdir, dkey, r, k)

                    params = (fs, k, self.__dbparams.getfloat(dkey, 'p10'))
                    if bsl == 'maxn':
                        labels = np.load(glob.glob(self.__pathcfg[dkey]["label"] + "*{0:s}*".format(rktpname))[0])
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
                    params = (fs, k, self.__dbparams.getfloat(dkey, 'p10'))
                    if bsl == 'maxn':
                        labels = np.load(glob.glob(self.__pathcfg[dkey]["label"] + "*{0:s}*".format(rktpname))[0])
                        fidx = np.argwhere(fold_idx[:, r] == 1)
                        bslarray = baseline_map[bsl](labels[fidx])

                    elif bsl == 'ranp':
                        bslarray = baseline_map[bsl](*params)

                    else:
                        bslarray = baseline_map[bsl](*params[0:2])

                    np.save(outfile, bslarray)

        return

### -------------------- UNUSED -------------------- ###
    def run_stat_positional_mr(self, expconfig):

        from rankutils.stat_mr import StatMR

        expcfg = cfgloader(expconfig)

        expname = expcfg.get('IRP', 'expname')
        dist_name = expcfg.get('IRP', 'distribution')
        step = expcfg.getint('IRP', 'positional_step')
        k = expcfg.getint('DEFAULT', 'topk')

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:

                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print(". Running <Statistical-Positional MR - IRP> for ", dataset, " -- descriptor", descnum)
                print(". Experiment name: ", expname)

                fold_idx = np.load(glob.glob(self.__pathcfg['rank'][dkey] + "*folds.npy")[0])
                n, rounds = fold_idx.shape

                # Loads and preprocesses the ranks. The result is an NxM matrix, where N is the
                # number of ranks and M is an arbitrary maximum size for the processed ranks.
                # M is used to homogenize the size of the ranks
                ranks = preprocess_ranks(self.__pathcfg['rank'][dkey], maxsz=8000)

                # Consistency checks
                assert n == ranks.shape[0], "Inconsistent number of indices <{0:d}> and rank files <{1:d}>."\
                                            .format(n, len(ranks.shape[0]))

                outdir = self.__pathcfg['output'][dkey] + "{0:s}/".format(expname)
                safe_create_dir(outdir)

                for r in tqdm(range(rounds), ncols=100, desc='Rounds ', total=rounds):
                    #print("  -> Starting round #:", r)

                    # Getting round r indices for fold 0 and fold 1
                    idx_0 = np.argwhere(fold_idx[:, r] == 0).reshape(-1)
                    idx_1 = np.argwhere(fold_idx[:, r] == 1).reshape(-1)

                    # Test is fold 0
                    TEST_X = ranks[idx_0, :]

                    # Pre-allocating the predictions. There is a total of m predictions, each of k positions.
                    # M is the number of ranks in the test fold.
                    predicted = np.zeros((TEST_X.shape[0], k), dtype=np.uint8)
                    for i, rk in enumerate(tqdm(TEST_X, total=TEST_X.shape[0], ncols=100, desc='    Rank')):
                        #print('     |_ [Rank {number:04d}]'.format(number=i))

                        t_vals = np.zeros(k, dtype=np.float64)
                        tail = rk[k:]

                        for pos in range(k):

                            tail = np.unique(tail[pos*step:])[::-1]
                            tail = tail[tail != -1]

                            dparams = StatMR.ev_estim_matlab(tail, dist_name)
                            t_vals[pos] = StatMR.ev_quant_matlab(dist_name, 0.99999999, scale=dparams['scale'],
                                                                 shape=dparams['shape'], loc=dparams['loc'])

                        predicted[i] = rk[0:k] > t_vals

                    outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    print("     ->", os.path.basename(outfile), "...", end="", flush=True)

                    np.save(outfile, predicted)


                    # Test is fold 1
                    TEST_X = ranks[idx_1, :]

                    # Pre-allocating the predictions. There is a total of m predictions, each of k positions.
                    # M is the number of ranks in the test fold.
                    predicted = np.zeros((TEST_X.shape[0], k), dtype=np.uint8)

                    for i, rk in enumerate(TEST_X):
                        print('     |_ [Rank {number:04d}]'.format(number=i))

                        t_vals = np.zeros(k, dtype=np.float64)
                        tail = rk[k:]

                        for pos in range(k):

                            tail = np.unique(tail[pos*step:])[::-1]
                            tail = tail[tail != -1]

                            dparams = StatMR.ev_estim_matlab(tail, dist_name)
                            t_vals[pos] = StatMR.ev_quant_matlab(dist_name, 0.99999999, scale=dparams['scale'],
                                                                 shape=dparams['shape'], loc=dparams['loc'])

                        predicted[i] = rk[0:k] > t_vals

                    outfile = "{0:s}{1:s}_r{2:03d}_001_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    print("     ->", os.path.basename(outfile), "...", end="", flush=True)

                    np.save(outfile, predicted)

        return
### ------------------------------------------------ ###

    def run_statistical_mr_v2(self, expconfig, sampling=-1.0, overwrite=False):

        from rankutils.stat_mr import StatMR

        expcfg = cfgloader(expconfig)

        expname = expcfg.get('IRP', 'expname')
        dist_name = expcfg.get('IRP', 'distribution')
        method = expcfg.get('IRP', 'method')
        opt = expcfg.get('IRP', 'optimization')
        k = expcfg.getint('DEFAULT', 'topk')

        if method == 'fixed':
            fvals = [0, 0.05, 0.15, 0.30, 0.40, 0.50, 0.60]
            zvals = [0.70, 0.80, 1.0]
            #fvals = [0, 0.005, 0.05, 0.15, 0.30, 0.50, 0.75]
            #zvals = [0.80, 1.0]
        elif method == 'mixt':
            fvals = [0, 0.05, 0.15, 0.30, 0.40, 0.50, 0.60]
            zvals = [0.75, 1.0]
            #fvals = [0, 0.005, 0.05, 0.15, 0.30, 0.50, 0.75]
            #zvals = [0.80, 1.0]

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Statistical MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                #fold_idx = np.load(glob.glob(self.__pathcfg[dkey]['rank'] + "*folds.npy")[0])
                #n, rounds = fold_idx.shape

                rounds = 5

                # Loads and preprocesses the ranks. The result is an NxM matrix, where N is the
                # number of ranks and M is an arbitrary maximum size for the processed ranks.
                # M is used to homogenize the size of the ranks
                ranks = preprocess_ranks(self.__pathcfg[dkey]['rank'], maxsz=15000)
                n = ranks.shape[0]

                labels = np.load(glob.glob(self.__pathcfg[dkey]['label'] + '*{0:s}*'.format(rktpname))[0])

                assert n == labels.shape[0], "Inconsistent number of indices <{0:d}> and labels <{1:d}>."\
                                             .format(n, labels.shape[0])

                assert labels.shape[1] >= k, "Inconsistent number of labels <{0:d}> and k <{1:d}>."\
                                             .format(labels.shape[1], k)

                if labels.shape[1] > k:
                    labels = labels[:, 0:k]

                outdir = self.__pathcfg[dkey]['output'] + "{0:s}/".format(expname)

                if os.path.isdir(outdir):
                    shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.uint8)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for. Retained in this function
                #      for compatibility, but will always be the same for all 0 <= i <= k
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                rstratkfold = RepeatedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                splitgen = rstratkfold.split(ranks)

                for r in range(rounds):
                    print("  -> Starting round #:", r)

                    # Run the two splits for the round, alternating train/test
                    for spl in range(2):

                        print("   -> split #{0:d}".format(spl+1))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            train_idx, test_idx = next(splitgen)

                        # Train is fold 1 and Test is fold 0
                        TEST_X = ranks[test_idx, :]
                        mr_path = "{outdir:s}{distname:s}-{method:s}_r{round:03d}_{split:03d}.mr".format(outdir=outdir,
                                                                                                  distname=dist_name,
                                                                                                  round=r,
                                                                                                  method=method,
                                                                                                  split=spl)

                        # Let's try to open the saved classifier file. If not possible, we've got to retrain it.
                        try:
                            if overwrite:
                                raise FileNotFoundError

                            with open(mr_path, 'rb') as inpf:
                                stat_mr = pickle.load(inpf)

                        except FileNotFoundError:
                            stat_mr = StatMR(dist_name=dist_name, k=k, method=method, opt_metric=opt, verbose=True)
                            TRAIN_X = ranks[train_idx, :]
                            TRAIN_y = labels[train_idx, :]

                            if sampling > 0.0 and TRAIN_X.shape[0] >= 100:
                                s = np.int(sampling*TRAIN_X.shape[0])
                                sample_i = np.random.choice(np.arange(TRAIN_X.shape[0]), size=s, replace=False)

                                TRAIN_X = TRAIN_X[sample_i, :]
                                TRAIN_y = TRAIN_y[sample_i, :]

                            stat_mr.fit(TRAIN_X, TRAIN_y, f_val=fvals, z_val=zvals)

                        print("     -> Split #{0:d} -- (F:{1:0.2f}, Z:{2:0.2f}): M = {3:0.3f}"
                              .format(spl, stat_mr.F, stat_mr.Z, stat_mr.opt_val))

                        pred_y, _ = stat_mr.predict(TEST_X)

                        assert predicted[r, test_idx].shape == pred_y.shape, \
                            "Shape of the output prediction <{0:s}> should be equal to the shape of the output array" \
                            "<{1:s}>.".format(str(pred_y.shape), str(predicted[r, test_idx].shape))

                        predicted[r, test_idx] = pred_y

                        # Let's save this predictor
                        with open(mr_path, 'wb') as outf:
                            pickle.dump(stat_mr, outf)
                        print(" Done split!\n")

                    # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                    # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                    # <r> and  for all rank positions
                    splits[r, train_idx, :] = 1

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp.npy".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

        return


    def run_pos_learning_mr_v2(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')
        featpack = expcfg.get('DEFAULT', 'features', fallback=expname)
        prob = expcfg.getboolean('DEFAULT', 'probability', fallback=False)

        rounds = 5

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Learning MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                feat_pack = np.load(glob.glob(self.__pathcfg[dkey]['feature'] + "*{0:s}*".format(featpack))[0])
                nk, n, v, d = feat_pack['features'].shape

                assert v == 1, "learning v2 does not working with multi-variation approach. " \
                               "Number of variations is <{0:d}>".format(v)

                features = feat_pack['features'].reshape(nk, n, d)
                labels = feat_pack['labels'].reshape(nk, n, 1)

                assert (features.shape[0] == labels.shape[0] and features.shape[1] == labels.shape[1]),\
                        "inconsistent shapes between features and labels. "

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}.{cfname:s}/".format(expname=expname,
                                                                                           cfname=cname)

                #if os.path.isdir(outdir):
                    #shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32)
                probabilities = np.zeros((rounds, n, k), dtype=np.float32)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                for m in tqdm(range(k), ncols=100, desc='        |_Position', total=k):

                    p_features = features[m]
                    p_labels = labels[m]

                    rstratkfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                    splitgen = rstratkfold.split(p_features, p_labels)

                    for r in tqdm(range(rounds), ncols=100, desc='   |_Round', total=rounds):

                        #if dataset == 'oxford':
                        #    pdb.set_trace()

                        # 1st split
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            train_idx, test_idx = next(splitgen)

                        pred_y, prob_y = run_positional_classification(p_features, p_labels, [train_idx, test_idx], cname,
                                                               False, get_prob=prob)

                        # test_idx has the sample indices which have been predicted for the 1st split of round <r> and
                        # for position <m> of the rank. We change those values in the <predicted> array to match the
                        # prediction
                        predicted[r, test_idx, m] = pred_y
                        if prob_y != []:
                            probabilities[r, test_idx, m] = prob_y
                        # ---------

                        #if dataset == 'oxford':
                        #    pdb.set_trace()
                        # 2nd split -- train from 1st split is test here, and vice-versa
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            train_idx, test_idx = next(splitgen)

                        pred_y, prob_y = run_positional_classification(p_features, p_labels, [train_idx, test_idx], cname,
                                                               False, get_prob=prob)

                        # test_idx has the sample indices which have been predicted for the 2nd split of round <r> and
                        # for position <m> of the rank. We change those values in the <predicted> array to match the
                        # prediction
                        predicted[r, test_idx, m] = pred_y
                        if prob_y != []:
                            probabilities[r, test_idx, m] = prob_y
                        # ---------

                        # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                        # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                        # <r> and for rank position <m>
                        splits[r, train_idx, m] = 1

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                np.save(outfile, probabilities)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                feat_pack.close()
                print('\n')

        return

    def run_block_learning_mr_v2(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg.getint('DEFAULT', 'topk')
        block = expcfg.getint('DEFAULT', 'block', fallback=5)
        featpack = expcfg.get('DEFAULT', 'features', fallback=expname)
        prob = expcfg.getboolean('DEFAULT', 'probability', fallback=False)

        rounds = 5

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Learning MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                feat_pack = np.load(glob.glob(self.__pathcfg[dkey]['feature'] + "*{0:s}*".format(featpack))[0])
                nk, n, v, d = feat_pack['features'].shape

                assert v == 1, "learning v2 does not working with multi-variation approach. " \
                               "Number of variations is <{0:d}>".format(v)

                features = feat_pack['features'].reshape(nk, n, d)
                labels = feat_pack['labels'].reshape(nk, n, 1)

                assert (features.shape[0] == labels.shape[0] and features.shape[1] == labels.shape[1]),\
                        "inconsistent shapes between features and labels. "

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}.{cfname:s}/".format(expname=expname,
                                                                                           cfname=cname)

                #if os.path.isdir(outdir):
                    #shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32)
                probabilities = np.zeros((rounds, n, k), dtype=np.float32)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                for m in tqdm(range(0, k, block), ncols=70, desc='        |_Block', total=int(np.floor(k/block))):

                    bs = m
                    be = m + block

                    rstratkfold = RepeatedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                    splitgen = rstratkfold.split(features[0])

                    for r in tqdm(range(rounds), ncols=70, desc='   |_Round', total=rounds):
                        # pdb.set_trace()

                        # 1st split
                        train_idx, test_idx = next(splitgen)

                        pred_list, prob_list = run_block_classification(features, labels, [train_idx, test_idx], cname,
                                                                        bs, be, False)

                        # test_idx has the sample indices which have been predicted for the 1st split of round <r> and
                        # for position <bs>  to <be> of the rank. We change those values in the <predicted> array to
                        # match the prediction
                        for p in range(block):
                            if bs+p >= k:
                                break
                            predicted[r, test_idx, bs+p] = pred_list[p]
                            if prob_list:
                                probabilities[r, test_idx, bs+p] = prob_list[p]
                        # ---------

                        # 2nd split -- train from 1st split is test here, and vice-versa
                        train_idx, test_idx = next(splitgen)

                        pred_list, prob_list = run_block_classification(features, labels, [train_idx, test_idx], cname,
                                                                        bs, be, False)

                        # test_idx has the sample indices which have been predicted for the 2nd split of round <r> and
                        # for position <bs>  to <be> of the rank. We change those values in the <predicted> array to
                        # match the prediction

                        for p in range(block):
                            if bs+p >= k:
                                break
                            predicted[r, test_idx, bs+p] = pred_list[p]
                            if prob_list:
                                probabilities[r, test_idx, bs+p] = prob_list[p]
                        # ---------

                        # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                        # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                        # <r> and for rank position <m>
                        splits[r, train_idx, :] = 1

                    outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                    np.save(outfile, predicted)

                    outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                    np.save(outfile, probabilities)

                    # saving the fold division
                    np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                    feat_pack.close()
                    print('\n')

        return

    def run_single_learning_mr_v2(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')
        featpack = expcfg.get('DEFAULT', 'features', fallback=expname)
        prob = expcfg.getboolean('DEFAULT', 'probability', fallback=False)

        rounds = 5

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Learning MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                feat_pack = np.load(glob.glob(self.__pathcfg[dkey]['feature'] + "*{0:s}*".format(featpack))[0])
                nk, n, v, d = feat_pack['features'].shape

                assert v == 1, "learning v2 does not working with multi-variation approach. " \
                               "Number of variations is <{0:d}>".format(v)

                features = feat_pack['features'].reshape(nk, n, d)
                labels = feat_pack['labels'].reshape(nk, n, 1)

                assert (features.shape[0] == labels.shape[0] and features.shape[1] == labels.shape[1]),\
                        "inconsistent shapes between features and labels. "

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}.{cfname:s}/".format(expname=expname,
                                                                                           cfname=cname)

                #if os.path.isdir(outdir):
                    #shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32)
                probabilities = np.zeros((rounds, n, k), dtype=np.float32)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                rstratkfold = RepeatedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                splitgen = rstratkfold.split(features[0])

                for r in tqdm(range(rounds), ncols=70, desc='   |_Round', total=rounds):
                    #pdb.set_trace()

                    # 1st split
                    train_idx, test_idx = next(splitgen)

                    pred_list, prob_list = run_single_classification(features, labels,
                                                          [train_idx, test_idx], cname, False, get_prob=prob)

                    # test_idx has the sample indices which have been predicted for the 1st split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    for m in range(k):
                        predicted[r, test_idx, m] = pred_list[m]
                        if prob_list:
                            probabilities[r, test_idx, m] = prob_list[m]
                    # ---------

                    # 2nd split -- train from 1st split is test here, and vice-versa
                    train_idx, test_idx = next(splitgen)

                    pred_list, prob_list = run_single_classification(features, labels,
                                                          [train_idx, test_idx], cname, False, get_prob=prob)

                    # test_idx has the sample indices which have been predicted for the 2nd split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    for m in range(k):
                        predicted[r, test_idx, m] = pred_list[m]
                        if prob_list:
                            probabilities[r, test_idx, m] = prob_list[m]
                    # ---------

                    # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                    # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                    # <r> and for rank position <m>
                    splits[r, train_idx, :] = 1

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                np.save(outfile, probabilities)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                feat_pack.close()
                print('\n')

        return

    def run_sequence_labeling_mr(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')
        featpack_name = expcfg.get('DEFAULT', 'features', fallback=expname)
        prob = expcfg.getboolean('DEFAULT', 'probability', fallback=False)
        seq_size = expcfg.getint('DEFAULT', 'sequence_size', fallback=k)

        rounds = 5

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Sequence Labeling MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                feat_pack = np.load(glob.glob(self.__pathcfg[dkey]['seqfeature'] + "*{0:s}*".format(featpack_name))[0])

                sequences = feat_pack['features']
                labels = feat_pack['labels']

                assert sequences.shape[1] == k, "Shape of sequence <{0:d}> inconsistent with k <{1:d}>"\
                                                .format(sequences.shape[1], k)

                assert (sequences.shape[0] == labels.shape[0]
                        and sequences.shape[1] == labels.shape[1]), \
                        "inconsistent shapes between features and labels. "

                assert k % seq_size == 0, "k <{0:d}> should be divisible by seq_size <{1:d}>".format(k, seq_size)

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}.{cfname:s}/".format(expname=expname,
                                                                                           cfname=cname)

                n, _, _ = sequences.shape

                #if os.path.isdir(outdir):
                    #shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32)
                probabilities = np.zeros((rounds, n, k), dtype=np.float32)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                rkfold = RepeatedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                splitgen = rkfold.split(sequences)

                for r in tqdm(range(rounds), ncols=100, desc='   |_Round', total=rounds):
                    #pdb.set_trace()

                    # 1st split
                    train_idx, test_idx = next(splitgen)

                    pred, prob = run_sequence_labeling(sequences, labels, [train_idx, test_idx], seq_size)

                    # test_idx has the sample indices which have been predicted for the 1st split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    predicted[r, test_idx] = pred
                    probabilities[r, test_idx] = prob
                    # ---------

                    # 2nd split -- train from 1st split is test here, and vice-versa
                    train_idx, test_idx = next(splitgen)

                    pred, prob = run_sequence_labeling(sequences, labels, [train_idx, test_idx], seq_size)

                    # test_idx has the sample indices which have been predicted for the 2nd split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    predicted[r, test_idx] = pred
                    probabilities[r, test_idx] = prob
                    # ---------

                    # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                    # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                    # <r> and for rank position <m>
                    splits[r, train_idx, :] = 1

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                np.save(outfile, probabilities)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                feat_pack.close()
                print('\n')

        return


    def run_single_learning_mr_v2(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')
        featpack = expcfg.get('DEFAULT', 'features', fallback=expname)
        prob = expcfg.getboolean('DEFAULT', 'probability', fallback=False)

        rounds = 5

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Learning MR - IRP> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                feat_pack = np.load(glob.glob(self.__pathcfg[dkey]['feature'] + "*{0:s}*".format(featpack))[0])
                nk, n, v, d = feat_pack['features'].shape

                assert v == 1, "learning v2 does not working with multi-variation approach. " \
                               "Number of variations is <{0:d}>".format(v)

                features = feat_pack['features'].reshape(nk, n, d)
                labels = feat_pack['labels'].reshape(nk, n, 1)

                assert (features.shape[0] == labels.shape[0] and features.shape[1] == labels.shape[1]),\
                        "inconsistent shapes between features and labels. "

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}.{cfname:s}/".format(expname=expname,
                                                                                           cfname=cname)

                #if os.path.isdir(outdir):
                    #shutil.rmtree(outdir)

                safe_create_dir(outdir)

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32)
                probabilities = np.zeros((rounds, n, k), dtype=np.float32)

                # Array storing train-test split for each sample, of each position, in each round
                # n is the number of samples
                # k is the number of top positions which we perform relevance prediction for
                # r is the number of rounds in which we divide the samples in two splits
                # Values of 0 and 1 differentiate the split at that round. By convention, 0 is train first, test last
                # and 1 is test first, train last
                splits = np.zeros((rounds, n, k), dtype=np.uint8)

                rstratkfold = RepeatedKFold(n_splits=2, n_repeats=rounds, random_state=bseed)
                splitgen = rstratkfold.split(features[0])

                for r in tqdm(range(rounds), ncols=70, desc='   |_Round', total=rounds):
                    #pdb.set_trace()

                    # 1st split
                    train_idx, test_idx = next(splitgen)

                    pred_list, prob_list = run_single_classification(features, labels,
                                                          [train_idx, test_idx], cname, False, get_prob=prob)

                    # test_idx has the sample indices which have been predicted for the 1st split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    for m in range(k):
                        predicted[r, test_idx, m] = pred_list[m]
                        if prob_list:
                            probabilities[r, test_idx, m] = prob_list[m]
                    # ---------

                    # 2nd split -- train from 1st split is test here, and vice-versa
                    train_idx, test_idx = next(splitgen)

                    pred_list, prob_list = run_single_classification(features, labels,
                                                          [train_idx, test_idx], cname, False, get_prob=prob)

                    # test_idx has the sample indices which have been predicted for the 2nd split of round <r> and
                    # for position <m> of the rank. We change those values in the <predicted> array to match the
                    # prediction
                    for m in range(k):
                        predicted[r, test_idx, m] = pred_list[m]
                        if prob_list:
                            probabilities[r, test_idx, m] = prob_list[m]
                    # ---------

                    # Remember -- Value 0 is train 1st test 2nd, while value 1 is test 1st train 2nd
                    # Since train_idx is storing the training indices for train 2nd, we change those to 1, at round
                    # <r> and for rank position <m>
                    splits[r, train_idx, :] = 1

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                np.save(outfile, probabilities)

                # saving the fold division
                np.save("{0:s}train_test_splits.npy".format(outdir), splits)

                feat_pack.close()
                print('\n')

        return


    def run_late_fusion(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']

        for dataset in self.__expmap:
            for rktpnum in self.__expmap[dataset]:
                dkey = "{0:s}_{1:03d}".format(dataset, rktpnum)
                rktpname = self.__pathcfg[dkey]['rktpdir']

                print(". Running <Late Fusion> for ", dataset, " -- ", rktpname)
                print(". Experiment name: ", expname)

                nf = 0
                fusion_list = []
                fusion_weights = []

                # Loads individual answers from methods
                for m in expcfg['methods']:
                    if m not in expcfg['DEFAULT']:
                        p_dir = self.__pathcfg[dkey]['output'] + "{0:s}/".format(expcfg.get('methods', m))
                        try:
                            fusion_list.append(np.load(glob.glob("{0:s}*_irp*".format(p_dir))[0]))
                        except IndexError:
                            print(p_dir)
                            raise IndexError
                        nf += 1

                    # Consistency check
                    if len(fusion_list) > 1:
                        shA = fusion_list[-1].shape
                        shB = fusion_list[-2].shape
                        assert shA == shB, "Shape <0:s> of <{1:s}> is inconsistent to <2:s>"\
                                           .format(str(shA), m, str(shB))

                # Reads weights
                for i in range(nf):
                    w_k = "w{0:d}".format(i)
                    fusion_weights.append(expcfg.getfloat('weights', w_k, fallback=1.0))

                outdir = self.__pathcfg[dkey]['output'] + "{expname:s}/".format(expname=expname)

                safe_create_dir(outdir)

                rounds, n, k = fusion_list[-1].shape

                # Stores the predicted labels for each round. The third dimension of size <rounds> aggregated the
                # predictions for both splits. To differentiate the train/test splits, the 'splits' array should
                # be used
                predicted = np.zeros((rounds, n, k), dtype=np.int32) - 1
                probabilities = np.zeros((rounds, n, k), dtype=np.float32) - 1

                for r in range(rounds):

                    fusion_array = np.stack([a[r] for a in fusion_list])

                    aux = np.average(fusion_array, axis=0, weights=fusion_weights)

                    assert aux.shape == (n, k), "Shape of avg. array <{0:s}> inconsitent to <{1:s}>"\
                                                .format(str(aux.shape), str((n, k)))

                    predicted[r] = (aux >= 0.5).astype(np.int32)
                    probabilities[r] = predicted[r]

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_irp".format(outdir, dkey, rounds, k)
                np.save(outfile, predicted)

                outfile = "{0:s}{1:s}_{2:d}rounds2splits_top{3:d}_prob".format(outdir, dkey, rounds, k)
                np.save(outfile, probabilities)

        return

### ------------------------------------------------ ###

    def run_relabeling(self, expcfg_):

        if isinstance(expcfg_, str):
            expcfg = cfgloader(expcfg_)
        elif isinstance(expcfg_, configparser.ConfigParser):
            expcfg = expcfg_
        else:
            raise TypeError("arg \'expcfg_\' should be either <str> or <configparser> type")

        expname = expcfg['DEFAULT']['expname']
        k = expcfg['DEFAULT'].getint('topk')
        suffix = expcfg.get('IRP', 'classifier', fallback='')
        if suffix != '':
            suffix = '.' + suffix

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print(". Running <Relabeling-frequencies> on", dataset, " -- descriptor", descnum)
                print(". Experiment name: ", expname)

                fold_idx = np.load(glob.glob(self.__pathcfg['rank'][dkey] + "*folds.npy")[0])
                n, rounds = fold_idx.shape

                gt_labels = np.load(glob.glob(self.__pathcfg['label'][dkey] + "*irp*")[0])

                indir = self.__pathcfg['output'][dkey] + "{expname:s}{suffix:s}/".format(expname=expname, suffix=suffix)
                predlfpaths = glob.glob(indir + "*.npy")
                predlfpaths.sort()

                outdir = indir[:-1] + ".relabel_p2:p10/"
                safe_create_dir(outdir)

                for r in range(rounds):
                    print("  -> Starting round #:", r)
                    idx_0 = np.flatnonzero(fold_idx[:, r] == 0).reshape(-1)
                    idx_1 = np.flatnonzero(fold_idx[:, r] == 1).reshape(-1)

                    # --- IDX 0 is test and IDX 1 is train
                    pred_labels = np.load(predlfpaths[r * 2])
                    train_labels = gt_labels[idx_1]

                    mod_labels = label_frequency_mod(pred_labels[:, 1:], train_labels[:, 1:])
                    mod_labels = np.hstack([pred_labels[:, 0:1], mod_labels])

                    outfile = "{0:s}{1:s}_r{2:03d}_000_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    np.save(outfile, mod_labels)
                    # ---

                    # --- IDX 1 is test and IDX 0 is train
                    pred_labels = np.load(predlfpaths[(r * 2) + 1])
                    train_labels = gt_labels[idx_0]

                    mod_labels = label_frequency_mod(pred_labels[:, 1:], train_labels[:, 1:])
                    mod_labels = np.hstack([pred_labels[:, 0:1], mod_labels])

                    outfile = "{0:s}{1:s}_r{2:03d}_001_top{3:d}_irp.npy".format(outdir, dkey, r, k)
                    np.save(outfile, mod_labels)
                    # ---


        return