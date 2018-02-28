#/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import glob
import pickle

from collections import OrderedDict

from common.labeling import *
from common.cfgloader import *
from common.utilities import safe_create_dir, preprocess_ranks
from common.mappings import descriptor_map, baseline_map
from common.classification import *

import numpy as np

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
                    self.__expmap[dataset] = [descnum]

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
                foldlist = [rkdir + d + "/" for d in os.listdir(rkdir) if os.path.isdir(rkdir + d + "/")]
                foldlist.sort()

                outdir = self.__pathcfg['output'][dkey] + "{0:s}/rel-prediction/".format(bslname)
                safe_create_dir(outdir)

                for f, folddir in enumerate(foldlist):
                    outfile = "{0:s}{1:s}_f{2:03d}_top{3:d}_bsl_irp.npy".format(outdir, dkey, f, k)

                    fs = len(glob.glob(folddir + "*.rk"))

                    params = (fs, k, self.__dbparams[dkey].getfloat('p10'))
                    if bsl == 'ranp':
                        bslarray = baseline_map[bsl](*params)
                    else:
                        bslarray = baseline_map[bsl](*params[0:2])

                    np.save(outfile, bslarray)

    # Deprecated
    def run_irp_labeling(self, k, fn):

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print("Running labeling on", dataset, " -- descriptor", descnum)

                gtdir = self.__pathcfg["groundtruth"][dkey]
                lbldir = self.__pathcfg["label"][dkey] + "rel-prediction/"
                rkdir = self.__pathcfg["rank"][dkey]

                for f in range(fn):

                    rkflist = os.listdir(rkdir + "fold_{0:d}/".format(f))
                    rkflist.sort()

                    outfile = "{0:s}{1:s}_desc{2:d}_f{3:03d}_top{4:d}_irp_lbls".format(lbldir, dataset, descnum, f, k)

                    lbllist = []

                    for rkf in rkflist:
                        aux = "{0:s}/*{1:s}*".format(gtdir, os.path.splitext(rkf)[0])
                        #print("fold", f, "-> ",os.path.splitext(rkf)[0])
                        gtf = glob.glob(aux)[0]
                        #print("          ", gtf)

                        lbllist.append(rp_labeling(gtf, k))

                    safe_create_dir(lbldir)
                    lblarray = np.vstack(lbllist)
                    np.save(outfile, lblarray)

    # Deprecated
    def run_rpp_labeling(self, k, fn):

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print("Running labeling on", dataset, " -- descriptor", descnum)

                gtdir = pathcfg["groundtruth"][dkey]
                lbldir = pathcfg["label"][dkey]
                rkdir = pathcfg["rank"][dkey]

                for f in range(fn):

                    rkflist = os.listdir(rkdir + "fold_{0:d}/".format(f))
                    rkflist.sort()

                    outfile = "{0:s}{1:s}_desc{2:d}_f{3:03d}_top{4:d}_labels".format(lbldir, dataset, descnum, f, k)

                    lbllist = []

                    for rkf in rkflist:
                        aux = "{0:s}/*{1:s}*".format(gtdir, os.path.splitext(rkf)[0])
                        # print("fold", f, "-> ",os.path.splitext(rkf)[0])
                        gtf = glob.glob(aux)[0]
                        # print("          ", gtf)

                        lbllist.append(pp_labeling(gtf, k))

                    safe_create_dir(lbldir)
                    lblarray = np.vstack(lbllist)
                    np.save(outfile, lblarray)

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


    def run_weibull_mr(self, expconfig, sampling=1.0, backend="matlab"):

        #from common.weibull_r import WeibullMR_R
        from common.weibull_m import WeibullMR_M

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        method = expcfg['IRP']['method']
        opt = expcfg['IRP']['optimization']
        notop = expcfg['DEFAULT'].getboolean('notop')
        k = expcfg['DEFAULT'].getint('topk')

        for dataset in self.__expmap:
                for descnum in self.__expmap[dataset]:
                    dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                    print(". Running <Weibull MR - IRP> for ", dataset, " -- descriptor", descnum)
                    print(". Experiment name: ", expname)

                    rkdir = self.__pathcfg['rank'][dkey]
                    foldlist = [rkdir + d + "/" for d in os.listdir(rkdir) if os.path.isdir(rkdir + d + "/")]
                    foldlist.sort()

                    lbllist = glob.glob(self.__pathcfg['label'][dkey] + 'rel-prediction/*.npy')
                    lbllist.sort()

                    assert len(foldlist) == len(lbllist)

                    outdir = self.__pathcfg['output'][dkey] + "{0:s}/".format(expname)
                    safe_create_dir(outdir + 'rel-prediction/')

                    folds = []
                    labels = []

                    for folddir, lblpath in zip(foldlist, lbllist):

                        folds.append(preprocess_ranks(folddir, colname='score', maxsz=8000))
                        labels.append(np.load(lblpath))

                    nf = len(foldlist)

                    if backend == "r":
                        #new_wbl = WeibullMR_R(k=k, method=method, opt_metric=opt, notop=notop, verbose=False)
                        raise ValueError("R backend is non-functional!")
                    elif backend == "matlab":
                        new_wbl = WeibullMR_M(k=k, method=method, opt_metric=opt, notop=notop, verbose=False)
                    else:
                        raise ValueError("Invalid backend <{0:s}> for Weibull MR".format(backend))

                    for f in range(nf):

                        wblpath = "{0:s}weibull-fixed_f{1:03d}.wbl".format(outdir, f)
                        try:
                            with open(wblpath, 'rb') as inpf:
                                wbl = pickle.load(inpf)
                        except FileNotFoundError:
                            wbl = new_wbl
                            train_X = np.vstack(folds[0:f] + folds[f+1:])
                            train_y = np.vstack(labels[0:f] + labels[f + 1:])

                            assert train_X.shape[0] == train_y.shape[0], "Number of training samples and labels do not" \
                                                                         "match"

                            if train_X.shape[0] >= 1000:
                                samplidx = np.arange(0, train_X.shape[0], dtype=np.int32)
                                nsamp = int(np.floor(sampling*train_X.shape[0]))

                                np.random.shuffle(samplidx)
                                samplidx = samplidx[0:nsamp]

                                wbl.fit(train_X[samplidx, :], train_y[samplidx, :],
                                        f_val=(0.1, 0.85, 0.1), z_val=(0.2, 1.1, 0.2))

                            else:
                                wbl.fit(train_X, train_y, f_val=(0.1, 0.85, 0.1), z_val=(0.2, 1.1, 0.2))

                        outfile = "{0:s}rel-prediction/{1:s}_f{2:03d}_top{3:d}_irp.npy".format(outdir, dkey, f, k)
                        print("    -> ", os.path.basename(outfile), "...", end="", flush=True)

                        test_X = folds[f]
                        predicted, _ = wbl.predict(test_X)

                        np.save(outfile, predicted)

                        with open(wblpath, 'wb') as outf:
                            pickle.dump(wbl, outf)
                        print(" Done!")


    def run_learning_mr(self, expconfig):

        expcfg = cfgloader(expconfig)

        expname = expcfg['DEFAULT']['expname']
        cname = expcfg['IRP']['classifier']
        k = expcfg['DEFAULT'].getint('topk')

        predicted = []

        for dataset in self.__expmap:
            for descnum in self.__expmap[dataset]:
                dkey = "{0:s}_desc{1:d}".format(dataset, descnum)

                print(". Running <Learning MR - IRP> on", dataset, " -- descriptor", descnum)
                print(". Experiment name: ", expname)

                featdir = self.__pathcfg['feature'][dkey] + expname + "/"
                lbldir = self.__pathcfg['label'][dkey] + "/rel-prediction/"

                outdir = self.__pathcfg['output'][dkey] + "{0:s}/rel-prediction/".format(expname)
                safe_create_dir(outdir)

                labels = load_labels(lbldir)
                fn = len(labels)  # Number of folds

                # Top-k features
                for m in range(0, k):
                    print("  -> Classifying Rank -", m+1)
                    features = load_features(featdir, "{0:d}".format(m+1))
                    print("        -> Feat shape:", [f.shape for f in features])
                    predicted_r = run_classification(features, labels, cname, scale=True, M=m)
                    if not predicted:
                        predicted = predicted_r
                    else:
                        for i, p in enumerate(predicted_r):
                            predicted[i] = np.hstack([predicted[i], p])

                for f in range(fn):
                    nlbl = labels[f].shape[0]
                    assert predicted[f].shape == labels[f].shape
                    outfile = "{0:s}{1:s}_f{2:03d}_top{3:d}_irp.npy".format(outdir, dkey, f, k)
                    print("  -> Saving:", outfile)
                    print("      |-> Shape:", predicted[f].shape)

                    np.save(outfile, predicted[f])

                predicted = []

        return







