#!/usr/bin/env python
#-*- coding: utf-8 -*-


import sys, os
import glob

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix

#import ipdb as pdb

from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir
from rankutils.drawing import heatmap
from rankutils.mappings import exkey_map


def parsecolor(colorstr):

    from rankutils.drawing import colors_from_cmap

    components = colorstr.split(',')

    if len(components) == 2:
        cmapname = components[0]
        cmapval = float(components[1])

        color = colors_from_cmap(cmapname, cmapval)[0]

    elif len(components) == 3:
        color = tuple([float(c) / 255.0 for c in components] + [1.0])

    elif len(components) == 4:
        color = tuple([float(c) / 255.0 for c in components])
    else:
        raise ValueError("The \'color\' configuration parameter must be a string in the form: <colormap>,<value> or"
                         "<r>,<g>,<b> or <r>,<g>,<b>,<a>")

    return color



def norm_acc(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    cfmat = confusion_matrix(y_true, y_pred)

    if cfmat.size == 1:
        nacc = 1.0

    else:
        TN = cfmat[0, 0]
        FN = cfmat[1, 0]

        FP = cfmat[0, 1]
        TP = cfmat[1, 1]

        if (TN + FP) != 0:
            TNR = TN / (TN + FP)
        else:
            TNR = 0.0

        if (TP + FN) != 0:
            TPR = TP / (TP + FN)
        else:
            TPR = 0.0


        nacc = (TNR + TPR) / 2

    return nacc

class Evaluator:

    # Measure = (<Name of measure>, <measure index in results>, <bottom limit of measure>)
    measure_map = dict(ACC=('accuracy', 0, 0.0),
                       NACC=('norm accuracy', 1, 0.0),
                       F1=('f1 score', 2, 0.0),
                       MCC=('matthews correlation coefficient', 3, -1.0))

    def __init__(self, evalcfgfile="", key="", pathcfg=None):

        self.key = key

        if self.key and pathcfg:
            self.lblpath = pathcfg['label'][self.key]
            self.outpath = pathcfg['output'][self.key]
            self.respath = pathcfg['result'][self.key]
            aux = glob.glob(pathcfg['rank'][self.key] + '/*.folds.npy')[0]
            self.foldidx = np.load(aux)

        self.evalname = ""

        self.__data = []

        self.__gt_irp_labels = []

        self.__k = 10
        self.__draw_params = dict()
        self.__s = 0
        self.__e = self.__k-1

        if evalcfgfile:
            self.load_configurations(evalcfgfile)

            if self.key and self.lblpath and self.outpath and self.respath:
                self.load_results_data()

    @property
    def data(self):
        return self.__data

    @property
    def gt_irp_labels(self):
        return self.__gt_irp_labels

    def load_configurations(self, evalcfgfile):

        evalcfg = cfgloader(evalcfgfile)

        self.evalname = evalcfg['DEFAULT']['evalname']

        self.__k = evalcfg.getint('DEFAULT', 'k')

        self.__s = evalcfg.getint('DEFAULT', 'lowM') - 1
        if self.__s < 0:
            self.__s = 0

        self.__e = evalcfg.getint('DEFAULT', 'highM')
        if self.__e > self.__k:
            self.__e = self.__k - 1

        self.__draw_params.setdefault("legend_columns", evalcfg.getint('DEFAULT', 'legend_columns'))

        self.respath += "{0:s}/".format(self.evalname)

        sect = evalcfg.sections()

        for s in sect:

            m = dict()
            params = dict()
            drawargs = dict()

            m['name'] = evalcfg[s]['name']
            params['add_correlation'] = evalcfg[s].getboolean('add_correlation', fallback=False)
            params['add_rpp'] = evalcfg[s].getboolean('add_rpp', fallback=False)
            params['label'] = evalcfg[s]['label']
            params['plot_type'] = evalcfg[s]['plot_type']

            drawargs['color'] = parsecolor(evalcfg.get(s, 'color'))
            drawargs['marker'] = evalcfg.get(s, 'marker', fallback='')
            drawargs['linestyle'] = evalcfg.get(s, 'linestyle', fallback='-')
            drawargs['markersize'] = evalcfg.getint(s, 'markersize', fallback=9)
            drawargs['markeredgewidth'] = evalcfg.getint(s, 'markeredgewidth', fallback=2)
            drawargs['linewidth'] = evalcfg.getint(s, 'linewidth', fallback=2)

            if 'markerfacecolor' in evalcfg[s].keys():
                drawargs['markerfacecolor'] = parsecolor(evalcfg.get(s, 'markerfacecolor'))
            else:
                drawargs['markerfacecolor'] = drawargs['color']

            m['params'] = params
            m['drawargs'] = drawargs

            self.__data.append(m)

    def load_results_data(self):

        n, rds = self.foldidx.shape

        # Loading Groundtruth labels

        irp_lbl = np.load(glob.glob(self.lblpath + '*irp*')[0])

        self.__gt_irp_labels = []

        for r in range(rds):
            idx_0 = np.flatnonzero(self.foldidx[:, r] == 0).reshape(-1)
            idx_1 = np.flatnonzero(self.foldidx[:, r] == 1).reshape(-1)

            self.__gt_irp_labels.append(irp_lbl[idx_0, :])
            self.__gt_irp_labels.append(irp_lbl[idx_1, :])

        ##

        assert self.__gt_irp_labels, "Empty IRP Groundtruth Labels"

        # Loading Predicted Labels
        for mdata in self.__data:

            nm = mdata['name']

            irp_flist = glob.glob(self.outpath + "{0:s}/*.npy".format(nm))
            irp_flist.sort()

            mdata['irp_results'] = list(map(np.load, irp_flist))

            assert mdata['irp_results'], "Empty IRP Predicted Labels for method {0:s}".format(mdata['name'])

    def evaluate(self):

        np.set_printoptions(linewidth=300, precision=4)
        np.seterr(divide='ignore', invalid='ignore')
        T_vals = np.arange(1, 11, 1).reshape(1, -1)

        for mdata in self.__data:
            #print("Evaluating:", mdata['name'])
            # Individual Rank Position (irp) Evaluation
            irp_evaluation = []
            #irp_evaluation_sample = []
            rpp_evaluation = []
            patk_prediction = []

            pos_evaluation = []

            predicted_counts = []

            irp_pred_labels = mdata['irp_results']

            teval = len(mdata['irp_results'])

            for i in range(teval):

                irp_true_labels_single = self.__gt_irp_labels[i]
                irp_pred_labels_single = irp_pred_labels[i]

                assert irp_true_labels_single.shape == irp_pred_labels_single.shape, "Inconsistent shapes between true" \
                                                                                     "and predicted labels"

                nex = irp_true_labels_single.shape[0]

                pos_nacc = np.zeros(self.__k, dtype=np.float64)
                for k in range(self.__k):
                    true_pos_labels = irp_true_labels_single[:, k]
                    pred_pos_labels = irp_pred_labels_single[:, k]

                    pos_nacc[k] = norm_acc(true_pos_labels.reshape(-1), pred_pos_labels.reshape(-1))

                pos_evaluation.append(pos_nacc)

                predicted_counts.append(np.bincount(np.sum(irp_pred_labels_single, axis=1).astype(np.int32),
                                                    minlength=self.__k+1))

                # Evaluating IRP -- ALL INSTANCES
                irp_acc = accuracy_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_nacc = norm_acc(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_mcc = matthews_corrcoef(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_f1 = f1_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))

                irp_evaluation.append([irp_acc, irp_nacc, irp_f1, irp_mcc])
                ###

                # Evaluating IRP -- PER INSTANCE
                irp_acc_s = np.zeros(nex, dtype=np.float64)
                irp_nacc_s = np.zeros(nex, dtype=np.float64)
                irp_mcc_s = np.zeros(nex, dtype=np.float64)
                irp_f1_s = np.zeros(nex, dtype=np.float64)

                #for j in range(nex):
                    #irp_acc_s[j] = accuracy_score(irp_true_labels_single[j], irp_pred_labels_single[j])
                    #irp_nacc_s[j] = norm_acc(irp_true_labels_single[j], irp_pred_labels_single[j])
                    #irp_mcc_s[j] = matthews_corrcoef(irp_true_labels_single[j], irp_pred_labels_single[j])
                    #irp_f1_s[j] = 1.0

                #irp_evaluation_sample.append([np.mean(irp_acc_s),
                #                              np.mean(irp_nacc_s),
                #                              np.mean(irp_f1_s),
                #                              np.mean(irp_mcc_s)])
                ###

                # Evaluating PATK for round 0
                if i == 0 or i == 1:

                    pred_patk = irp_pred_labels_single.sum(axis=1).reshape(-1, 1).astype(np.int32)
                    true_patk = irp_true_labels_single.sum(axis=1).reshape(-1, 1).astype(np.int32)

                    patk_prediction.append(np.hstack([pred_patk, true_patk]))
                ###

                # Evaluating RPP
                rpp_pred_labels_single = (irp_pred_labels_single.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)
                rpp_true_labels_single = (irp_true_labels_single.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)

                rpp_nacc = np.array([norm_acc(rpp_true_labels_single[:, x], rpp_pred_labels_single[:, x])
                                     for x in range(0, T_vals.size)])

                rpp_evaluation.append(rpp_nacc)

                ###


            #for t in irp_evaluation: print(t)
            irp_evaluation = np.vstack(irp_evaluation)
            #irp_evaluation_sample = np.vstack(irp_evaluation_sample)
            pos_evaluation = np.vstack(pos_evaluation)
            patk_prediction = np.vstack(patk_prediction)
            rpp_evaluation = np.vstack(rpp_evaluation)
            predicted_counts = np.vstack(predicted_counts)



            mdata['irp_evaluation'] = np.vstack([irp_evaluation, np.mean(irp_evaluation, axis=0).reshape(1, -1)])
            #mdata['irp_evaluation_sample'] = np.vstack([irp_evaluation_sample,
            #                                            np.mean(irp_evaluation_sample, axis=0).reshape(1, -1)])
            mdata['pos_evaluation'] = np.vstack([pos_evaluation, np.mean(pos_evaluation, axis=0).reshape(1, -1)])
            mdata['predicted_counts'] = np.vstack([predicted_counts, np.mean(predicted_counts, axis=0).reshape(1, -1)])
            mdata['rpp_evaluation'] = np.vstack([rpp_evaluation, np.mean(rpp_evaluation, axis=0).reshape(1, -1)])
            mdata['patk_prediction'] = patk_prediction

            #pdb.set_trace()
            np.seterr(divide='warn', invalid='warn')
            ##

    def write_results(self, fold=-1, outprefix=""):

        irp_eval = []

        safe_create_dir(self.respath)

        if outprefix == "":
            outprefix = self.evalname + ".{0:s}.".format(self.key)

        if fold == -1:
            outfilename = "{0:s}/{2:s}all_".format(self.respath, self.evalname, outprefix)
        else:
            outfilename = "{0:s}/{2:s}_fold{3:03d}_".format(self.respath, self.evalname, outprefix, fold)

        for mdata in self.data:

            irp_eval.append((mdata['params']['label'], mdata['irp_evaluation'][-1, :]))

        with open(outfilename + "irp.csv", 'w') as outf:
            outf.write('label, acc, norm acc, f1, mcc\n')
            for tpl in irp_eval:
                outf.write(tpl[0])
                for v in tpl[1]:  outf.write(",{0:0.4f}".format(v))
                outf.write('\n')


