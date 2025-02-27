#!/usr/bin/env python
#-*- coding: utf-8 -*-


import sys, os
import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix

from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir

# Measure = (<Name of measure>, <measure index in results>, <bottom limit of measure>)
measure_map = dict(ACC=('accuracy', 0, 0.0),
                   NACC=('norm accuracy', 0, 0.0),
                   F1=('f1 score', 1, 0.0),
                   MCC=('matthews correlation coefficient', 2, -1.0))


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
                         "<r>,<g>,<b> or <r>,<g>,<b>,<a>. In RGB or RGBA the range is 0-255")

    return color

def multi_f1(y_true, y_pred):

    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    nrows = y_true.shape[0]
    f1scores = np.zeros(nrows, dtype=np.float64)

    for i in range(nrows):

        y_true_row = y_true[i]
        y_pred_row = y_pred[i]

        f = f1_score(y_true_row, y_pred_row)

        f1scores[i] = f

    return f1scores, np.mean(f1scores)


def multi_mcc(y_true, y_pred):

    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    nrows = y_true.shape[0]
    mccs = np.zeros(nrows, dtype=np.float64)

    for i in range(nrows):

        y_true_row = y_true[i]
        y_pred_row = y_pred[i]

        m = matthews_corrcoef(y_true_row, y_pred_row)

        mccs[i] = m

    return mccs, np.mean(mccs)


def multi_norm_acc(y_true, y_pred, get_rates=False):

    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    nrows = y_true.shape[0]
    naccs = np.zeros(nrows, dtype=np.float64)
    tprs = np.zeros(nrows, dtype=np.float64)
    tnrs = np.zeros(nrows, dtype=np.float64)

    for i in range(nrows):

        y_true_row = y_true[i]
        y_pred_row = y_pred[i]

        nacc, tpr, tnr = norm_acc(y_true_row, y_pred_row, True)

        naccs[i] = nacc
        tprs[i] = tpr
        tnrs[i] = tnr

    if not get_rates:
        return naccs, np.mean(naccs)
    else:
        return naccs, np.mean(naccs), tprs, np.mean(tprs), tnrs, np.mean(tnrs)


def norm_acc(y_true, y_pred, get_rates=False):
    assert y_true.shape == y_pred.shape, "Inconsistent shapes between true labels <{0:s}> and predicted " \
                                         "labels <{1:s}>.".format(str(y_true.shape), str(y_pred.shape))

    cfmat = confusion_matrix(y_true, y_pred)

    if cfmat.size == 1:
        nacc = 1.0
        TPR = 1.0
        TNR = 1.0

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

        if TN + FP == 0:
            nacc = TPR
        elif TP + FN == 0:
            nacc = TNR
        else:
            nacc = (TNR + TPR) / 2

    if not get_rates:
        return nacc
    else:
        return nacc, TPR, TNR


def comp_recall(relevance):
    total = relevance.sum()
    recall = np.add.accumulate(relevance) / total

    # The np.add.accumulate does not get recall value 0.0, so I stack it
    return np.concatenate([np.zeros(1, dtype=np.float64), recall.reshape(-1)])


def comp_precision(relevance):
    # 1st create a vector full of ones the same size of relevance
    # 2nd accumulate the addition of the ones above
    div = np.add.accumulate(np.ones_like(relevance))
    precision = np.add.accumulate(relevance) / div

    return np.concatenate([np.ones(1, dtype=np.float64), precision.reshape(-1)])


def interp_prec_recall_curve(precision, recall, points=np.array([])):
    if points.size == 0:
        points = recall

    i_precision = np.zeros(points.size)

    for i, r in enumerate(points):
        idx = recall >= r
        i_precision[i] = precision[idx].max()

    return i_precision


class Evaluator:

    def __init__(self, evalcfgfile="", key="", pathcfg=None):

        self.key = key

        if isinstance(pathcfg, str):
            paths = cfgloader(pathcfg)
        elif isinstance(pathcfg, configparser.ConfigParser):
            paths = pathcfg

        if self.key and pathcfg:

            self.rktpalias = paths[self.key]['rktpdir']
            self.rkpath = paths[self.key]['rank']
            self.lblpath = paths[self.key]['label']
            self.outpath = paths[self.key]['output']
            self.respath = paths[self.key]['result']

        self.evalname = ""

        self.__data = []

        self.__gt_irp_labels = []

        self.__k = 10
        self.__draw_params = dict()

        if evalcfgfile:
            self.load_configurations(evalcfgfile)

            if self.key and self.lblpath and self.outpath and self.respath:
                self.load_results_data_v2()

    @property
    def k(self):
        return self.__k

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

        self.__draw_params.setdefault("legend_columns", evalcfg.getint('DEFAULT', 'legend_columns'))

        self.respath += "{0:s}/".format(self.evalname)

        sect = evalcfg.sections()

        for s in sect:

            m = dict()
            params = dict()
            drawargs = dict()

            m['name'] = s
            params['label'] = evalcfg.get(s, 'label')
            params['plot_type'] = evalcfg.get(s, 'plot_type', fallback='bar')

            drawargs['color'] = parsecolor(evalcfg.get(s, 'color'))
            drawargs['marker'] = evalcfg.get(s, 'marker', fallback='')
            drawargs['linestyle'] = evalcfg.get(s, 'linestyle', fallback='-')
            drawargs['markersize'] = evalcfg.getint(s, 'markersize', fallback=9)
            drawargs['markeredgewidth'] = evalcfg.getint(s, 'markeredgewidth', fallback=2)
            drawargs['linewidth'] = evalcfg.getint(s, 'linewidth', fallback=2)
            drawargs['alpha'] = evalcfg.getfloat(s, 'alpha', fallback=1.0)

            if 'markerfacecolor' in evalcfg[s].keys():
                drawargs['markerfacecolor'] = parsecolor(evalcfg.get(s, 'markerfacecolor'))
            else:
                drawargs['markerfacecolor'] = drawargs['color']

            m['params'] = params
            m['drawargs'] = drawargs

            self.__data.append(m)

    def load_results_data(self):

        aux = glob.glob(self.rkpath + '/*folds.npy')[0]
        self.foldidx = np.load(aux)
        n, rds = self.foldidx.shape

        # Loading Groundtruth labels

        irp_lbl = np.load(glob.glob(self.lblpath + '*{0:s}*'.format(self.rktpalias))[0])
        irp_lbl = irp_lbl[:, 0:self.__k]

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

            irp_flist = glob.glob(self.outpath + "{0:s}/*irp.npy".format(nm))
            irp_flist.sort()

            mdata['irp_results'] = list(map(np.load, irp_flist))

            assert mdata['irp_results'], "Empty IRP Predicted Labels at: " + self.outpath + "{0:s}/".format(nm)

            for i in range(len(mdata['irp_results'])):
                mdata['irp_results'][i] = mdata['irp_results'][i][:, 0:self.__k]

    def load_results_data_v2(self):

        # Loading Groundtruth labels

        irp_lbl = np.load(glob.glob(self.lblpath + '*{0:s}*'.format(self.rktpalias))[0])
        self.__gt_irp_labels = irp_lbl[:, 0:self.__k]

        assert self.__gt_irp_labels.size > 0, "Empty IRP Groundtruth Labels"

        # Loading Predicted Labels
        for mdata in self.__data:

            nm = mdata['name']

            irp_f = glob.glob(self.outpath + "{0:s}/*irp.npy".format(nm))

            mdata['irp_results'] = np.load(irp_f[0])

            assert mdata['irp_results'].size > 0, "Empty IRP Predicted Labels at: " + self.outpath + "{0:s}/".format(nm)
            assert mdata['irp_results'].shape[1] == self.__gt_irp_labels.shape[0], \
                  "Inconsistent number of labels in results data <{0:d}> " \
                  "and groundtruth data <{1:d}>".format(mdata['irp_results'].shape[1], self.__gt_irp_labels.shape[0])

            mdata['irp_results'] = mdata['irp_results'][:, :, 0:self.__k]

    def evaluate_v2(self):

        np.set_printoptions(linewidth=300, precision=4)
        np.seterr(divide='ignore', invalid='ignore')
        T_vals = np.arange(1, 11, 1).reshape(1, -1)

        for mdata in self.__data:
            #print("Evaluating:", mdata['name'])
            # Individual Rank Position (irp) Evaluation
            irp_evaluation = []
            rpp_evaluation = []

            pos_evaluation_nacc = []
            pos_evaluation_f1 = []

            irp_pred_labels = mdata['irp_results']

            rounds = mdata['irp_results'].shape[0]

            for i in range(rounds):

                irp_pred_labels_round = irp_pred_labels[i]

                assert self.__gt_irp_labels.shape == irp_pred_labels_round.shape, \
                       "Inconsistent shapes between true <{0:s}> and predicted <{1:s}> " \
                       "labels".format(str(self.__gt_irp_labels.shape), str(irp_pred_labels_round.shape))

                # multi_norm_acc operates over rows. Rows, in the original arrays, are samples, while columns are
                # positions. To evaluate the positional norm acc, transpose should be used to turn positions into rows
                pos_nacc, _ = multi_norm_acc(self.__gt_irp_labels.transpose(), irp_pred_labels_round.transpose())
                pos_f1, _ = multi_f1(self.__gt_irp_labels.transpose(), irp_pred_labels_round.transpose())

                pos_evaluation_nacc.append(pos_nacc.reshape(-1))
                pos_evaluation_f1.append(pos_f1.reshape(-1))

                # Evaluating IRP -- ALL INSTANCES
                irp_nacc = norm_acc(self.__gt_irp_labels.reshape(-1), irp_pred_labels_round.reshape(-1))
                irp_mcc = matthews_corrcoef(self.__gt_irp_labels.reshape(-1), irp_pred_labels_round.reshape(-1))
                irp_f1 = f1_score(self.__gt_irp_labels.reshape(-1), irp_pred_labels_round.reshape(-1))

                irp_evaluation.append([irp_nacc, irp_f1, irp_mcc])
                ###

                # Evaluating RPP
                rpp_pred_labels_single = (irp_pred_labels_round.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)
                rpp_true_labels_single = (self.__gt_irp_labels.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)

                rpp_nacc = np.array([norm_acc(rpp_true_labels_single[:, x], rpp_pred_labels_single[:, x])
                                     for x in range(0, T_vals.size)])

                rpp_evaluation.append(rpp_nacc)

                ###

            irp_evaluation = np.vstack(irp_evaluation)
            pos_evaluation_nacc = np.vstack(pos_evaluation_nacc)
            pos_evaluation_f1 = np.vstack(pos_evaluation_f1)
            rpp_evaluation = np.vstack(rpp_evaluation)

            mdata['irp_evaluation'] = np.vstack([irp_evaluation, np.mean(irp_evaluation, axis=0).reshape(1, -1)])
            mdata['pos_evaluation_nacc'] = np.vstack([pos_evaluation_nacc, np.mean(pos_evaluation_nacc, axis=0).reshape(1, -1)])
            mdata['pos_evaluation_f1'] = np.vstack([pos_evaluation_f1, np.mean(pos_evaluation_f1, axis=0).reshape(1, -1)])
            mdata['rpp_evaluation'] = np.vstack([rpp_evaluation, np.mean(rpp_evaluation, axis=0).reshape(1, -1)])

            #pdb.set_trace()
            np.seterr(divide='warn', invalid='warn')
            ##

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

            pos_evaluation_nacc = []
            pos_evaluation_f1 = []

            predicted_counts = []

            irp_pred_labels = mdata['irp_results']

            teval = len(mdata['irp_results'])

            for i in range(teval):

                irp_true_labels_single = self.__gt_irp_labels[i]
                irp_pred_labels_single = irp_pred_labels[i]

                assert irp_true_labels_single.shape == irp_pred_labels_single.shape, "Inconsistent shapes between true <{0:s}> " \
                                                                                     "and predicted <{1:s}> labels".format(str(irp_true_labels_single.shape), str(irp_pred_labels_single.shape))

                nex = irp_true_labels_single.shape[0]

                pos_nacc = np.zeros(self.__k, dtype=np.float64)
                pos_f1 = np.zeros(self.__k, dtype=np.float64)
                for k in range(self.__k):
                    true_pos_labels = irp_true_labels_single[:, k]
                    pred_pos_labels = irp_pred_labels_single[:, k]

                    pos_nacc[k] = norm_acc(true_pos_labels.reshape(-1), pred_pos_labels.reshape(-1))
                    pos_f1[k] = f1_score(true_pos_labels.reshape(-1), pred_pos_labels.reshape(-1))

                pos_evaluation_nacc.append(pos_nacc)
                pos_evaluation_f1.append(pos_f1)

                predicted_counts.append(np.bincount(np.sum(irp_pred_labels_single, axis=1).astype(np.int32),
                                                    minlength=self.__k+1))

                # Evaluating IRP -- ALL INSTANCES
                irp_acc = accuracy_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_nacc = norm_acc(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_mcc = matthews_corrcoef(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_f1 = f1_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))

                irp_evaluation.append([irp_nacc, irp_f1, irp_mcc])
                ###

                # Evaluating IRP -- PER INSTANCE
                #irp_acc_s = np.zeros(nex, dtype=np.float64)
                #irp_nacc_s = np.zeros(nex, dtype=np.float64)
                #irp_mcc_s = np.zeros(nex, dtype=np.float64)
                #irp_f1_s = np.zeros(nex, dtype=np.float64)

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
            pos_evaluation_nacc = np.vstack(pos_evaluation_nacc)
            pos_evaluation_f1 = np.vstack(pos_evaluation_f1)
            patk_prediction = np.vstack(patk_prediction)
            rpp_evaluation = np.vstack(rpp_evaluation)
            predicted_counts = np.vstack(predicted_counts)



            mdata['irp_evaluation'] = np.vstack([irp_evaluation, np.mean(irp_evaluation, axis=0).reshape(1, -1)])
            #mdata['irp_evaluation_sample'] = np.vstack([irp_evaluation_sample,
            #                                            np.mean(irp_evaluation_sample, axis=0).reshape(1, -1)])
            mdata['pos_evaluation_nacc'] = np.vstack([pos_evaluation_nacc, np.mean(pos_evaluation_nacc, axis=0).reshape(1, -1)])
            mdata['pos_evaluation_f1'] = np.vstack([pos_evaluation_f1, np.mean(pos_evaluation_f1, axis=0).reshape(1, -1)])
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


