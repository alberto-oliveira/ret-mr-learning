#!/usr/bin/env python
#-*- coding: utf-8 -*-


import sys, os
import glob

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix

import ipdb as pdb

from common.cfgloader import *
from common.utilities import safe_create_dir

parsecolor = lambda cstr: tuple([float(c)/255.0 for c in cstr.split(',')])

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

        #self.__fig, self.__axes = plt.subplots(3, 3, figsize=(20, 10))
        #plt.subplots_adjust(hspace=0.1, top=1.0, bottom=0.8)

        self.__data = []

        self.__gt_irp_labels = []
        self.__gt_rpp_labels = []

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

    @property
    def gt_rpp_labels(self):
        return self.__gt_rpp_labels

    def load_configurations(self, evalcfgfile):

        evalcfg = cfgloader(evalcfgfile)

        self.evalname = evalcfg['DEFAULT']['evalname']

        self.__k = evalcfg['DEFAULT'].getint('k')

        self.__s = evalcfg['DEFAULT'].getint('lowM') - 1
        if self.__s < 0:
            self.__s = 0

        self.__e = evalcfg['DEFAULT'].getint('highM')
        if self.__e > self.__k:
            self.__e = self.__k - 1

        self.respath += "{0:s}/".format(self.evalname)

        sect = evalcfg.sections()

        for s in sect:

            m = dict()
            params = dict()

            m['name'] = evalcfg[s]['name']
            params['label'] = evalcfg[s]['label']
            params['marker'] = evalcfg[s]['marker']
            params['linestyle'] = evalcfg[s]['linestyle']

            params['markersize'] = evalcfg[s].getint('markersize')
            params['markeredgesize'] = evalcfg[s].getint('markeredgesize')
            params['linewidth'] = evalcfg[s].getint('linewidth')

            params['color'] = parsecolor(evalcfg[s]['color'])
            m['params'] = params

            self.__data.append(m)

    def load_results_data(self):

        n, rds = self.foldidx.shape

        # Loading Groundtruth labels

        irp_lbl = np.load(glob.glob(self.lblpath + '*irp*')[0])
        rpp_lbl = np.load(glob.glob(self.lblpath + '*rpp*')[0])

        self.__gt_irp_labels = []
        self.__gt_rpp_labels = []

        for r in range(rds):
            idx_0 = np.argwhere(self.foldidx[:, r] == 0).reshape(-1)
            idx_1 = np.argwhere(self.foldidx[:, r] == 1).reshape(-1)

            self.__gt_irp_labels.append(irp_lbl[idx_0, :])
            self.__gt_irp_labels.append(irp_lbl[idx_1, :])

            self.__gt_rpp_labels.append(rpp_lbl[idx_0, self.__s:self.__e])
            self.__gt_rpp_labels.append(rpp_lbl[idx_1, self.__s:self.__e])

        ##

        assert self.__gt_irp_labels, "Empty IRP Groundtruth Labels"
        assert self.__gt_rpp_labels, "Empty RPP Groundtruth Labels"

        # Loading Predicted Labels
        for mdata in self.__data:

            #pdb.set_trace()
            nm = mdata['name']

            irp_flist = glob.glob(self.outpath + "{0:s}/rel-prediction/*.npy".format(nm))
            rpp_flist = glob.glob(self.outpath + "{0:s}/perf-prediction/*.npy".format(nm))

            irp_flist.sort()
            rpp_flist.sort()

            mdata['irp_results'] = list(map(np.load, irp_flist))
            mdata['rpp_results'] = list(map(np.load, rpp_flist))

            assert mdata['irp_results'], "Empty IRP Predicted Labels for method {0:s}".format(mdata['name'])
            assert mdata['rpp_results'], "Empty RPP Predicted Labels for method {0:s}".format(mdata['name'])

        #

    def evaluate(self):

        np.seterr(divide='ignore', invalid='ignore')
        for mdata in self.__data:
            #print("Evaluating:", mdata['name'])
            # Individual Rank Position (irp) Evaluation
            irp_evaluation = []

            irp_pred_labels = mdata['irp_results']

            teval = len(mdata['irp_results'])

            for k in range(teval):

                irp_true_labels_single = self.__gt_irp_labels[k]
                irp_pred_labels_single = irp_pred_labels[k]

                #print("true shape:", irp_true_labels_fold.shape)
                #print("pred shape:", irp_pred_labels_fold.shape)

                #np.seterr(all='raise')
                irp_acc = accuracy_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_nacc = Evaluator.norm_acc(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_mcc = matthews_corrcoef(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_f1 = f1_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                #np.seterr(all='ignore')

                irp_evaluation.append([irp_acc, irp_nacc, irp_f1, irp_mcc])

            #for t in irp_evaluation: print(t)
            irp_evaluation = np.vstack(irp_evaluation)
            mdata['irp_evaluation'] = np.vstack([irp_evaluation, np.mean(irp_evaluation, axis=0).reshape(1, -1)])
            ##

            ## Rank Performance Prediction (rpp) Evaluation
            rpp_acc_evaluation = []
            rpp_nacc_evaluation = []
            rpp_f1_evaluation = []
            rpp_mcc_evaluation = []

            rpp_pred_labels = mdata['rpp_results']

            for k in range(teval):

                #print(" |_ RPP Evaluation: Fold #{0:d}".format(k))
                rpp_true_labels_single = self.__gt_rpp_labels[k]
                rpp_pred_labels_single = rpp_pred_labels[k]

                assert rpp_pred_labels_single.shape == rpp_true_labels_single.shape,\
                       "Inconsistent shapes between RPP predicted labels {0:s} and true labels {1:s}, at fold {2:d}"\
                       .format(str(rpp_pred_labels_single.shape), str(rpp_true_labels_single.shape), k)

                acc_m = []
                nacc_m = []
                f1_m = []
                mcc_m = []

                #print(rpp_true_labels_fold.shape)
                #print(rpp_pred_labels_fold.shape)
                for c in range(rpp_pred_labels_single.shape[1]):

                    #print("    -> M = {0:d}".format(c+2))

                    true_l = rpp_true_labels_single[:, c].reshape(-1)
                    pred_l = rpp_pred_labels_single[:, c].reshape(-1)

                    #print("--\n", rpp_true_labels_fold.shape)
                    #print(rpp_pred_labels_fold.shape)

                    #print("   {0:<20s}".format("groundtruth:"), true_l, file=sys.stdout, flush=True)
                    #print("   {0:<20s}".format("predicted:"), pred_l, end="\n\n", file=sys.stdout, flush=True)

                    acc_m.append(accuracy_score(true_l, pred_l))
                    nacc_m.append(Evaluator.norm_acc(true_l, pred_l))
                    f1_m.append(f1_score(true_l, pred_l))
                    mcc_m.append(matthews_corrcoef(true_l, pred_l))

                rpp_acc_evaluation.append(acc_m)
                rpp_nacc_evaluation.append(nacc_m)
                rpp_f1_evaluation.append(f1_m)
                rpp_mcc_evaluation.append(mcc_m)

            rpp_acc_evaluation = np.vstack(rpp_acc_evaluation)
            rpp_nacc_evaluation = np.vstack(rpp_nacc_evaluation)
            rpp_f1_evaluation = np.vstack(rpp_f1_evaluation)
            rpp_mcc_evaluation = np.vstack(rpp_mcc_evaluation)

            mdata['rpp_evaluation'] = dict(acc=np.vstack([rpp_acc_evaluation, np.mean(rpp_acc_evaluation, axis=0)]),
                                           normacc=np.vstack([rpp_nacc_evaluation, np.mean(rpp_nacc_evaluation, axis=0)]),
                                           f1=np.vstack([rpp_f1_evaluation, np.mean(rpp_f1_evaluation, axis=0)]),
                                           mcc=np.vstack([rpp_mcc_evaluation, np.mean(rpp_mcc_evaluation, axis=0)]))
            np.seterr(divide='warn', invalid='warn')
            ##

    def write_results(self, fold=-1, outprefix=""):

        irp_eval = []
        rpp_eval_acc = []
        rpp_eval_f1 = []
        rpp_eval_mcc = []

        safe_create_dir(self.respath)

        if outprefix == "":
            outprefix = self.evalname + ".{0:s}.".format(self.key)

        if fold == -1:
            outfilename = "{0:s}/{2:s}all_".format(self.respath, self.evalname, outprefix)
        else:
            outfilename = "{0:s}/{2:s}_fold{3:03d}_".format(self.respath, self.evalname, outprefix, fold)

        for mdata in self.data:

            irp_eval.append((mdata['params']['label'], mdata['irp_evaluation'][-1, :]))
            rpp_eval_acc.append((mdata['params']['label'], mdata['rpp_evaluation']['acc'][-1, :]))
            rpp_eval_f1.append((mdata['params']['label'], mdata['rpp_evaluation']['f1'][-1, :]))
            rpp_eval_mcc.append((mdata['params']['label'], mdata['rpp_evaluation']['mcc'][-1, :]))

        with open(outfilename + "irp.csv", 'w') as outf:
            outf.write('label, acc, norm acc, f1, mcc\n')
            for tpl in irp_eval:
                outf.write(tpl[0])
                for v in tpl[1]:  outf.write(",{0:0.4f}".format(v))
                outf.write('\n')

        # We use self.__s + 1 and self.__e + 1 to convert them from indexes to values of M.
        with open(outfilename + "rpp_acc.csv", 'w') as outf:
            outf.write('label, ' + str([x for x in range(self.__s+1, self.__e+1)]).strip('[]') + '\n')
            for tpl in rpp_eval_acc:
                outf.write(tpl[0])
                for v in tpl[1]:  outf.write(",{0:0.4f}".format(v))
                outf.write('\n')

        with open(outfilename + "rpp_f1.csv", 'w') as outf:
            outf.write('label, ' + str([x for x in range(self.__s + 1, self.__e + 1)]).strip('[]') + '\n')
            for tpl in rpp_eval_f1:
                outf.write(tpl[0])
                for v in tpl[1]:  outf.write(",{0:0.4f}".format(v))
                outf.write('\n')

        with open(outfilename + "rpp_mcc.csv", 'w') as outf:
            outf.write('label, ' + str([x for x in range(self.__s + 1, self.__e + 1)]).strip('[]') + '\n')
            for tpl in rpp_eval_mcc:
                outf.write(tpl[0])
                for v in tpl[1]:  outf.write(",{0:0.4f}".format(v))
                outf.write('\n')

    @staticmethod
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

            TNR = TN/(TN + FP)
            TPR = TP/(TP + FN)

            nacc = (TNR + TPR)/2

        return nacc


    def draw_irp_results(self, outprefix="", measure='MCC', dbparams=None, outf=None):

        plt.figure()
        meas_name, ri, blim = Evaluator.measure_map[measure]

        nmet = len(self.data)

        fw = 0.1    # space betweem limiting border and axis
        sw = 0.025  # space between bars
        bw = 0.25   # bar width

        rects = dict()

        x = fw
        for i, mdata in enumerate(self.data):

            val = mdata['irp_evaluation'][-1, ri]

            #print(mdata['params']['color'])

            rect = plt.bar(x, val, bw, 0, align='edge', label=mdata['params']['label'], color=mdata['params']['color'])
            rects[mdata['name']] = rect

            posx = rect[0].get_x()
            posy = rect[0].get_y()
            hgt = rect[0].get_height()
            wdt = rect[0].get_width()

            plt.text(posx + wdt/2, posy + hgt + 0.01, "{0:0.3f}".format(val), fontsize=16, bbox={'alpha': 0.0},
                     horizontalalignment='center')

            x += bw + sw

        max_x = fw + nmet*bw + (nmet-1)*sw + fw
        plt.xlim(0.0, max_x)

        if measure != 'MCC':
            if dbparams:
                p10 = dbparams[self.key].getfloat('p10')
                line = plt.plot([0.0, 4.0], [p10, p10], 'r--', linewidth=2)
            plt.ylim(bottom=blim, top=1.0)
            leg_ypos = 0.10
            plt.yticks([float(x) / 10 for x in range(0, 11, 1)])
        else:
            plt.ylim(bottom=blim, top=1.0)
            leg_ypos = 0.1
            plt.yticks([float(x) / 10 for x in range(-10, 11, 2)])

        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

        plt.ylabel(measure)
        plt.grid(True, axis='y')
        plt.title("Individual Rank Prediction -- {0:s}\n{1:s}".format(measure, self.key),
                      fontdict={'fontsize': 18, 'horizontalalignment': 'center'})

        if nmet > 3:
            nleg_col = nmet-2
        else:
            nleg_col = nmet
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, leg_ypos), bbox_transform=plt.gcf().transFigure,
                   fancybox=True, shadow=True, ncol=nleg_col)

        safe_create_dir(self.respath)

        if not outf:
            if outprefix == "":
                 outprefix = self.evalname + ".{0:s}.".format(self.key)

            outfilename = "{0:s}/{2:s}_irp_{3:s}.pdf".format(self.respath, self.evalname, outprefix, measure)

            plt.savefig(outfilename)

        else:
            outf.savefig()
        plt.close()

























