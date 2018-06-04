#!/usr/bin/env python
#-*- coding: utf-8 -*-


import sys, os
import glob

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix

import ipdb as pdb

from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir
from rankutils.drawing import heatmap
from rankutils.mappings import exkey_map

parsecolor = lambda cstr: tuple([float(c)/255.0 for c in cstr.split(',')])


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

            m['name'] = evalcfg[s]['name']
            params['label'] = evalcfg[s]['label']
            params['marker'] = evalcfg[s].get('marker', fallback='')
            params['plot_type'] = evalcfg[s]['plot_type']
            params['linestyle'] = evalcfg[s]['linestyle']

            params['markersize'] = evalcfg[s].getint('markersize')
            params['markeredgewidth'] = evalcfg[s].getint('markeredgewidth')
            params['linewidth'] = evalcfg[s].getint('linewidth')
            params['add_correlation'] = evalcfg[s].getboolean('add_correlation', fallback=False)
            params['add_rpp'] = evalcfg[s].getboolean('add_rpp', fallback=False)

            params['color'] = parsecolor(evalcfg[s]['color'])
            try:
                params['markerfacecolor'] = parsecolor(evalcfg[s]['markerfacecolor'])
            except KeyError as ke:
                params['markerfacecolor'] = params['color']
            try:
                params['linecolor'] = parsecolor(evalcfg[s]['linecolor'])
            except KeyError as ke:
                params['linecolor'] = params['color']
            m['params'] = params

            self.__data.append(m)

    def load_results_data(self):

        n, rds = self.foldidx.shape

        # Loading Groundtruth labels

        irp_lbl = np.load(glob.glob(self.lblpath + '*irp*')[0])

        self.__gt_irp_labels = []

        for r in range(rds):
            idx_0 = np.argwhere(self.foldidx[:, r] == 0).reshape(-1)
            idx_1 = np.argwhere(self.foldidx[:, r] == 1).reshape(-1)

            self.__gt_irp_labels.append(irp_lbl[idx_0, :])
            self.__gt_irp_labels.append(irp_lbl[idx_1, :])

        ##

        assert self.__gt_irp_labels, "Empty IRP Groundtruth Labels"

        # Loading Predicted Labels
        for mdata in self.__data:

            nm = mdata['name']

            irp_flist = glob.glob(self.outpath + "{0:s}/rel-prediction/*.npy".format(nm))

            irp_flist.sort()

            mdata['irp_results'] = list(map(np.load, irp_flist))

            assert mdata['irp_results'], "Empty IRP Predicted Labels for method {0:s}".format(mdata['name'])

    #def get_statistical_significance(self):



    def evaluate(self):

        np.seterr(divide='ignore', invalid='ignore')
        for mdata in self.__data:
            #print("Evaluating:", mdata['name'])
            # Individual Rank Position (irp) Evaluation
            irp_evaluation = []
            rpp_evaluation = []
            patk_prediction = []

            irp_pred_labels = mdata['irp_results']

            rpp_pred_labels = []
            rpp_true_labels = []

            teval = len(mdata['irp_results'])

            for i in range(teval):

                irp_true_labels_single = self.__gt_irp_labels[i]
                irp_pred_labels_single = irp_pred_labels[i]


                # Evaluating IRP
                irp_acc = accuracy_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_nacc = norm_acc(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_mcc = matthews_corrcoef(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))
                irp_f1 = f1_score(irp_true_labels_single.reshape(-1), irp_pred_labels_single.reshape(-1))

                irp_evaluation.append([irp_acc, irp_nacc, irp_f1, irp_mcc])
                ###

                # Evaluating PATK for round 0
                if i == 0 or i == 1:

                    pred_patk = irp_pred_labels_single.sum(axis=1).reshape(-1, 1).astype(np.int32)
                    true_patk = irp_true_labels_single.sum(axis=1).reshape(-1, 1).astype(np.int32)

                    patk_prediction.append(np.hstack([pred_patk, true_patk]))
                ###

                # Evaluating RPP
                T_vals = np.arange(1, 11, 1).reshape(1, -1)
                rpp_pred_labels_single = (irp_pred_labels_single.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)
                rpp_true_labels_single = (irp_true_labels_single.sum(axis=1).reshape(-1, 1) >= T_vals).astype(np.uint8)

                rpp_nacc = np.array([norm_acc(rpp_true_labels_single[:, x], rpp_pred_labels_single[:, x])
                                     for x in range(0, T_vals.size)])

                rpp_evaluation.append(rpp_nacc)

                ###

            #for t in irp_evaluation: print(t)
            irp_evaluation = np.vstack(irp_evaluation)
            patk_prediction = np.vstack(patk_prediction)
            rpp_evaluation = np.vstack(rpp_evaluation)

            mdata['irp_evaluation'] = np.vstack([irp_evaluation, np.mean(irp_evaluation, axis=0).reshape(1, -1)])
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

    def draw_rpp_results(self, outprefix="", outf=None):

        meas_name, ri, blim = Evaluator.measure_map["NACC"]

        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_size_inches(3, 3)

        ax.set_title("{0:s}".format(exkey_map[self.key]),
                      fontdict=dict(fontsize=14, horizontalalignment='center', style='italic'))

        line_handlers = []
        line_labels = []

        for i, mdata in enumerate(self.data):

            if mdata['params']['add_rpp']:

                mdata = self.data[i]
                vals = mdata['rpp_evaluation'][-1]
                nv = vals.shape[0]

                line, = ax.plot(np.arange(1, nv+1, 1), vals, color=mdata['params']['linecolor'],
                                linewidth=mdata['params']['linewidth'], linestyle=mdata['params']['linestyle'],
                                marker=mdata['params']['marker'], markeredgewidth=mdata['params']['markeredgewidth'],
                                markersize=mdata['params']['markersize'], fillstyle='none', alpha=0.9)

                line_handlers.append(line)
                line_labels.append(mdata['params']['label'])

        ax.set_ylim(bottom=blim, top=1.0)
        ax.set_yticks([x for x in np.arange(0.1, 1.0, 0.2)], minor=True)
        ax.set_yticks([x for x in np.arange(0.0, 1.1, 0.2)])
        ax.set_yticklabels(["{0:0.1f}".format(x) for x in np.arange(0.0, 1.1, 0.2)], fontdict=dict(fontsize=12))
        ax.set_ylabel("nACC", fontdict=dict(fontsize=12))

        ax.set_xlim(left=1.0, right=10.0)
        ax.set_xticks([x for x in np.arange(1, 11, 1)])
        ax.set_xticklabels(["{0:d}".format(x) for x in np.arange(1, 11, 1)], fontdict=dict(fontsize=12))
        ax.set_xlabel("T value", fontdict=dict(fontsize=12))


        # Line Legend
        plt.legend(line_handlers, line_labels, loc='upper center', bbox_to_anchor=[0.5, -0.3],
                                        fancybox=True, shadow=True, ncol=2)

        ax.grid(True, which='both')
        #plt.tight_layout()


        # Saving
        safe_create_dir(self.respath)

        if not outf:
            if outprefix == "":
                 outprefix = self.evalname + ".{0:s}.".format(self.key)

            outfilename = "{0:s}/{2:s}_rpp_{3:s}.pdf".format(self.respath, self.evalname, outprefix, measure)

            plt.savefig(outfilename)

        else:
            outf.savefig()

        plt.close()


    def draw_irp_results(self, outprefix="", measure='MCC', outf=None):

        meas_name, ri, blim = Evaluator.measure_map[measure]

        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_size_inches(4, 5)

        fig.set_constrained_layout_pads(w_pad=0.6)

        if measure != 'MCC':
            ax.set_ylim(bottom=blim, top=1.0)
            ax.set_yticks([float(x) / 10 for x in range(0, 11, 1)])
            fw = 0.0010    # space betweem limiting border and axis
            sw = 0.0005  # space between bars
            bw = 0.0025   # bar width
        else:
            ax.set_ylim(bottom=blim, top=1.0)
            ax.set_yticks([float(x) / 10 for x in range(-10, 11, 2)])
            fw = 0.0010    # space betweem limiting border and axis
            sw = 0.0005  # space between bars
            bw = 0.0025   # bar width

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_ylabel(measure)
        #ax.grid(True, axis='y')
        ax.set_title("Relevance Prediction\n{0:s}\n{1:s}".format(meas_name, self.key),
                      fontdict={'fontsize': 14, 'horizontalalignment': 'center'})


        # Enqueues the index of line plots to plot after plotting the bars
        line_queue = []

        rect_handlers = []  # Bar legend
        rect_labels = []
        line_handlers = []  # Line legend
        line_labels = []

        x = fw
        nbars = 0

        for i, mdata in enumerate(self.data):

            val = mdata['irp_evaluation'][-1, ri]

            if mdata['params']['plot_type'] == 'bar':

                rect, = ax.bar(x, val, bw, 0, align='edge', label=mdata['params']['label'],
                               color=mdata['params']['color'])

                rect_handlers.append(rect)
                rect_labels.append(mdata['params']['label'])

                posx = rect.get_x()
                posy = rect.get_y()
                hgt = rect.get_height()
                wdt = rect.get_width()

                ax.text(posx + wdt/2, posy + hgt + 0.01, "{0:0.3f}".format(val), fontsize=9, bbox={'alpha': 0.0},
                         horizontalalignment='center')

                x += bw + sw
                nbars += 1

            elif mdata['params']['plot_type'] == 'line':
                line_queue.append(i)

        max_x = fw + nbars*bw + (nbars-1)*sw + fw
        ax.set_xlim(0.0, max_x)

        # Lines are plotted after the xlimit is defined, which can only happen when all
        # bars are drawn
        for i in line_queue:

            mdata = self.data[i]
            val = mdata['irp_evaluation'][-1, ri]

            line, = ax.plot([0.0, max_x], [val-0.01, val-0.01], color=mdata['params']['color'],
                            linewidth=mdata['params']['linewidth'], linestyle=mdata['params']['linestyle'])
            line_handlers.append(line)
            line_labels.append(mdata['params']['label'])

            ax.text(max_x, val, "{0:0.3f}".format(val), fontsize=8, color=mdata['params']['color'],
                    horizontalalignment='left',  verticalalignment='center')

        # Line Legend
        plt.gca().add_artist(plt.legend(line_handlers, line_labels, loc='upper center', bbox_to_anchor=[0.5, -0.3],
                                        fancybox=True, shadow=True, ncol=1))

        # Rect Legend
        ax.legend(rect_handlers, rect_labels, loc='upper center', bbox_to_anchor=[0.5, -0.05], fancybox=True,
                  shadow=True, ncol=self.__draw_params.get('legend_columns', 1))


        # Saving
        safe_create_dir(self.respath)

        if not outf:
            if outprefix == "":
                 outprefix = self.evalname + ".{0:s}.".format(self.key)

            outfilename = "{0:s}/{2:s}_irp_{3:s}.pdf".format(self.respath, self.evalname, outprefix, measure)

            plt.savefig(outfilename)

        else:
            outf.savefig()

        plt.close()

"""
    def draw_patk_correlation(self, outprefix="", outf=None):

        np.set_printoptions(precision=2, threshold=10000, linewidth=200)
        cidx = []
        for i, mdata in enumerate(self.data):
            if mdata['params']['add_correlation']:
                cidx.append(i)

        fig, axes = plt.subplots(nrows=len(cidx))
        #fig.set_size_inches(2.9, 4.0)

        fig.suptitle("{1:s}".format(self.__k, key_map[self.key]),
                     fontdict=dict(fontsize=12, horizontalalignment='center', style='italic'))

        class_labels = ["{0:0.1f}".format(v) for v in np.arange(0.0, 1.1, 0.1)]

        for i, c in enumerate(cidx):

            ax = axes[i]
            mdata = self.data[c]

            cnf_matrix = confusion_matrix(mdata['patk_prediction'][:, 1],
                                          mdata['patk_prediction'][:, 0],
                                          labels=[x for x in range(0, 11)])

            cnf_matrix = np.nan_to_num(cnf_matrix.astype(np.float32)/cnf_matrix.sum(axis=1)[:, np.newaxis], copy=False)

            heatmap(cnf_matrix, class_labels, class_labels, ax=ax,
                    label="p@{0:d}".format(self.__k), title=mdata['params']['label'], cmap="Purples")

        plt.tight_layout()
        plt.subplots_adjust(top=0.89)


        # Saving
        safe_create_dir(self.respath)

        if not outf:
            if outprefix == "":
                 outprefix = self.evalname + ".{0:s}.".format(self.key)

            outfilename = "{0:s}/{1:s}_pat{2:d}_correlation.pdf".format(self.respath, outprefix, self.__k)

            plt.savefig(outfilename)

        else:
            outf.savefig()

        plt.close()
"""


"""
    def draw_patk_correlation(self, outprefix="", outf=None):

        cidx = []
        for i, mdata in enumerate(self.data):
            if mdata['params']['add_correlation']:
                cidx.append(i)

        fig, axes = plt.subplots(nrows=len(cidx), constrained_layout=True)
        fig.set_size_inches(2.9, 4.0)

        plt.suptitle("P@{0:d} Correlation\n{1:s}".format(self.__k, self.key),
                     fontdict={'fontsize': 14, 'horizontalalignment': 'center'})

        class_names = ["{0:0.1f}".format(v) for v in np.arange(0.0, 1.1, 0.2)]

        for i, c in enumerate(cidx):

            ax = axes[i]
            mdata = self.data[c]

            #pdb.set_trace()

            ax.set_xlabel("Predicted p@{0:d}".format(self.__k))
            ax.set_xticklabels(["{0:0.1f}".format(v) for v in np.arange(0.0, 1.1, 0.2)], fontsize=8)
            ax.set_xlim(-0.1, 1.1)

            ax.set_ylabel("True p@{0:d}".format(self.__k))
            ax.set_yticklabels(["{0:0.1f}".format(v) for v in np.arange(0.0, 1.1, 0.2)], fontsize=8)
            ax.set_ylim(-0.1, 1.1)

            s1 = ax.scatter(mdata['patk_prediction'][0][:-1, 0], mdata['patk_prediction'][0][:-1, 1],
                      s=mdata['params']['markersize'],
                      c='blue',
                      marker=mdata['params']['marker'],
                      linewidths=mdata['params']['markeredgewidth'],
                      alpha=0.05)

            s2 = ax.scatter(mdata['patk_prediction'][1][:-1, 0], mdata['patk_prediction'][1][:-1, 1],
                      s=mdata['params']['markersize'],
                      c='blue',
                      marker=mdata['params']['marker'],
                      linewidths=mdata['params']['markeredgewidth'],
                      alpha=0.05)

            ax.legend([s1, s2], ["{0:s} - Fold 0".format(mdata['params']['label']),
                       "{0:s} - Fold 1".format(mdata['params']['label'])],
                      loc='upper center', bbox_to_anchor=[0.5, -0.2], fancybox=True,
                      shadow=True, ncol=1)


        # Saving
        safe_create_dir(self.respath)

        if not outf:
            if outprefix == "":
                 outprefix = self.evalname + ".{0:s}.".format(self.key)

            outfilename = "{0:s}/{1:s}_pat{2:d}_correlation.pdf".format(self.respath, outprefix, self.__k)

            plt.savefig(outfilename)

        else:
            outf.savefig()

        plt.close()
"""



