#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt



def draw_irp_results(resfile, outfile, measure, dbparams):

    meas_name, ri, blim = measure_map[measure]

    # +1 to column because column 0 are labels
    data = np.loadtxt(resfile, delimiter=',', skiprows=1, usecols=ri+1)

    M = data.shape[0]

    fw = 0.1
    sw = 0.025
    bw = (1 + sw - M * sw - fw / 2) / (M - 0.5)

    if measure == 'ACC':
        resi = 0
    elif measure == 'F1':
        resi = 1
    elif measure == 'MCC':
        resi = 2

    rects = dict()

    for i, mdata in enumerate(self.data):
        x = i * (bw + sw) + fw / 2

        val = mdata['irp_evaluation'][-1, resi]

        print(mdata['params']['color'])
        # try:
        rect = plt.bar(x, val, label=mdata['params']['label'])
        # except:
        # return
        rects[mdata['name']] = rect

        # plt.autoscale(False)

        posx = rect[0].get_x()
        posy = rect[0].get_y()
        hgt = rect[0].get_height()

        plt.text(posx, posy + hgt + 0.01, "{0:0.2f}".format(val), fontsize=12, bbox={'alpha': 0.0})

    if measure != 'MCC':
        if dbcfg:
            p10 = dbcfg[self.key].getfloat('p10')
            plt.plot([0.0, 4.0], [p10, p10], 'r--', linewidth=2)
        plt.ylim(bottom=0.0, top=1.0)
        leg_ypos = 0.08
        plt.yticks([float(x) / 10 for x in range(0, 11, 1)])
    else:
        plt.ylim(bottom=-1.0, top=1.0)
        leg_ypos = 0.14
        plt.yticks([float(x) / 10 for x in range(-10, 11, 2)])

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    plt.ylabel(measure)
    plt.grid(True, axis='y')
    plt.title("Individual Rank Prediction -- {0:s} -- {1:s}".format(measure, self.key),
              fontdict={'fontsize': 18})

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, leg_ypos), bbox_transform=plt.gcf().transFigure,
               fancybox=True, shadow=True, ncol=nmet)

    # plt.tight_layout()

    safe_create_dir(self.respath)

    if outprefix == "":
        outprefix = self.evalname + ".{0:s}.".format(self.key)

    outfilename = "{0:s}/{2:s}_irp_{3:s}.pdf".format(self.respath, self.evalname, outprefix, measure)

    print(outfilename)
    plt.savefig(outfilename)
