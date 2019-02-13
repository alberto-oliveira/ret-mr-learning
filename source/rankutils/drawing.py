#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys

import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rankutils.evaluation import measure_map

from sklearn.preprocessing import MinMaxScaler
#import ipdb as pdb


def colors_from_cmap(cmap_name, values, bounds=(0.0, 1.0)):

    lower, upper = bounds
    if lower >= upper:
        raise ValueError("minv should be less than maxv")

    if lower < 0.0:
        lower = 0.0
    if upper > 1.0:
        upper = 1.0

    if not isinstance(values, np.ndarray):
        aux = np.array([values])
    else:
        aux = values

    cmap = plt.get_cmap(cmap_name)

    # Only scales if a vector of colors is given
    if aux.size > 1:
        mms = MinMaxScaler((lower, upper))
        tvalues = mms.fit_transform(aux.reshape(-1, 1)).reshape(-1)
    else:
        tvalues = aux

    clist = [cmap(v) for v in tvalues]

    return clist


def rank_plot(rank, k, ax=None, title='', **kwargs):

    cmaptop = kwargs.get('cmaptop', 'Purples')
    cmaptail = kwargs.get('cmaptail', 'Reds')
    cnorm_bottom = kwargs.get('cmapnormbot', 0.5)
    cnorm_top = kwargs.get('cmapnormtop', 1.0)
    limit = kwargs.get('limit', rank.size)
    start_pos = kwargs.get('start', 1)
    bw = kwargs.get('barwidth', 1.0)

    if not ax:
        ax = plt.gca()

    ax.set_title("{0:s}".format(title))

    top = rank[0:k]
    tail = rank[k:limit]

    x_top = np.arange(start_pos, k+start_pos, 1, dtype=np.int32)
    x_tail = np.arange(k+start_pos, limit+start_pos, 1, dtype=np.int32)
    coloridx = np.arange(0, limit, 1, dtype=np.float64)

    if top.size > 0:
        clist_top = colors_from_cmap(cmaptop, coloridx, cnorm_bottom, cnorm_top)[0:k]
        ax.bar(x_top, top, bw, 0.0, align='edge', color=clist_top, label='Top {0:d}'.format(k))

    clist_tail = colors_from_cmap(cmaptail, coloridx, cnorm_bottom, cnorm_top)[k:]
    ax.bar(x_tail, tail, bw, 0.0, align='edge', color=clist_tail, label='Tail')

    ax.set_xlim(left=start_pos, right=x_tail[-1] + 1)
    #ax.set_ylim(bottom=0, top=np.max(rank))

    ax.set_xlabel('Rank Position')
    ax.set_ylabel('Score')

    return


def irp_results_barv_draw(data, meas_key, method_key='set', ax=None, textfmt="{0:0.3f}"):
    measure, idx, blim = measure_map[meas_key]

    if not ax:
        ax = plt.gca()

    if method_key == 'set':
        mk = 'irp_evaluation'
    elif method_key == 'sample':
        mk = 'irp_evaluation_sample'
    else:
        raise ValueError("Invalid method key <{key:s}>. Choices are (\'set\', \'sample\')".format(method_key))

    # Enqueues the index of line plots to plot after plotting the bars
    line_queue = []

    xpos = 1

    handles = []
    labels = []

    for i, mdata in enumerate(data):

        val = mdata[mk][-1, idx]

        if mdata['params']['plot_type'] == 'bar':

            # print(mdata['name'], '->', mdata['drawargs']['color'])
            rect, = ax.bar(xpos, val, 0.9, 0, align='center', label=mdata['params']['label'] + "{0:1s}".format(''),
                           color=mdata['drawargs']['color'], alpha=1.0)
            handles.append(rect)
            labels.append(mdata['params']['label'] + "{0:1s}".format(''))

            posx = rect.get_x()
            posy = rect.get_y()
            hgt = rect.get_height()
            wdt = rect.get_width()

            ax.text(posx + wdt / 2, posy + hgt - 0.05, textfmt.format(val), fontsize=15, bbox={'alpha': 0.0},
                    rotation=270, color='white', fontweight='bold', horizontalalignment='right')

            xpos += 1

        elif mdata['params']['plot_type'] == 'line':
            line_queue.append(i)

    rbound = xpos
    ax.set_xlim(left=0.0, right=rbound)

    # Lines are plotted after the xlimit is defined, which can only happen when all
    # bars are drawn
    for i in line_queue:
        mdata = data[i]
        val = mdata[mk][-1, idx]

        line, = ax.plot([0.0, rbound], [val, val], **mdata['drawargs'])
        handles.append(line)
        labels.append(mdata['params']['label'] + "{0:15s}".format(''))

        ax.text(0.02, val, "{0:0.3f}".format(val), fontsize=12, color=mdata['drawargs']['color'],
                horizontalalignment='left', verticalalignment='bottom')

    ax.set_ylim(bottom=blim, top=1.0)
    ax.set_ylabel(meas_key)

    ax.set_xlabel('Method')
    ax.set_xticks([])

    # Rect Legend
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=[1.0, 0], fancybox=True, shadow=True)

    return


def irp_results_barh_draw(data, meas_key, method_key='set', ax=None, textfmt="{0:0.3f}", xlabel=None, ylabel=None):
    measure, idx, blim = measure_map[meas_key]

    #gettc = lambda v: 'black' if np.linalg.norm(v[0:3]) >= 0.8 else 'white'

    if not ax:
        ax = plt.gca()

    if method_key == 'set':
        mk = 'irp_evaluation'
    elif method_key == 'sample':
        mk = 'irp_evaluation_sample'
    else:
        raise ValueError("Invalid method key <{key:s}>. Choices are (\'set\', \'sample\')".format(method_key))

    # Enqueues the index of line plots to plot after plotting the bars
    line_queue = []

    xpos = 1

    handles = []
    labels = []

    for i, mdata in enumerate(data):

        val = mdata[mk][-1, idx]

        if mdata['params']['plot_type'] == 'bar':

            # print(mdata['name'], '->', mdata['drawargs']['color'])
            rect, = ax.barh(xpos, val, 0.9, 0, align='center', label=mdata['params']['label'] + "{0:1s}".format(''),
                            color=mdata['drawargs']['color'], alpha=0.4)
            handles.append(rect)
            labels.append(mdata['params']['label'] + "{0:1s}".format(''))

            posx = rect.get_x()
            posy = rect.get_y()
            hgt = rect.get_height()
            wdt = rect.get_width()

            ax.text(posx, posy + hgt / 2, textfmt.format(val), fontsize=12, bbox={'alpha': 0.0},
                    color='black', fontweight='bold', horizontalalignment='left',
                    verticalalignment='center')

            xpos += 1

        elif mdata['params']['plot_type'] == 'line':
            line_queue.append(i)

    rbound = xpos
    ax.set_ylim(bottom=0.0, top=rbound)

    # Lines are plotted after the xlimit is defined, which can only happen when all
    # bars are drawn
    for i in line_queue:
        mdata = data[i]
        val = mdata[mk][-1, idx]

        line, = ax.plot([0.0, rbound], [val, val], **mdata['drawargs'])
        handles.append(line)
        labels.append(mdata['params']['label'] + "{0:15s}".format(''))

        ax.text(0.02, val, "{0:0.3f}".format(val), fontsize=12, color=mdata['drawargs']['color'],
                horizontalalignment='left', verticalalignment='top')

    ax.plot([0.0, 0.0], [0.5, len(labels) + 0.5], linewidth=2, color='black')

    ax.set_xlim(left=blim, right=1.0)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0.5, len(labels) + 0.5)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    return


def irp_results_pos_draw(data, k, ax=None, measure='NACC', xlabel=None, ylabel=None):
    if not ax:
        ax = plt.gca()

    if measure == 'NACC':
        mkey = 'pos_evaluation_nacc'
    elif measure == 'F-Score':
        mkey = 'pos_evaluation_f1'
    else:
        raise ValueError('Invalid measure keyword -- options are <NACC> and <F-Score>')


    handles = []
    labels = []

    for i, mdata in enumerate(data):
        y = mdata[mkey][-1]
        x = np.arange(1, k + 1).astype(np.int32)

        line, = ax.plot(x, y, fillstyle='full', **mdata['drawargs'])

        handles.append(line)
        labels.append(mdata['params']['label'])

    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_yticks([y for y in np.arange(0.0, 1.1, 0.2)])
    ax.set_yticklabels(["{0:0.1f}".format(y) for y in np.arange(0.0, 1.1, 0.2)], fontdict=dict(fontsize=12))
    if ylabel:
        ax.set_ylabel(ylabel, fontdict=dict(fontsize=12))

    ax.set_xlim(left=1.0, right=k)
    ax.set_xticks([x for x in np.arange(1, k + 1, 1)])
    ax.set_xticklabels(["{0:d}".format(x) for x in np.arange(1, k + 1, 1)], fontdict=dict(fontsize=8))
    if xlabel:
        ax.set_xlabel(xlabel, fontdict=dict(fontsize=12))

    # Line Legend
    fig = plt.gcf()

    ax.grid(True, which='both')

    return handles, labels


def rpp_results_draw(data, k, ax=None, xlabel=None, ylabel=None):
    if not ax:
        ax = plt.gca()

    mkey = 'rpp_evaluation'

    handles = []
    labels = []

    for i, mdata in enumerate(data):
        y = mdata[mkey][-1]
        x = np.arange(1, k + 1).astype(np.int32)

        line, = ax.plot(x, y, fillstyle='full', **mdata['drawargs'])

        handles.append(line)
        labels.append(mdata['params']['label'])

    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_yticks([y for y in np.arange(0.0, 1.1, 0.2)])
    ax.set_yticklabels(["{0:0.1f}".format(y) for y in np.arange(0.0, 1.1, 0.2)], fontdict=dict(fontsize=12))
    if ylabel:
        ax.set_ylabel(ylabel, fontdict=dict(fontsize=12))

    ax.set_xlim(left=1.0, right=k)
    ax.set_xticks([x for x in np.arange(1, k + 1, 1)])
    ax.set_xticklabels(["{0:d}".format(x) for x in np.arange(1, k + 1, 1)], fontdict=dict(fontsize=8))
    if xlabel:
        ax.set_xlabel(xlabel, fontdict=dict(fontsize=12))

    ax.grid(True, which='both')

    return handles, labels


def rankimg(names, rel=False, ax=None, gridwidth=3.0, title="", colorkw=dict(), **kwargs):

    if not ax:
        ax = plt.gca()

    kw = dict(cmap='tab20b', cidx=[0, 1])
    kw.update(colorkw)

    k = names.size

    c = plt.get_cmap(kw['cmap'])
    c_arr = np.array([c.colors[i] for i in kw['cidx']]).reshape(2, 1, 3)

    if rel:
        imdata = np.tile(c_arr, [int(np.floor(k/2)), 3, 1])[0:k]
        col_labels = ['#', 'score', 'rel.']
    else:
        imdata = np.tile(c_arr, [int(np.floor(k/2)), 2, 1])[0:k]
        col_labels = ['#', 'scores']

    # Plot the heatmap
    im = ax.imshow(imdata, **kwargs)

    ax.set_title(title, fontdict=dict(fontsize=20), pad=40)

    # We want to show all ticks...
    ax.set_xticks(np.arange(imdata.shape[1]))
    ax.set_yticks(np.arange(imdata.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontdict=dict(fontsize=14))
    ax.set_yticklabels(names, fontdict=dict(fontsize=14))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             #rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(imdata.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(imdata.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=gridwidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def tableimg(shape, row_labels, col_labels, ax=None, gridwidth=3.0, title="", colorkw=dict(), **kwargs):

    if not ax:
        ax = plt.gca()

    kw = dict(cmap='tab20b', cidx=[0, 1])
    kw.update(colorkw)

    r, c = shape

    cmap = plt.get_cmap(kw['cmap'])
    c_arr = np.array([cmap.colors[i] for i in kw['cidx']]).reshape(2, 1, 3)

    imdata = np.tile(c_arr, [int(np.ceil(r/2)), c, 1])
    imdata = imdata[0:r]

    im = ax.imshow(imdata, **kwargs)

    ax.set_title(title, fontdict=dict(fontsize=20), pad=40)

    # We want to show all ticks...
    ax.set_xticks(np.arange(imdata.shape[1]))
    ax.set_yticks(np.arange(imdata.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontdict=dict(fontsize=14))
    ax.set_yticklabels(row_labels, fontdict=dict(fontsize=14))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             #rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(imdata.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(imdata.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=gridwidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_tableimg(tableimg, table, colfmt=[], **textkw):

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if colfmt:
        colfmt = [matplotlib.ticker.StrMethodFormatter(fmt) for fmt in colfmt]

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []

    c = len(table)
    r = table[0].size  # All columns should have consistent number of rows

    for i in range(r):

        for j in range(c):

            #print("{0:d}:{1:d} =".format(j, i), table[j][i])
            fmt = colfmt[j]

            text = tableimg.axes.text(j, i, fmt(table[j][i], None), **kw)
            texts.append(text)

    return texts


# author='matplotlib'
def heatmap(data, row_labels, col_labels, ax=None, gridwidth=3.0, cbarlabel="", title="", cbar_kw={}, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbarlabel != "":
        cbar = ax.figure.colorbar(im, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontdict=dict(fontsize=14))

    ax.set_title(title, fontdict=dict(fontsize=20), pad=20)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontdict=dict(fontsize=14))
    ax.set_yticklabels(row_labels, fontdict=dict(fontsize=14))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=gridwidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im

# author='matplotlib'
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)


    print(data.shape)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    #print("t: ", threshold)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            #print("[{0:d}, {1:d}]".format(i, j), data[i, j])
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
