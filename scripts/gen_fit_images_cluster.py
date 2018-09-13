#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
sys.path.append("../source")
import argparse
import glob
from collections import OrderedDict

import ipdb as pdb

import numpy as np
from jenkspy import jenks_breaks
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AffinityPropagation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matlab.engine
matlab_engine = matlab.engine.start_matlab()

from rankutils.cfgloader import *
from rankutils.utilities import safe_create_dir, completedir, getbasename
from rankutils.rIO import read_rank
from clustering import head_tail_clustering, diff_break_clustering

col_map = dict(oxford100k='votes',
               unicamp100k='votes',
               vggfaces='dists',
               places365='dists',
               imagenet='dists')

min_tail_size = 5

def get_tail(fulltail, f, z):

    tail_sz = fulltail.shape[0]  # Full tail size

    if tail_sz < min_tail_size:
        return []

    sidx = int(np.floor(f * tail_sz))  # starting index of the distribution
    tlen = int(np.floor(z * (fulltail[sidx:].shape[0])))  # tail distribution length

    eidx = sidx + tlen  # ending index of the distribution

    # While the tail distribution does not hit minimum size, increase its size by one on each end that can be
    # increased. If no minimum tail size is hit, returns t = -1 as a dummy prediction that no top score is
    # relevant.
    while tlen < min_tail_size and (sidx > 0 or eidx < tail_sz):
        if sidx - 1 >= 0:
            sidx -= 1
            tlen += 1
        if eidx + 1 <= tail_sz:
            eidx += 1
            tlen += 1

    if tlen < 5:
        return []

    tail_dist = fulltail[sidx:eidx]

    tail_dist[tail_dist == 0] += 0.00001

    return tail_dist, sidx, eidx

def fit_distribution(data, dist):

    data_m = matlab.double(data.reshape(-1).tolist())

    if dist == 'W':
        estpar, _ = matlab_engine.wblfit(data_m, nargout=2)
        scale = estpar[0][0]
        shape = estpar[0][1]
        location = 0
    elif dist == 'W_f':
        #estpar, _ = matlab_engine.wblfit(data_m, nargout=2)
        aux = matlab.double(np.arange(1, data.size+1).reshape(-1).tolist())
        estpar, _ = matlab_engine.wblfit(aux, [], [], data_m, nargout=2)
        scale = estpar[0][0]
        shape = estpar[0][1]
        location = 0
    else:
        estpar, _ = matlab_engine.gevfit(data_m, nargout=2)
        shape = estpar[0][0]
        scale = estpar[0][1]
        location = estpar[0][2]

    return scale, shape, location


def gen_pdf(shape, scale, location, dist, num):

    x = matlab.double(np.arange(0, num, 0.01).reshape(-1).tolist())

    if dist == 'W':
        y = matlab_engine.wblpdf(x, scale, shape)
        T = matlab_engine.wblinv(0.99999999, scale, shape)
    else:
        y = matlab_engine.gevpdf(x, shape, scale, location)
        T = matlab_engine.gevinv(0.99999999, shape, scale, location)

    y = np.array(y, dtype=np.float64).reshape(-1)
    x = np.array(x, dtype=np.float64).reshape(-1)

    return x, y, T


def get_rk_parts(unique_rk, p, sidx, eidx):

    s = p + sidx

    e = s + (eidx - sidx)

    #print("sidx: ", sidx, " | eidx: ", eidx)
    utopk = unique_rk[:p]
    #print("utopk (0:{0:d}): ".format(p), utopk)

    amb1 = unique_rk[p:s]
    #print("amb1 ({0:d}:{1:d}): ".format(p, s), amb1)

    dist = unique_rk[s:e]
    #print("dist ({0:d}:{1:d}): ".format(s, e), dist)

    amb2 = unique_rk[e:]
    #print("amb2 ({0:d}:): ".format(e), amb2)

    return utopk, amb1, dist, amb2, s



def plot_dist(x, y, l, distparts, title, outpdf):

    bar_colors = [(0.5, 0.5, 0.5),
                  (1.0, 0.5, 0.5),
                  (0.75, 0, 0.75),
                  (0.5, 0.5, 1.0)]

    bar_labels = ["topk", "amb. 1", "tail", "amb. 2"]

    rect_handles = []
    line_handles = []

    rect_labels = []

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 5)

    ax.set_title("{0:s}".format(title),
                 fontdict=dict(fontsize=14, horizontalalignment='center', style='italic'))

    line, = ax.plot(x[0], y[0], color='blue', linewidth=2, linestyle='-', label=l[0])
    line_handles.append(line)

    line, = ax.plot(x[1], y[1], color='red', linewidth=2, linestyle='-', label=l[1])
    line_handles.append(line)

    #line, = ax.plot(x[2], y[2], color=(0, 0.75, 0, 1.0), linewidth=2, linestyle='-', label=l[2])
    #line_handles.append(line)

    s = 1
    for x in range(4):
        part = distparts[x]
        clr = bar_colors[x]
        lbl = bar_labels[x]

        e = part.size + s

        #print("s:{0:d}, e:{1:d}\n".format(s, e))
        if s < e:
            idx = np.arange(s, e)
            #print(idx)
            rect = ax.bar(idx, part, 1.0, 0.0, align='edge', color=clr)
            rect_handles.append(rect[0])
            rect_labels.append(lbl)
        s = e

    plt.gca().add_artist(plt.legend(rect_handles, rect_labels, loc='upper left', bbox_to_anchor=[1.0, 0.4],
                                    fancybox=True, shadow=True, ncol=1, fontsize=10))

    plt.legend(line_handles, l, loc='upper left', bbox_to_anchor=[1.0, 1.0], fancybox=True, shadow=True, ncol=1,
               fontsize=8)

    ma = distparts[0][1]*1.5
    ax.set_ylim(bottom=0.05, top=0.2)
    ax.set_ylabel("P", fontdict=dict(fontsize=12))

    mi = np.min(np.array(x).reshape(-1))
    ma = np.max(np.array(x).reshape(-1))

    ax.set_xlim(left=0, right=s+5)
    ax.set_xlabel("x", fontdict=dict(fontsize=12))

    ax.grid(True, which='both')
    #plt.tight_layout()

    outpdf.savefig()

    #plt.show()
    plt.close()


def plot_dist2(x, y, l, distparts, title, outpdf):

    bar_colors = [(0.5, 0.5, 0.5),
                  (1.0, 0.5, 0.5),
                  (0.75, 0, 0.75),
                  (0.5, 0.5, 1.0)]
    bar_labels = ["topk", "amb. 1", "tail", "amb. 2"]

    rect_handles = []
    rect_labels = []

    fig, [ax0, ax1] = plt.subplots(2)
    fig.set_size_inches(14, 5)
    fig.suptitle(title)

    # Axis 0
    s = 0
    for k in range(4):
        part = distparts[k]
        clr = bar_colors[k]
        lbl = bar_labels[k]

        e = part.size + s

        #print("s:{0:d}, e:{1:d}\n".format(s, e))
        if s < e:
            idx = np.arange(s, e)
            #print(idx)
            rect = ax0.bar(idx, part, 1.0, 0.0, align='edge', color=clr)
            rect_handles.append(rect[0])
            rect_labels.append(lbl)
        s = e

    aux = np.sort(np.hstack(distparts).reshape(-1))
    ax0.set_ylim(bottom=aux[0]*0.999, top=aux[-2]*1.01)
    ax0.set_ylabel("P", fontdict=dict(fontsize=8))

    ax0.set_xlim(left=0, right=e)
    ax0.set_xlabel("x", fontdict=dict(fontsize=8))

    plt.legend(rect_handles, rect_labels, loc='upper left', bbox_to_anchor=[1.0, 1.0],
                                    fancybox=True, shadow=True, ncol=1, fontsize=6)

    ####

    # Axis 1
    lg_handles = []
    lg_labels = l + ['Fitted Tail']

    tail = distparts[2]
    ts = tail.size

    line, = ax1.plot(x[0], y[0], color='blue', linewidth=2, linestyle='-', label=l[0])
    lg_handles.append(line)

    line, = ax1.plot(x[1], y[1], color='red', linewidth=2, linestyle='-', label=l[1])
    lg_handles.append(line)

    rect = ax1.bar(np.arange(ts), tail, 1.0, 0.0, align='edge', color=bar_colors[2], label='Fitted tail')
    lg_handles.append(rect)

    ax1.set_ylim(bottom=tail[-1]*0.999, top=tail[0]*1.01)
    ax1.set_ylabel("P", fontdict=dict(fontsize=8))

    ax1.set_xlim(left=0, right=ts)
    ax1.set_xlabel("x", fontdict=dict(fontsize=8))

    plt.gca().add_artist(plt.legend(lg_handles, lg_labels,
               loc='upper left', bbox_to_anchor=[1.0, 1.0], fancybox=True, shadow=True, ncol=1, fontsize=6))

    ####


    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    outpdf.savefig()

    #plt.show()
    plt.close()


def plot_diff(y, title, outpdf):

    x = np.arange(0, y.size)

    mi = np.min(y)
    ma = np.max(y)
    M = np.mean(y)
    std = np.std(y)

    fig, axes = plt.subplots(3)
    fig.set_size_inches(10, 7)
    fig.suptitle(title)

    axes[0].set_title("Diffs.", fontdict=dict(fontsize=12, horizontalalignment='center', style='italic'))
    axes[1].set_title("Sorted Diffs.", fontdict=dict(fontsize=12, horizontalalignment='center', style='italic'))
    axes[2].set_title("Above mean.", fontdict=dict(fontsize=12, horizontalalignment='center', style='italic'))

    lbl = "Ranked diffs\nmin = {0:0.2f}\nmax = {1:0.2f}\nmean = {2:0.2f}\nstd = {3:0.3f}".format(mi, ma, M, std)

    #ax.plot(x, y, color="red", linewidth=0.1, linestyle='-', label=lbl)
    axes[0].bar(x, y+0.1, 1.0, -0.1, align='edge', color="red", label=lbl)
    axes[0].legend(loc='upper right', bbox_to_anchor=[1.0, 1.0], fancybox=True, shadow=True, ncol=1, fontsize=10)

    axes[1].bar(x, np.sort(y)[::-1]+0.1, 1.0, -0.1, align='edge', color="blue")

    aux = y <= M
    ymean = np.array(y)
    ymean[aux] = 0.0

    axes[2].bar(x, ymean+0.1, 1.0, -0.1, align='edge', color='purple')


    for ax in axes:
        ax.set_ylim(bottom=-0.1, top=ma)
        ax.set_ylabel("Difference", fontdict=dict(fontsize=8))

        ax.set_xlim(left=0, right=x[-1] + 1)
        ax.set_xlabel("Rank position", fontdict=dict(fontsize=8))

    plt.tight_layout()

    fig.subplots_adjust(top=0.85)
    outpdf.savefig()
    plt.close()

def plot_clusters(clusterings, title, outpdf):

    cmap = plt.get_cmap('winter')

    nc = len(clusterings.keys())
    keys = list(clusterings.keys())

    fig, axes = plt.subplots(nc, 1)
    fig.set_size_inches(5, 15)
    fig.suptitle(title)

    cval = 0.0
    cinc = 1.0/(nc-1)

    for ax, key in zip(axes, keys):

        ax.set_title("{0:s}".format(key), fontdict=dict(fontsize=12, horizontalalignment='center', style='italic'))

        y = clusterings[key]
        if y.size == 0:
            y = np.array([1])
        x = np.arange(0, y.size)

        lbl = "# clusters = {0:d}".format(y.size)

        rect = ax.bar(x, y, 1, 0.0, align='edge', color=cmap(cval), label=lbl)

        cval += cinc

        ax.legend([rect], [lbl], loc='upper right', bbox_to_anchor=[1.0, 1.0], fancybox=True, shadow=True, ncol=1,
                  fontsize=10)

        ax.set_ylim(bottom=np.min(y)*0.999, top=np.max(y))
        ax.set_ylabel("Score", fontdict=dict(fontsize=8))

        ax.set_xlim(left=0, right=x[-1])
        ax.set_xlabel("Rank position", fontdict=dict(fontsize=8))

    plt.tight_layout()

    fig.subplots_adjust(top=0.92)
    outpdf.savefig()
    plt.close()


def plot_histogram(rk, wpar, title, outpdf):

    cml = ['inferno', 'viridis', 'plasma', 'magma']
    btl = ['doane', 'fd', 20, 50]
    ll = ['Doane', 'FD', '20 bins', '50 bins']

    fig, axes = plt.subplots(4, 1)
    fig.set_size_inches(6, 10)
    fig.suptitle(title)

    for i, ax in enumerate(axes):
        cmap = plt.get_cmap(cml[i])

        lbl = ll[i]
        handles = []
        labels = [lbl]

        hist, edges = np.histogram(rk, bins=btl[i], density=True)
        hist = hist[::-1]
        edges = edges[::-1]

        print("Hist ", btl[i], " -> size: {0:d}".format(hist.size))
        print("     -> max: {0:f}".format(float(np.max(hist))))
        print("     -> min: {0:f}".format(float(np.min(hist))))
        print("vals: ", hist)
        print("edges: ", edges)

        if lbl == 'Doane' or lbl == 'FD':
            lbl += " ({0:d} bins)".format(hist.size)

        clist = [cmap(w) for w in (hist - np.min(hist))/(np.max(hist) - np.min(hist))]
        x = np.arange(0, hist.size+1)

        rect = ax.bar(x[:-1], hist, 1, 0.0, align='edge', color=clist)
        handles.append(rect)

        yw = []

        if i == 0:
            xw, yw, _ = gen_pdf(shape=wpar[0], scale=wpar[1], location=wpar[2], dist='W', num=hist.size+1)
            line, = ax.plot(xw, yw, linewidth=1, color='red')
            handles.append(line)
            labels.append("Shape = {0:0.1f}, Scale = {1:0.1f}, Loc = {2:0.1f}".format(wpar[0], wpar[1], wpar[2]))

        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=[1.0, 1.0], fancybox=True, shadow=True, ncol=1,
                  fontsize=10)

        ax.set_ylim(bottom=0.0, top=np.max(hist)*1.05)
            #ax.set_ylim(bottom=0.0, top=np.max(yw)*1.05)
        ax.set_ylabel("Frequency", fontdict=dict(fontsize=8))

        ax.set_xlim(0, np.max(x)+1)
        #ax.set_xticks(x)
        ax.set_xticklabels(["{0:0.1f}".format(e) for e in edges], fontdict=dict(fontsize=6, rotation=-45))
        ax.set_xlabel("Intervals", fontdict=dict(fontsize=8))
        print("####")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    #plt.show()
    outpdf.savefig()
    plt.close()




def generate_fits(outdir, expkey, k, f, z, l, c, num):

    mms = MinMaxScaler((0, 1))

    np.set_printoptions(threshold=100, linewidth=300)

    np.random.seed(1)

    col = col_map[expkey.split('_')[0]]

    pathcfg = cfgloader("/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/path_2.cfg")
    rkflist = glob.glob(pathcfg['rank'][expkey] + "*.rk")

    choices = np.arange(0, len(rkflist))
    np.random.shuffle(choices)

    fulloutdir = "{0:s}{1:s}_k{2:d}.f{3:0.2f}.z{4:0.2f}.l{5:d}/".format(outdir, expkey, k, f, z, l)
    safe_create_dir(fulloutdir)

    fit_outPDF = PdfPages(fulloutdir + "score_fits.pdf")
    diff_outPDF = PdfPages(fulloutdir + "rank_diffs.pdf")
    cluster_outPDF = PdfPages(fulloutdir + "clusterings.pdf")
    hist_outPDF = PdfPages(fulloutdir + "hists.pdf")

    for i in choices[:num]:

        rkfpath = rkflist[i]
        rkname = getbasename(rkfpath)
        print("{0:s}...".format(rkname), end='\n', flush=True)

        rk = read_rank(rkfpath, col)
        if col == 'dists':
            rk = np.max(rk) - rk

        if l != -1:
            rk = rk[:l]

        rk = mms.fit_transform(rk.reshape(-1, 1)).reshape(-1)

        unique_rk = np.unique(rk)
        unique_rk.sort()
        unique_rk = np.flip(unique_rk, axis=0)

        p = np.argwhere(unique_rk == rk[k-1])[0][0] + 1

        try:
            taildist, sidx, eidx = get_tail(unique_rk[p:], f, z)
        except ValueError as ve:
            print("Could not find a suitable tail")
            continue

        utopk, amb1, dist, amb2, s = get_rk_parts(unique_rk, p, sidx, eidx)

        assert (taildist.size == dist.size and taildist[0] == dist[0] and taildist[1] == dist[1]), \
               "{0:s} not equal to {1:s}".format(str(taildist), str(dist))

        nz = np.sum(taildist)

        scl_w, shp_w, loc_w = fit_distribution(taildist, 'W')
        scl_wf, shp_wf, loc_wf = fit_distribution(taildist, 'W_f')
        #scl_g, shp_g, loc_g = fit_distribution(taildist, 'GEV')


        x_w, y_w, T_w = gen_pdf(shp_w, scl_w, loc_w, 'W', utopk.size + amb1.size + dist.size + amb2.size)
        x_wf, y_wf, T_wf = gen_pdf(shp_wf, scl_wf, loc_wf, 'W', utopk.size + amb1.size + dist.size + amb2.size)
        #x_g, y_g, T_g = gen_pdf(shp_g, scl_g, loc_g, 'GEV', utopk.size + amb1.size + dist.size + amb2.size)

        label_w = "Weibull\nshape = {0:0.2f}\nscale = {1:0.2f}\nT = {2:0.5f}\n".format(shp_w, scl_w, T_w/nz)
        label_wf = "Weibull by Freq\nshape = {0:0.2f}\nscale = {1:0.2f}\nT = {2:0.5f}\n"\
                    .format(shp_wf, scl_wf, T_wf/nz)
        #label_g = "GEV\nshape = {0:0.2f}\nscale = {1:0.2f}\nloc = {2:0.2f}\nT = {3:0.2f}\n"\
        #           .format(shp_g, scl_g, loc_g, T_g)


        title = "{0:s}\n{1:s}".format(expkey, rkname)


        plot_dist2([x_w, x_wf], [y_w, y_wf], [label_w, label_wf], [utopk/nz, amb1/nz, dist/nz, amb2/nz], title,
                   fit_outPDF)

        if col == 'votesff':
            rk_diff = unique_rk[0:-1] - unique_rk[1:]
        else:
            rk_diff = rk[0:-1] - rk[1:]
        #print(rk_diff)
        #rk_diff.sort()

        plot_diff(rk_diff[10:], title, diff_outPDF)

        kmn = KMeans(n_clusters=c)
        mns = MeanShift()
        dbs = DBSCAN()
        afp = AffinityPropagation()

        clusterings = OrderedDict(no_cluster=rk, unique=unique_rk)

        kmn.fit(rk.reshape(-1, 1))
        clusterings['kmeans'] = np.sort(kmn.cluster_centers_.reshape(-1))[::-1]
        #print("kmeans: ", clusterings['kmeans'])

        clusterings['jenks'] = np.sort(jenks_breaks(rk.reshape(-1), c-1))[::-1]
        #print("jenks: ", clusterings['jenks'])

        mns.fit(rk.reshape(-1, 1))
        clusterings['mean_shift'] = np.sort(mns.cluster_centers_.reshape(-1))[::-1]
        #print("Mean Shift: ", clusterings['mean_shift'])

        clusters = diff_break_clustering(rk.reshape(-1))
        clusterings['diff_break'] = np.sort(clusters).reshape(-1)[::-1]
        #print("Diff Break: ", clusterings['diff_break'])

        #_, clusters = head_tail_clustering(rk.reshape(-1), 0.4)
        #clusterings['head_tail'] = np.sort(clusters).reshape(-1)[::-1]
        #print("Head Tail: ", clusterings['head_tail'])

        #afp.fit(rk.reshape(-1, 1))
        #clusterings['affinity_prop'] = np.sort(afp.cluster_centers_.reshape(-1))[::-1]
        #print("Affinity prop: ", clusterings['affinity_prop'])

        plot_clusters(clusterings, title, cluster_outPDF)

        plot_histogram(taildist, (shp_w, scl_w, loc_w), title, hist_outPDF)

        print("Done!\n")

    fit_outPDF.close()
    diff_outPDF.close()
    cluster_outPDF.close()
    hist_outPDF.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("expkey", help="Experiment key", type=str)
    parser.add_argument("top", help="Top positions of interest.", type=int)
    parser.add_argument("outdir", help="Output directory.", type=str)
    parser.add_argument("-f", "--fval", help="F tail cut value. Default = 0.0", type=float, default=0.0)
    parser.add_argument("-z", "--zval", help="Z tail cut value. Default = 1.0", type=float, default=1.0)
    parser.add_argument("-l", "--limit", help="Limits the number of scores to be loaded. Default is -1, all of them.",
                        type=int, default=-1)
    parser.add_argument("-c", "--clusters", help="Number of clusters for tail. Default = 100", type=int, default=100)
    parser.add_argument("-n", "--num", help="Number of plots to be generated. Random sampled from all ranks. -1 is all,"
                                            "default is -1", type=int, default=-1)
    #parser.add_argument("-d", "--dist", help="Type of distribution to be fitted. Extreme Value (EV), Generalized "
    #                                         "Extreme Value (GEV), or Weibull (W). Default = \'W\'", type=str,
    #                    default='W', choices=['GEV', 'W'])

    args = parser.parse_args()

    generate_fits(completedir(args.outdir), args.expkey, args.top, args.fval, args.zval, args.limit, args.clusters,
                  args.num)



