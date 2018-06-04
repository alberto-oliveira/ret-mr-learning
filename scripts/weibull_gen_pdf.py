#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import weibull_min as wbl

kvar = [0.5, 1.0, 1.5, 5.0, 15.0]
lvar = [1.0, 1.25, 1.5, 1.75, 2.0]
pvar = [0.0, 0.5, 1.0, 1.5]

cmap = plt.get_cmap('Set1')

outpdf = PdfPages('weibull_example_plots.pdf')

x = np.arange(0.0, 10.0, 0.01)

lbda = 1.0
for i in range(5):
    k = kvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.2}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Scale = 1.0)')
outpdf.savefig()
plt.close()

k = 0.5
for i in range(5):
    lbda = lvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.3}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 0.5)')
outpdf.savefig()
plt.close()

k = 1.0
for i in range(5):
    lbda = lvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.3}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 1.0)')
outpdf.savefig()
plt.close()

k = 1.5
for i in range(5):
    lbda = lvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.3}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 1.5)')
outpdf.savefig()
plt.close()

k = 5.0
for i in range(5):
    lbda = lvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.3}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 5.0)')
outpdf.savefig()
plt.close()

k = 15.0
for i in range(5):
    lbda = lvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda)
    plt.plot(x, w, color=c, label='lambda = {0:0.3}, k = {1:0.1f}'.format(lbda, k), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 15.0)')
outpdf.savefig()
plt.close()

k = 5.0
lbda = 1
for i in range(4):
    p = pvar[i]
    c = cmap(i)
    w = wbl.pdf(x, k, scale=lbda, loc=p)
    plt.plot(x, w, color=c, label='Location = {0:0.2f}'.format(p), linewidth=1)

plt.xlim(0.0, 3.5)
plt.ylim(0.0, 2.5)
plt.legend()
plt.suptitle('Weibull PDF (Fixed Shape = 5.0 and Scale = 1.0)')
outpdf.savefig()
plt.close()

outpdf.close()