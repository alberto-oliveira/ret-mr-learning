#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import shutil
import glob

sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source")
from common.cfgloader import *

def get_outfile_name(rkdir):

    if rkdir[-1] == "/":  aux = rkdir[:-1]
    else:  aux = rkdir

    parts = aux.rsplit("/", 2)

    outname = parts[1] + "_" + parts[2] + ".txt"

    return aux + "/" + outname

pathcfg = cfgloader('/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source/path.cfg')

for key in pathcfg['rank']:
    if key not in pathcfg['DEFAULT']:

        rkdir = pathcfg['rank'][key]
        lsdir = os.listdir(rkdir)

        print(". {0:s}".format(rkdir))
        outfname = get_outfile_name(rkdir)

        with open(outfname, 'w') as of:

            for m in lsdir:

                fulldir = rkdir + m + "/"
                if os.path.isdir(fulldir):
                    print("    |_ {0:s}".format(m))

                    fnum = int(m.rsplit("_", 1)[-1])

                    rkfiles = os.listdir(fulldir)
                    rkfiles.sort()

                    #print(outfname)

                    for rkfn in rkfiles:

                        line = "{0:03d} {1:s}\n".format(fnum, rkfn)
                        of.write(line)
                        #print("          ->", line)