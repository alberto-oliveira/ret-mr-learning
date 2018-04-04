#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import shutil
import glob

sys.path.append("/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source")

from common.cfgloader import *

pathcfg = cfgloader('/home/alberto/SpotME/projects/performance-prediction/sources/ret-mr-learning/source/path.cfg')

for key in ['oxford_desc1', 'oxford100k_desc1', 'unicamp_desc1', 'unicamp100k_desc1']:
    if key not in pathcfg['DEFAULT']:

        rkdir = pathcfg['rank'][key]
        lsdir = os.listdir(rkdir)

        print(". {0:s}".format(rkdir))

        for m in lsdir:

            fulldir = rkdir + m + "/"
            if os.path.isdir(fulldir):
                print("    |_ {0:s}".format(m))

                rkfiles = glob.glob(fulldir + "*.rk")

                for rkf in rkfiles:

                    #parts = (os.path.basename(rkf)).split("_", 1)
                    #newname = parts[1]

                    print("      |_ {0:s}".format(os.path.basename(rkf)))
                    #shutil.move(rkf, fulldir + newname)

        print("---")

