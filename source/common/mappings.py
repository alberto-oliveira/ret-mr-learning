#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from common.baselines import *

# Maps available descriptor numbers for each dataset
descriptor_map = OrderedDict(oxford=[1],
                             oxford100k=[1],
                             unicamp=[1],
                             unicamp100k=[1],
                             vggfaces=[1],
                             places365=[1])
                             #brodatz=[2, 6],
                             #mpeg7=[2, 6],
                             #multimodal=[2, 11])
#                             imagenet=[1])

baseline_map = dict(ran=baseline_ran,
                    ranp=baseline_ran,
                    fulltop=baseline_fulltop,
                    halftop=baseline_halftop)

