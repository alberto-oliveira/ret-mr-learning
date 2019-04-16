#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from rankutils.baselines import *

# Maps available descriptor numbers for each dataset
ranking_type_map = OrderedDict(places365=[1, 2, 3, 4, 5],
                               vggfaces=[1, 2, 3, 4, 5],
                               imagenet=[1, 2, 3, 4],
                               oxford=[1],
                               unicamp=[1],
                               MPEG7=[1, 2, 3, 4, 5])

baseline_map = dict(ran=baseline_ran,
                    ranp=baseline_ran,
                    fulltop=baseline_fulltop,
                    halftop=baseline_halftop,
                    maxn=baseline_maxnacc)

