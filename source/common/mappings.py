#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from common.baselines import *

# Maps available descriptor numbers for each dataset
descriptor_map = OrderedDict(oxford=[1],
                             oxford100k=[1],
                             unicamp=[1],
                             unicamp100k=[1],
                             vggfaces=[1, 2],
                             places365=[1, 2],
                             imagenet=[1])
                             #test=[1])
                             #brodatz=[2, 6],
                             #mpeg7=[2, 6],
                             #multimodal=[2, 11])
#                             )

key_map = OrderedDict(oxford_desc1="Oxford SURF 3000x500",
                      oxford100k_desc1="Oxford100k SURF 2000",
                      unicamp_desc1="Unicamp SURF 3000x500",
                      unicamp100k_desc1="Unicamp100k SURF 2000",
                      vggfaces_desc1="VGGfaces Deep Features",
                      vggfaces_desc2="VGGfaces Deep Features + Perturbations",
                      places365_desc1="Places365 Deep Features VGG16",
                      places365_desc2="Places365 Deep Features Resnet152",
                      imagenet_desc1="ImageNET Deep Features ResnetV2",
                      test_desc1="Test Descriptor 1")
                             #brodatz=[2, 6],
                             #mpeg7=[2, 6],
                             #multimodal=[2, 11])
#                             imagenet=[1])

baseline_map = dict(ran=baseline_ran,
                    ranp=baseline_ran,
                    fulltop=baseline_fulltop,
                    halftop=baseline_halftop,
                    maxn=baseline_maxnacc)

