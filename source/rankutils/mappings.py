#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from rankutils.baselines import *

# Maps available descriptor numbers for each dataset
descriptor_map = OrderedDict(#oxford=[1],
                             #oxford100k=[1],
                             #unicamp=[1],
                             #unicamp100k=[1],
                             vggfaces=[1, 2],
                             places365=[1, 2],
                             imagenet=[1])
                             #test=[1])
                             #brodatz=[2, 6],
                             #mpeg7=[2, 6],
                             #multimodal=[2, 11])
#                             )

exkey_map = OrderedDict(oxford_desc1="Oxford SURF 3000x500",
                        oxford100k_desc1="OXF",
                        unicamp_desc1="Unicamp SURF 3000x500",
                        unicamp100k_desc1="UNI",
                        vggfaces_desc1="VGGF",
                        vggfaces_desc2="VGGF+P",
                        places365_desc1="P365_VGG16",
                        places365_desc2="P365_RES",
                        imagenet_desc1="ImageNET Deep Features ResnetV2",
                        test_desc1="Test Descriptor 1")

baseline_map = dict(ran=baseline_ran,
                    ranp=baseline_ran,
                    fulltop=baseline_fulltop,
                    halftop=baseline_halftop,
                    maxn=baseline_maxnacc)

