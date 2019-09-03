#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from rankutils.baselines import *

# Maps available descriptor numbers for each dataset
ranking_type_map = OrderedDict(places365=[1, 2, 3],
                               vggfaces=[1, 2],
                               imagenet=[1, 2, 3],
                               oxford=[1, 2, 3, 4, 5],
                               unicamp=[1, 2, 3, 4, 5],
                               corel=[1, 2])#,
                               #MPEG7=[1, 2, 3, 4, 5],
                               #multimodal=[3, 4, 6, 10, 12])

baseline_map = dict(ran=baseline_ran,
                    ranp=baseline_ran,
                    fulltop=baseline_fulltop,
                    halftop=baseline_halftop,
                    maxn=baseline_maxnacc)

exp_aliases = dict(oxford_001='oxford.001.SURF-500-L2-VOTES', oxford_002='oxford.002.SURF-500-L2-COMB',
                   oxford_003='oxford.003.SURF-2000-L2-VOTES', oxford_004='oxford.004.resnetv2-L2',
                   oxford_005='oxford.005.vgg16-L2', unicamp_001='Unicamp SURF 500',
                   unicamp_002='unicamp.002.SURF-500-L2-COMB', unicamp_003='unicamp.003.SURF-2000-L2-VOTES',
                   unicamp_004='Unicamp ResnetV2', unicamp_005='unicamp.005.vgg16-L2',
                   corel_001='corel.001.resnetv2-L2', corel_002='corel.002.vgg16-L2',
                   places365_001='Places365 VGG16 $L_2$', places365_002='Places365 ResnetV2 $L_2$',
                   places365_003='places365.003.vgg16-Cos', places365_004='P365 VGG16-Cheby',
                   places365_005='P365 VGG16-Canb', vggfaces_001='VGGFace VGG16',
                   vggfaces_002='VGGFace VGG16 + Pert.', vggfaces_003='VGGF VGG16-Cos',
                   vggfaces_004='VGGF VGG16-Cheb', vggfaces_005='VGGF VGG16-Canb',
                   imagenet_001='ImageNet ResnetV2 $L_2$', imagenet_004='INET Rv2-Canb',
                   imagenet_003='ImageNet ResnetV2 $L_{\infty}$', imagenet_002='imagenet.002.resnetv2-Cos',
                   MPEG7_001='MPEG-7 BAS', MPEG7_002="MPEG-7 IDSC",
                   MPEG7_003="MPEG-7 ASC", MPEG7_004="MPEG-7 AIR",
                   MPEG7_005="MPEG-7 CFD", multimodal_003="Text FV DICE",
                   multimodal_004="Text FV BoW", multimodal_006="Text FV Jaccard",
                   multimodal_010="Img. FV CSD", multimodal_012="Img. FV Color Bmap")
