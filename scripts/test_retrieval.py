
# coding: utf-8

# In[1]:


import sys, os
import time
sys.path.append('../')
sys.path.append('/home/alberto/phD/projects/performance_prediction/ret-mr-retrieval_lite/')


# In[2]:


import numpy as np
np.set_printoptions(precision=3, linewidth=300, suppress=True)
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import cv2.flann

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors



from libretrieval.features.io import load_features


k = 10

features = load_features("/home/alberto/phD/datasets/places365/vgg16/features/collection/")
features_l2 = normalize(features, 'l2', axis=1)

vars = np.var(features, axis=0, dtype=np.float64)
print(vars.dtype)

q_features = load_features("/home/alberto/phD/datasets/places365/vgg16/features/query/")
q_features_l2 = normalize(q_features, 'l2', axis=1)


brute_ngbrs = NearestNeighbors(n_neighbors=10000, algorithm='brute', metric=sys.argv[1], n_jobs=1)
_ = brute_ngbrs.fit(features)


print(brute_ngbrs.get_params())
ts = time.perf_counter()
print("Brute Query -- ", end='', flush=False)
dists_sk, indices_sk = brute_ngbrs.kneighbors(q_features[0:k], n_neighbors=10000)
print("  Elapsed {0:0.8f}s".format(time.perf_counter() - ts))
