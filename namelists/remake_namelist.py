import sys, os
sys.path.append('../source/')

from rankutils.utilities import get_classname

import numpy as np

def get_classname_vggf(name):

    parts = name.rsplit('_', 2)

    return parts[0]

def get_classname_all(name):

    parts = name.rsplit('_', 1)

    return parts[0]

namelistfile = sys.argv[1]
outfile = sys.argv[2]
try:
    isvggf = bool(sys.argv[3])
except IndexError:
    isvggf = False

dt = dict(names=('name', 'numfeat'), formats=('U100', np.int32))
namearray = np.loadtxt(namelistfile, dtype=dt)

class_map = dict()
class_id = 0

class_array = np.zeros(namearray['name'].size, dtype=np.int32)

for i, name in enumerate(namearray['name']):

    if isvggf:
        nameclass = get_classname_vggf(name)
    else:
        nameclass = get_classname_all(name)
    if nameclass not in class_map:
        class_map[nameclass] = class_id
        class_array[i] = class_id
        class_id += 1
    else:
        class_array[i] = class_map[nameclass]

outdt = dict(names=('name', 'numfeat', 'cid'), formats=('U100', np.int32, np.int32))
aux = zip(namearray['name'], namearray['numfeat'], class_array)
outarray = np.array([a for a in aux], dtype=outdt)

np.savetxt(outfile, outarray, fmt="%-50s %05d %03d")