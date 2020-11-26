import datetime

import numpy as np

print(datetime.datetime.now())
f = np.load("/home/rick/dev/python/mjaigym_ml/output/localfs/supervised_dataset/feature/2017010118gm-00e1-0000-687b081f.mjson.npz")
all_keys = f.keys()
for key in all_keys:
    # print(key)
    s = f[key].shape

print(len(all_keys), s)
print(datetime.datetime.now())
