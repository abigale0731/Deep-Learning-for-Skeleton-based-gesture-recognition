import numpy as np
import pickle
import os
import time


def prepocess(data):
    size = data.shape[0]
    data.resize(size, 39, 19, 3)
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 3)
    old = np.expand_dims(data, axis=-1)
    return old

train=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_train.npy")
train=prepocess(train)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/old_method/train_data.npy',train)

test=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_val.npy")
test=prepocess(test)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/old_method/test_data.npy',test)