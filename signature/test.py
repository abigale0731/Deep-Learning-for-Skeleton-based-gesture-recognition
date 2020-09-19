import numpy as np
import pickle
import torch
import signatory
import datetime
import os
import time

def logsig(step,stride,level,data):
    size = data.shape[0]
    frame=data.shape[1]
    data = data.reshape(size, 39, 19, 3)
    data = np.swapaxes(data, 1, 2)
    data = data.reshape(size * 19, 39, 3)
    sig = list()
    for i in range(0, frame - step + 1, stride):
        temp_data = data[:, i:i + step, :]
        temp_data = torch.from_numpy(temp_data).float()
        temp_sig = (signatory.logsignature(temp_data, level))
        temp_data = temp_data.cpu().numpy()
        temp_sig = temp_sig.cpu().numpy()
        temp_data = np.swapaxes(temp_data, 0, 1)
        temp_data = temp_data[0]
        att = np.concatenate((temp_data, temp_sig), axis=1)
        sig.append(att)

    if frame % stride>1:
        temp_data1 = data[:, frame-step+1:frame, :]
        temp_data1 = torch.from_numpy(temp_data1).float()
        temp_sig1 = (signatory.logsignature(temp_data1, level))
        temp_data1 = temp_data1.cpu().numpy()
        temp_sig1 = temp_sig1.cpu().numpy()
        temp_data1 = np.swapaxes(temp_data1, 0, 1)
        temp_data1 = temp_data1[0]
        att1 = np.concatenate((temp_data1, temp_sig1), axis=1)
        sig.append(att1)

    sig = np.swapaxes(sig, 0, 1)
    t1 = sig.shape[1]
    s1 = sig.shape[2]
    sig = sig.reshape(size, 19, t1, s1)
    sig = np.swapaxes(sig, 1, 3)
    sig = np.expand_dims(sig, axis=-1)
    return sig




step=2
stride=2
level=4
train=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_train.npy")
train=logsig(step,stride,level,train)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/test_sig/train_data.npy',train)

test=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_val.npy")
test=logsig(step,stride,level,test)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/test_sig/test_data.npy',test)