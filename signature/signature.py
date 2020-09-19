import numpy as np
import pickle
import torch
import signatory
import datetime
import os
import time

def sig(step,stride,level,data):
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
        temp_sig1 = (signatory.signature(temp_data1, level))
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
    return s1,sig



img_dir=r"/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split"
time_list = list()
steps =[2,3]
levels =[1,2,3,4]
save_dir_base = "/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/sig_method"
lists_train = [[] for i in range(4)]
lists_val = [[] for i in range(4)]
lists_time = [[] for i in range(4)]
for root, dirs, files in os.walk(img_dir, topdown=False):
    for name in files:

        base_name = name[:-4]
        class_name = name[-4:]
        if class_name != ".npy":
            continue
        num = int(base_name.split("_")[0][-1:])
        for step in steps:
            for level in levels:
                strides = [1,step]
                for stride in strides:
                    startTime = time.time()
                    img_path = img_dir + "//" + name
                    data = np.load(img_path)
                    s1,sig = sig(step, stride, level, data)
                    save_dir = os.path.join(save_dir_base,base_name)+"_step"+str(step)+"_level"+str(level)+"_stride"+str(stride)+"_channel"+str(s1)+".npy"
                    np.save(save_dir,sig)
                    endTime = time.time()
                    temp_time = endTime - startTime
                    if(base_name[:3]=="tra"):
                        lists_train[num-1].append(temp_time)
                    else:
                        lists_val[num-1].append(temp_time)

print(lists_train,lists_val)