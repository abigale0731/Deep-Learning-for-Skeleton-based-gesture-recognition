import numpy as np
import pickle
import torch

data=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_train.npy")

label=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_train_label.npy")
x=list()
y=list()
for i in range(len(label)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    x.append(temp_str)
    y.append(np.argmax(label[i]))

key_value=dict()
lists = [[] for i in range(4)]
labels = [[] for i in range(4)]
for i in range(len(x)):
    key_value[i]=y[i]
key_value=sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]))
i=0
for key,value in key_value:
    lists[i%4].append(key)
    labels[i%4].append(value)
    i+=1



#data1
traina1=[data[lists[0][i]]for i in range(len(lists[0]))]
traina2=[data[lists[1][i]]for i in range(len(lists[1]))]
traina3=[data[lists[2][i]]for i in range(len(lists[2]))]
traina=np.concatenate((traina1, traina2), axis = 0)
traina=np.concatenate((traina, traina3), axis = 0)
traina = np.array(traina)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train1_data.npy',traina)
labelsa=labels[0]+labels[1]+labels[2]
xta=list()
for i in range(len(labelsa)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xta.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train1_label.pkl', 'wb') as f:
    pickle.dump((xta, labelsa), f)

vala=[data[lists[3][i]]for i in range(len(lists[3]))]
vala = np.array(vala)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val1_data.npy',vala)
xva=list()
for i in range(len(labels[3])):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xva.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val1_label.pkl', 'wb') as f:
    pickle.dump((xva, labels[3]), f)

#data2
trainb1=[data[lists[0][i]]for i in range(len(lists[0]))]
trainb2=[data[lists[1][i]]for i in range(len(lists[1]))]
trainb3=[data[lists[3][i]]for i in range(len(lists[3]))]
trainb=np.concatenate((trainb1, trainb2), axis = 0)
trainb=np.concatenate((trainb, trainb3), axis = 0)
trainb = np.array(trainb)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train2_data.npy',trainb)
labelsb=labels[0]+labels[1]+labels[3]
xtb=list()
for i in range(len(labelsb)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xtb.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train2_label.pkl', 'wb') as f:
    pickle.dump((xtb, labelsb), f)

valb=[data[lists[2][i]]for i in range(len(lists[2]))]
valb = np.array(valb)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val2_data.npy',valb)
xvb=list()
for i in range(len(labels[2])):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xvb.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val2_label.pkl', 'wb') as f:
    pickle.dump((xvb, labels[2]), f)

#data3
trainc1=[data[lists[0][i]]for i in range(len(lists[0]))]
trainc2=[data[lists[2][i]]for i in range(len(lists[2]))]
trainc3=[data[lists[3][i]]for i in range(len(lists[3]))]
trainc=np.concatenate((trainc1, trainc2), axis = 0)
trainc=np.concatenate((trainc, trainc3), axis = 0)
trainc = np.array(trainc)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train3_data.npy',trainc)
labelsc=labels[0]+labels[2]+labels[3]
xtc=list()
for i in range(len(labelsc)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xtc.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train3_label.pkl', 'wb') as f:
    pickle.dump((xtc, labelsc), f)

valc=[data[lists[1][i]]for i in range(len(lists[1]))]
valc = np.array(valc)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val3_data.npy',valc)
xvc=list()
for i in range(len(labels[1])):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xvc.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val3_label.pkl', 'wb') as f:
    pickle.dump((xvc, labels[1]), f)

#data4
traind1=[data[lists[1][i]]for i in range(len(lists[1]))]
traind2=[data[lists[2][i]]for i in range(len(lists[2]))]
traind3=[data[lists[3][i]]for i in range(len(lists[3]))]
traind=np.concatenate((traind1, traind2), axis = 0)
traind=np.concatenate((traind, traind3), axis = 0)
traind = np.array(traind)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train4_data.npy',traind)
labelsd=labels[1]+labels[2]+labels[3]
xtd=list()
for i in range(len(labelsd)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xtd.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/train4_label.pkl', 'wb') as f:
    pickle.dump((xtd, labelsd), f)

vald=[data[lists[0][i]]for i in range(len(lists[0]))]
vald = np.array(vald)
np.save('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val4_data.npy',vald)
xvd=list()
for i in range(len(labels[0])):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    xvd.append(temp_str)
with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/split/val4_label.pkl', 'wb') as f:
    pickle.dump((xvd, labels[0]), f)

