import numpy as np
import pickle


label=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_train_label.npy")
x=list()
y=list()
for i in range(len(label)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    x.append(temp_str)
    y.append(np.argmax(label[i]))

with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original_label/train_label.pkl', 'wb') as f:
    pickle.dump((x, list(y)), f)



label=np.load("/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original/cha_val_label.npy")
x=list()
y=list()
for i in range(len(label)):
    temp_str=str(i)
    temp_str = temp_str.zfill(4)
    x.append(temp_str)
    y.append(np.argmax(label[i]))

with open('/Users/caoyue/Desktop/Sig-ST-GCN/data/chalearn2013/original_label/test_label.pkl', 'wb') as f:
    pickle.dump((x, list(y)), f)