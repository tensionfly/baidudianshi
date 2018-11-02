import cv2
import numpy as np

with open('DataSet/index.txt','r') as f:
    index = [t.split() for t in f.readlines()]

img_dir='DataSet/bb-pretrain-traindataset-2000/pretrain-traindataset-2000/'

rgb_dic={}
for i in range(6):
    rgb_dic[str(i)]=[]

for name in index:
    img_path=img_dir+name[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_list=[img[:,:,0].mean(),img[:,:,1].mean(),img[:,:,2].mean()]
    rgb_dic[name[1]].append(rgb_list)

for i in range(6):
    rgb=np.array(rgb_dic[str(i)]).mean(axis=0)/255.0
    rgb_dic[str(i)]=rgb
