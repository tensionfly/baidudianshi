import os
import cv2
import numpy as np
import mxnet as mx

class DataSet(mx.gluon.data.Dataset):

    class_name =['OCEAN','MOUNTAIN','LAKE','FARMLAND','DESERT','CITY']

    def __init__(self,img_dir: str, dataset_index:str, transform=None, **kwargs):

        super(DataSet, self).__init__(**kwargs)

        with open(dataset_index) as f:
            self.dataset_index = [t.split() for t in f.readlines()]

        self.img_dir = img_dir
        self.transform = transform
    
    def __getitem__(self, idx):
        idx = self.dataset_index[idx]
        img_path = os.path.join(self.img_dir,idx[0])
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(256,256))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))

        label=int(idx[1])
        
        if self.transform is None:
            return img, label
        else:
            return self.transform(img, label)

    def __len__(self):
        return len(self.dataset_index)

def show_images(data,label,ds:DataSet):
    img = np.transpose(data, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print(ds.class_name[int(label)])

    cv2.imshow("Img", img)
    cv2.waitKey(0)