import numpy as np
import random
import cv2
import mxnet as mx

def random_flip(data):
    if np.random.uniform() > 0.5:
        data = np.flip(data, axis=2)
        # print('h')

    if np.random.uniform() > 0.5:
        data = np.flip(data, axis=1)
        # print('v')
    
    if np.random.uniform() > 0.5:
        data=np.rot90(data,1,(1,2))
        # print('r')

    if np.random.uniform() > 0.5:
        data = np.transpose(data, (1, 2, 0))

        size_guass=random.randint(3,10)
        x=random.uniform(0,3)
        if size_guass%2==0:
            size_guass+=1
        data=cv2.GaussianBlur(data,(size_guass,size_guass),x)

        data = np.transpose(data, (2, 0, 1))

        # print("blu")

    return data


def random_flip4test(data):
    data=data[0].asnumpy()
    data1 = np.flip(data, axis=2)
    data2 = np.flip(data, axis=1)
    data3 = np.rot90(data,1,(1,2))
    data4 = np.rot90(data,2,(1,2))
    data5 = np.rot90(data,3,(1,2))

    # data6 = mx.nd.array(data).transpose((1,2,0))
    # aug1 = mx.image.RandomCropAug([224,224])
    # data6 = aug1(data6)
    # data6 = mx.image.imresize(data6, 256, 256)
    # data6 = data6.transpose((2,0,1)).asnumpy()
    
    data = np.expand_dims(data,axis=0)
    data1 = np.expand_dims(data1,axis=0)
    data2 = np.expand_dims(data2,axis=0)
    data3 = np.expand_dims(data3,axis=0)
    data4 = np.expand_dims(data4,axis=0)
    data5 = np.expand_dims(data5,axis=0)
    # data6 = np.expand_dims(data6,axis=0)

    data_concat=np.concatenate((data,data1,data2,data3,data4,data5),axis=0)

    return data_concat
# import os
# img_dir='DataSet/bb-pretrain-traindataset-2000/pretrain-traindataset-2000/'
# img_list=os.listdir(img_dir)

# for name in img_list:
#     path=img_dir+name
#     img=cv2.imread(path)
#     cv2.imshow("img1",img)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.transpose(img, (2, 0, 1))

#     img=random_flip(img)

#     img = np.transpose(img, (1, 2, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("img2",img)
#     cv2.waitKey(0)
