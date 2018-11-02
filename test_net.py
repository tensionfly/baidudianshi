from model import net
from utils import random_flip4test
import mxnet as mx
from mxnet import nd
import numpy as np
import os
import cv2
from config import cfg

class_name =['OCEAN','MOUNTAIN','LAKE','FARMLAND','DESERT','CITY']
ctx=mx.cpu()

def getclass(rgb,cls_items):
    result=nd.zeros((cls_items.shape[0],),ctx)
    for i in range(cls_items.shape[0]):
        delta=(rgb-cfg.rgb_dic[str(int(cls_items[i]))]).abs().mean().asscalar()
        result[i]=delta
     
    get_cls=nd.argmin(result,axis=0).asscalar()
    cls_return=int(cls_items[int(get_cls)])

    return cls_return

def evaluate_accuracy(img_dir,net,result_path):
    k=0
    img_list=os.listdir(img_dir)
    with open(result_path,'w') as f:
        for i,img_name in enumerate(img_list):
            # print(img_name)
            img_path=img_dir+img_name
            img = cv2.imread(img_path)
            img = cv2.resize(img,(256,256))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img,axis=0)
            data=nd.array(img/255.0).astype('float32')

            data = data.as_in_context(ctx)
            data_copy=data.copy()
            
            data=random_flip4test(data)
            data=nd.array(data,ctx=ctx)

            output = net(data)
            output.wait_to_read()

            predictions = nd.argmax(output, axis=1)
            a=predictions.copy()
            # print(predictions)
            predictions = np.argmax(np.bincount(predictions.asnumpy().astype(np.int32)))
            if (a-predictions).sum().asscalar()!=0:
                print(img_name)
                print(a)
                k+=1

                num=((a==predictions)*1.0).sum()
                cls_items=set(a.asnumpy())

                if num==3 and len(cls_items)==2 and (4 in cls_items):
                    if 1 in cls_items:
                        cls_items.remove(4)
                    
                    rgb=nd.array([data_copy[0,0].mean().asscalar(),data_copy[0,1].mean().asscalar(),
                                data_copy[0,2].mean().asscalar()],ctx=ctx)
                    
                    
                    cls_items=np.array(list(cls_items))

                    get_cls=getclass(rgb,cls_items)
                    print(class_name[int(get_cls)])
                    predictions=get_cls
                    print('er')


            f.write(img_name+','+class_name[int(predictions)]+'\n')
            print(i)
    print(k)

net.collect_params().load('Models/save_models/valiacc-0.986.gluonmodel',ctx=ctx)
net.hybridize()
dir_path='DataSet/bb4testA1000/testA-1000/'
result_path='DataSet/bb4testA1000/R986.txt'

evaluate_accuracy(dir_path,net,result_path)
