from config import cfg
from dataset import *
from model import net
from utils import random_flip,random_flip4test
import mxnet as mx
from mxnet import nd
import math
import numpy as np
import time
from tqdm import tqdm
import os

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(cfg.ctx)
        label = label.as_in_context(cfg.ctx)

        # data=random_flip4test(data)
        # data=nd.array(data,ctx=cfg.ctx)
        output = net(data)
        output.wait_to_read()
        predictions = nd.argmax(output, axis=1)
        # predictions = np.argmax(np.bincount(predictions.asnumpy().astype(np.int32)))
        # predictions = nd.array([predictions],ctx=cfg.ctx)

        acc.update(preds=predictions, labels=label)
    nd.waitall()
    return acc.get()[1]

def confusion_matrix(pred,gt):
    conf_matrix=nd.zeros((cfg.num_classes,cfg.num_classes),ctx=gt.context)
    pred=pred.reshape((-1,1))
    gt=gt.reshape((-1,1))

    index=nd.concat(pred,gt,dim=1)
    index=index.asnumpy()
    for ind in index:
        conf_matrix[int(ind[0]),int(ind[1])]+=1
    
    return conf_matrix

def dice_loss(pred,gt):
    pred=pred.argmax(axis=1)
    pred=pred.one_hot(cfg.num_classes)
    gt=gt.one_hot(cfg.num_classes)

    inter=(pred*gt).sum()
    outer=gt.shape[0]
    return_loss=1.0-inter/outer

    return return_loss

def loss_cls(pred,cls_target):
    pred=-mx.nd.log_softmax(pred)

    cls_target_onehot=cls_target.reshape((-1,1)).one_hot(pred.shape[1])
    cls_target_onehot=cls_target_onehot.transpose((1,0,2))[0]

    ratio_array=nd.array([1.2,1.18,3.2,2.5,0.5,10],ctx=pred.context).reshape((1,-1)).broadcast_to(shape=(pred.shape[0],cfg.num_classes))

    return (pred*cls_target_onehot*ratio_array).sum()/cls_target.shape[0]

def test_transformation(data,label):
    return nd.array(data/255.0).astype('float32'),nd.array([label]).asscalar().astype('float32')

def train_transformation(data, label):
    data = random_flip(data)
    data = nd.array(data/255.0).astype('float32')

    if np.random.uniform() > 0.5:
        data = data.transpose((1,2,0))
        aug1 = mx.image.RandomCropAug([224,224])
        data = aug1(data)
        data = mx.image.imresize(data, 256, 256)
        data = data.transpose((2,0,1))

    return data,nd.array([label]).asscalar().astype('float32')
    # return data,label

test_dataset = DataSet(img_dir=cfg.img_dir,
                        dataset_index=cfg.test_index,
                        transform=test_transformation)

test_datait = mx.gluon.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)


train_dataset = DataSet(img_dir=cfg.img_dir,
                        dataset_index=cfg.train_index,
                        transform=train_transformation)
# for i in range(len(train_dataset)):
#         show_images(*train_dataset[i],train_dataset)

train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

# softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
net.collect_params().load('Models/save_models/valiacc-0.988.gluonmodel',ctx=cfg.ctx)
net.hybridize()

# vali_acc=evaluate_accuracy(test_datait, net)
# trainer = mx.gluon.Trainer(net.collect_params(), 'adadelta', {'learning_rate' :math.pow(10,-4),'wd': 0.025,'rho': 0.95})
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd',
                             {'learning_rate' :math.pow(10,-4) ,
                              'wd': 0.05,
                              'momentum': 0.9})
ctx=cfg.ctx
valiacc=0.0

for epoch in range(100):
    sum_conf=nd.zeros((cfg.num_classes,cfg.num_classes),ctx=ctx)

    st=time.clock()
    train_loss=0.
    train_acc = mx.metric.Accuracy()

    if epoch>50:
        trainer.set_learning_rate(math.pow(10,-5))
    # if epoch>500:
    #     trainer.set_learning_rate(math.pow(10,-6))
    
    for data, label in tqdm(train_datait):

        data = data.as_in_context(ctx)
        # _n, _c, h, w = data.shape
        label = label.as_in_context(ctx)

        with mx.autograd.record():
            output=net(data)
            loss1=loss_cls(output,label)
            loss2=dice_loss(output,label)
            loss=loss1+loss2
            # print(loss)
            # print(nd.sum(loss).asscalar())
        # if (loss2.asscalar()!=0):
        #     conf=confusion_matrix(output.argmax(axis=1),label)
        #     sum_conf+=conf
        
        train_loss += nd.sum(loss).asscalar()   
        loss.backward()

        predictions = nd.argmax(output, axis=1)
        train_acc.update(preds=predictions, labels=label)

        trainer.step(data.shape[0])

        if (loss2.asscalar()!=0):
            conf=confusion_matrix(output.argmax(axis=1),label)
            sum_conf+=conf

            # for i in range(2):
            #     with mx.autograd.record():
            #         output=net(data)
            #         loss1=loss_cls(output,label)
            #         loss2=dice_loss(output,label)
            #         loss=loss1+loss2
            #     loss.backward()
            #     trainer.step(data.shape[0])

    nd.waitall()
   
    vali_acc=evaluate_accuracy(test_datait, net)
    # vali_acc=-1

    en=time.clock()
    print("epoch %d, loss: %f, train_acc: %f, vali_acc: %f,time: %.2f s" %(
        epoch, train_loss/1500, train_acc.get()[1], vali_acc,(en-st)))

    with open('confusion_matrix.txt','a+') as f:
        CM=sum_conf.asnumpy().astype(np.int32)
        f.write('epoch:'+str(epoch)+':\n')
        for i in range(cfg.num_classes):
            for j in range(cfg.num_classes):
                f.write(str(CM[i,j])+' ')
            f.write('\n')

    print(sum_conf)
    if vali_acc>valiacc:
        if os.path.exists(cfg.model_path.format(valiacc)):
            os.remove(cfg.model_path.format(valiacc))
        net.collect_params().save(cfg.model_path.format(vali_acc)) 
        valiacc=vali_acc
    # net.collect_params().save(cfg.model_path.format(train_acc.get()[1],train_loss/2000))