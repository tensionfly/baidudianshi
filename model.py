import mxnet as mx
from config import cfg

# pretrained_model=mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True,ctx=cfg.ctx)
pretrained_model=mx.gluon.model_zoo.vision.resnet50_v2()
# pretrained_model=mx.gluon.model_zoo.vision.alexnet()

net=mx.gluon.nn.HybridSequential()
net.add(pretrained_model.features,mx.gluon.nn.Dense(6))
# net[-1].initialize(ctx=cfg.ctx,init=mx.init.Xavier(magnitude=2.24))
# print(pretrained_model)