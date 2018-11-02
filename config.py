import mxnet as mx
class _Config:
    # Dataset config
    img_dir='DataSet/bb-pretrain-traindataset-2000/pretrain-traindataset-2000/'

    train_index='DataSet/train_index.txt'
    test_index='DataSet/test_index.txt'

    ctx=mx.gpu()
    num_classes=6
    batch_size=8
    model_path='Models/valiacc-{:.5}.gluonmodel'

    rgb_0=mx.nd.array([0.27619422, 0.32709425, 0.35705303],ctx=ctx)
    rgb_1=mx.nd.array([0.39110827, 0.39011621, 0.34046577],ctx=ctx)
    rgb_2=mx.nd.array([0.43456235, 0.42935566, 0.38999044],ctx=ctx)
    rgb_3=mx.nd.array([0.45051188, 0.45451944, 0.39119266],ctx=ctx)
    rgb_4=mx.nd.array([0.60290523, 0.54860175, 0.48185672],ctx=ctx)
    rgb_5=mx.nd.array([0.48350936, 0.50148165, 0.49328902],ctx=ctx)


    rgb_dic={'0': rgb_0, '1': rgb_1, '2': rgb_2, '3': rgb_3, '4': rgb_4, '5': rgb_5}

cfg=_Config()