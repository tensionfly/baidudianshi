# with open('DataSet/index.txt','r') as f:
#     index_list=[t.strip() for t in f.readlines()]
# name2index={}
# for x in index_list:
#     name2index[x[0]]=x[1]

# names=['MWI_GIZC22GeRV5Wavzw.jpg','MWI_GIhNUFAA8tczfJqD.jpg','MWI_GJZ7WCu2E1GH8sCP.jpg','MWI_GK8M4doZRWGj1THz.jpg','MWI_GMtlFZ9qGKtAoeHW.jpg',
#        'MWI_GO3idfCD5jfrk0T8.jpg','MWI_GQPrNapzMbiOE0Nz.jpg','MWI_GQrP7moncsOdiefL.jpg','MWI_GRRgrqh1ioz8tw11.jpg','MWI_GS9noWIGCTefL3WL.jpg']

# import time

# for x in names:
#     st=time.clock()
#     index=name2index[x]
#     print(time.clock()-st) #-5 -6
# import random

# def getindex(file:str,index_train:str,index_test:str,ratio):
#     with open(file,'r') as f:
#         filenames_list=[t.strip() for t in f.readlines()]
    
#     num_test=int(len(filenames_list)*ratio)

#     list_test=random.sample(filenames_list,num_test)
#     list_train=list(set(filenames_list).difference(set(list_test)))

#     with open(index_train,'w') as f:
#         for i in range(len(list_train)):
#             f.write(list_train[i]+'\n')
    
#     with open(index_test,'w') as f:
#         for i in range(len(list_test)):
#             f.write(list_test[i]+'\n')

# getindex('DataSet/index.txt','DataSet/train_index.txt','DataSet/test_index.txt',0.25)

import os
dir_path='DataSet/bb4testA1000/testA-1000/'
img_list=os.listdir(dir_path)
with open('DataSet/name_list4test.txt','w') as f:
    for name in img_list:
        f.write(name+' '+str(-1)+'\n')
print('done!')