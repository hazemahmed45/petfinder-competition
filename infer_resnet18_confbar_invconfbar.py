import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import Resnet18WithConfBarWithInvConfBarPawpularityRegressor
from dataloaders.regression_dataloader import PawpularityWithConfbarWithInvConfBarDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import numpy as np
import math
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
transform=get_transform_pipeline(224,224,False)

pretrained_weight='exp/109/pawpularity_resnet18_109.pt'
transform_dict={
    'train':get_transform_pipeline(224,224,True),
    'valid':get_transform_pipeline(224,224,False)
}
splitter=PawpularityWithConfbarWithInvConfBarDatasetSplitter(img_dir='Dataset/train',meta_csv='Dataset/train.csv',transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(splitter.generate_train_valid_dataset(0.8)))

model=Resnet18WithConfBarWithInvConfBarPawpularityRegressor()
model.to('cuda')

model.load_state_dict(torch.load(pretrained_weight))
# model.fc_conf_bar=torch.nn.Sequential(*list(model.fc_conf_bar.children())[:-1])
# model.fc_inv_conf_bar=torch.nn.Sequential(*list(model.fc_inv_conf_bar.children())[:-1])
print(summary(model,input_data=(3,224,224)))

target_df=pd.read_csv('Dataset/train.csv')
data_dir='Dataset/train'
if(os.path.exists('test_invconf_conf.csv')):
    df=pd.read_csv('test_invconf_conf.csv')

    conf=np.zeros((target_df.shape[0],100))
    inv_conf=np.zeros((target_df.shape[0],100))
    with open('conf.npy','rb') as f:
        conf=np.load(f)
    with open('inv_conf.npy','rb') as f:
        inv_conf=np.load(f)
    arr=[]

    for i in range(10):
        # print(conf[i])
        # print(inv_conf[i])
        s_conf=conf[i].reshape((-1,1))
        s_inv_conf=inv_conf[i].reshape((-1,1))
        print(np.concatenate((s_conf,s_inv_conf),axis=1))
        print(target_df.iloc[i]['Pawpularity'])
        print(np.array(s_inv_conf>0,dtype=int).argmax())
        # cv2.waitKey(0)

    final_pred=((np.array(conf>0,dtype=int).sum(axis=1)+np.array(inv_conf>0,dtype=int).sum(axis=1))/2)

    arr.append(math.sqrt(mean_squared_error(final_pred,target_df['Pawpularity'])))
    print(min(arr))
    exit()
else:
    df={'Id':[],'Pawpularity':[]}
    conf=np.zeros((target_df.shape[0],100))
    inv_conf=np.zeros((target_df.shape[0],100))
    for id,row in target_df.iterrows():
        img_name=row['Id']
        paw_value=row['Pawpularity']
        img_path=os.path.join(data_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        img=transform(image=img)['image']
        img=img.view((-1,3,224,224)).to('cuda')
        # print(img.shape)
        out_conf,out_inv_conf=model(img)
        out_conf=out_conf.detach().cpu().numpy()
        out_inv_conf=out_inv_conf.detach().cpu().numpy()
        # print(paw_value)
        # print(np.array(out_conf>0.4,dtype=int).sum())
        # print(np.array(out_inv_conf>0.4,dtype=int).sum())
        # print((np.array(out_conf>0.4,dtype=int).sum()+np.array(out_inv_conf>0.4,dtype=int).sum())/2)
        final_pred=(np.array(out_conf>1,dtype=int).sum()+np.array(out_inv_conf>1,dtype=int).sum())/2
        df['Id'].append(img_name)
        df['Pawpularity'].append(final_pred)
        conf[id]=out_conf
        inv_conf[id]=out_inv_conf
        # print(out_conf)
        # print(out_inv_conf)
        # exit()
        # print(pred)
        # df['Id'].append(img_name)
        # df['class'].append(0 if pred<0.5 else 1)

    df=pd.DataFrame(df)
    df.to_csv('test_invconf_conf.csv',index=None)
    with open('conf.npy','wb')as f:
        np.save(f,conf)
    with open('inv_conf.npy','wb')as f:
        np.save(f,inv_conf)

df.sort_values(by='Id',inplace=True)
target_df.sort_values(by='Id',inplace=True)

print(math.sqrt(mean_squared_error(df['Pawpularity'],target_df['Pawpularity'])))