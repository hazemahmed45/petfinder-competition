import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import Resnet18Backbone
from dataloaders.classification_dataloaders import OpenImageDogCatDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score
transform=get_transform_pipeline(224,224,False)

pretrained_weight='exp/75/pawpularity_resnet18_classifier_75.pt'
backone=Resnet18Backbone()
model=torch.nn.Sequential(
    backone,
    torch.nn.Linear(backone.emb_dim,512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,1),
    
    )                     
model.to('cuda')
model.eval()
model.load_state_dict(torch.load(pretrained_weight))

print(summary(model,input_data=(3,224,224)))

target_df=pd.read_csv('Dataset/train_dog_cat.csv')
data_dir='Dataset/train'
df={'Id':[],'class':[]}
for img_name in sorted(os.listdir(data_dir)):
    img_path=os.path.join(data_dir,img_name)
    img=cv2.imread(img_path)
    # cv2.imshow('',img)
    # cv2.waitKey(0)
    img=transform(image=img)['image']
    img=img.view((-1,3,224,224)).to('cuda')
    # print(img.shape)
    pred=torch.sigmoid(model(img)).detach().cpu().item()
    # print(pred)
    df['Id'].append(img_name)
    df['class'].append(0 if pred<0.5 else 1)

df=pd.DataFrame(df)

df.sort_values(by='Id',inplace=True)
target_df.sort_values(by='Id',inplace=True)

print(accuracy_score(df['class'],target_df['class']))