import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import Resnet18WithConfBarWithSpeciesPawpularityRegressor
from dataloaders.regression_dataloader import PawpularityWithSpeciesWithConfBarDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from callbacks import CheckpointCallback
from torchsummary import summary
import os
from metric import RMSE
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
import math
import random
# transform=get_transform_pipeline(224,224,False)

# pretrained_weight='exp/100/pawpularity_resnet18100.pt'
# model=Resnet18WithConfBarWithSpeciesPawpularityRegressor()

# model.to('cuda')
# model.eval()
# model.load_state_dict(torch.load(pretrained_weight))

# print(summary(model,input_data=(3,224,224)))
# target_df=pd.read_csv('Dataset/train_dog_cat.csv')
data_dir='Dataset/train'
meta_csv='Dataset/train.csv'
species_csv='Dataset/train_dog_cat.csv'
transform_dict={
    'train':get_transform_pipeline(224,224,False),
    'valid':get_transform_pipeline(224,224,False)
}
splitter=PawpularityWithSpeciesWithConfBarDatasetSplitter(img_dir=data_dir,meta_csv=meta_csv,species_csv=species_csv,transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(splitter.generate_train_valid_dataset(0.8)))

target_df=pd.read_csv('Dataset/train.csv')
target_df=target_df[['Id','Pawpularity']]
# seeds=[]
seeds=[]
means=[]
stds=[]
rmses=[]
for i in range(1000):
    np.random.seed(i)
    random.seed(i)
    target_df=pd.read_csv('Dataset/train.csv')
    target_df=target_df[['Id','Pawpularity']]
    for mean_val in range(0,101):
        for std_val in range(0,101):
            df={'Id':target_df['Id'].to_list(),'Pawpularity':np.random.normal(mean_val,std_val,(10,target_df.shape[0])).mean(axis=0).tolist()}
            df=pd.DataFrame(df)
            rmse_val=math.sqrt(mean_squared_error(df['Pawpularity'],target_df['Pawpularity']))
            print("SEED: ",str(i),"MEAN: ",mean_val,"STD: ",std_val," RMSE: ",math.sqrt(mean_squared_error(df['Pawpularity'],target_df['Pawpularity'])))
            rmses.append(rmse_val)
            seeds.append(i)
            means.append(mean_val)
            stds.append(std_val)
    # exit()
best_idx=np.array(rmses).argmin()
print("RMSE: ",rmses[best_idx])
print("MEAN: ",means[best_idx])
print("STD: ",stds[best_idx])
print("SEED: ",seeds[best_idx])

##BEST SEED 878
##BEST MEAN 38
##BEST STD 4