# import sys
# sys.path.append('../augmentation.py')

import torch
from torch.functional import split
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,StratifiedGroupKFold
# from augmentation import get_transform_pipeline,get_torch_transform_pipeline
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
import seaborn as sb
import matplotlib.pyplot as plt
from torchvision.transforms import Compose as t_compose
from albumentations import Compose as a_compose
from enum import Enum



SEED = 999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CATEGORY_TO_LABEL_PATH='categories_to_label.json'
LABEL_TO_CATEGORY_PATH='label_to_categories.json'

class PawpularityDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name,Subject_Focus,Eyes,Face,Near,Action,Accessory,Group,Collage,Human,Occlusion,Info,Blur=self.dataframe.iloc[index,:-1]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        return img,torch.tensor(paw_value,dtype=torch.float32)
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return 
class PawpularityWithMetaDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index,0]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        # print(meta.shape)
        img=img/255.0
        return img.float(),torch.tensor(meta).float(),torch.tensor(paw_value,dtype=torch.float32)
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return 
class PawpularityWithMetaWithConfidenceBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index,0]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        paw_index=round(paw_value*100)
        confidence_bar=np.zeros((1,100))+np.concatenate((np.ones((int(paw_index))),np.zeros((int(100-paw_index)))))
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(meta).float(),torch.tensor(paw_value,dtype=torch.float32),torch.tensor(confidence_bar,dtype=torch.float32).view((-1))
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return   
    
class PawpularityWithMetaWithBinsBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None,bin_increment=5):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms
        self.bin_increment=bin_increment
        self.bin_num=100//self.bin_increment
        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index,0]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        paw_index=round(paw_value*100)
        bins=np.zeros((self.bin_num))
        bins[paw_index%self.bin_num]=1
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(meta).float(),torch.tensor(paw_value,dtype=torch.float32),torch.tensor(bins,dtype=torch.float32).view((-1))
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return       
class PawpularityWithMetaWithConfidenceBinsBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None,bin_increment=5):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms
        self.bin_increment=bin_increment
        self.bin_num=100//self.bin_increment
        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index,0]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        paw_index=round(paw_value*100)
        bins=np.zeros((self.bin_num))
        print(bins.shape)
        print(paw_index)
        print(paw_index%self.bin_num)
        
        bins[:(paw_index%self.bin_num)+1]=1#np.ones((paw_index%self.bin_num))
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(meta).float(),torch.tensor(paw_value,dtype=torch.float32),torch.tensor(bins,dtype=torch.float32).view((-1))
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return       
class PawpularityWithConfidenceBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index,0]
        paw_value=self.dataframe.iloc[index,-1]
        
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        paw_index=round(paw_value*100)
        confidence_bar=np.zeros((1,100))+np.concatenate((np.ones((int(paw_index))),np.zeros((int(100-paw_index)))))
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(paw_value,dtype=torch.float32),torch.tensor(confidence_bar,dtype=torch.float32).view((-1))
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return     
class PawpularityWithSpeciesWithConfBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index]['Id']
        paw_value=self.dataframe.iloc[index]['Pawpularity']
        species_type=self.dataframe.iloc[index]['species']
        # print(paw_value,species_type)
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        meta=self.dataframe.iloc[index,1:-1].to_numpy(dtype=np.float32).reshape((1,-1))
        paw_index=round(paw_value*100)
        confidence_bar=np.zeros((1,100))+np.concatenate((np.ones((int(paw_index))),np.zeros((int(100-paw_index)))))
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(paw_value,dtype=torch.float32),torch.tensor(confidence_bar,dtype=torch.float32).view((-1)),torch.tensor(species_type).float()
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return     
class PawpularityWithConfBarWithInvConfBarDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms

        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index]['Id']
        paw_value=self.dataframe.iloc[index]['Pawpularity']
        # print(paw_value,species_type)
        img_path=os.path.join(self.img_dir,img_name+'.jpg')
        img=cv2.imread(img_path)
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        paw_index=round(paw_value*100)
        confidence_bar=np.zeros((1,100))+np.concatenate((np.ones((int(paw_index))),np.zeros((int(100-paw_index)))))
        inv_confidence_bar=np.zeros((1,100))+np.concatenate((np.zeros((int(paw_index-1))),np.ones((int(100-paw_index)+1))))
        # confidence_bar[:,paw_index:min(paw_index+5,100)]=np.linspace(0.9,0.1,5)[:(min(paw_index+5,100)-paw_index)]
        return img.float(),torch.tensor(confidence_bar,dtype=torch.float32).view((-1)),torch.tensor(inv_confidence_bar,dtype=torch.float32).view((-1))
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return     



class PawpularityDatasetSplitter():
    def __init__(self,img_dir,meta_csv:str,transforms_dict:dict):
        self.img_dir=img_dir
        self.data_df=pd.read_csv(meta_csv)
        self.transform_dict=transforms_dict
        # self.new_df=pd.DataFrame()
        # for i in range(1,101):
        #     sub_data=self.data_df[self.data_df['Pawpularity']==i]
        #     # print(sub_data.shape,sub_data.iloc[:min(sub_data.shape[0],70)].shape)
        #     # print()
        #     self.new_df=pd.concat([self.new_df,sub_data.iloc[:min(sub_data.shape[0],70)]])
        # self.data_df=self.new_df
            
        
        # self.with_meta=with_meta
        self.data_df.iloc[:,-1]=self.normalize(self.data_df.iloc[:,-1])
        # print(self.Y.describe())
        self.k_folder=KFold()
        # self.with_conf_bar=with_conf_bar
        return 
    def generate_train_valid_dataset(self,train_split=0.8):
        # train_idx,valid_idx=train_test_split(np.arange(self.data_df.shape[0]),train_size=train_split,stratify=self.data_df.iloc[:,-1])
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

    def normalize(self,series:pd.Series) -> pd.Series:
        
        return (series-0)/(series.max()-0)
    
    
class PawpularityWithMetaDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict):
        super().__init__(img_dir, meta_csv, transforms_dict)
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
class PawpularityWithMetaWithConfidenceDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict):
        super().__init__(img_dir, meta_csv, transforms_dict)
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
class PawpularityWithConfidenceDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict):
        super().__init__(img_dir, meta_csv, transforms_dict)
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
class PawpularityWithConfbarWithInvConfBarDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict):
        super().__init__(img_dir, meta_csv, transforms_dict)
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithConfBarWithInvConfBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithConfBarWithInvConfBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

class PawpularityWithMetaWithBinsDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict,bin_increment:int):
        super().__init__(img_dir, meta_csv, transforms_dict)
        self.bin_increment=bin_increment
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithMetaWithBinsBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithBinsBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
class PawpularityWithBinsDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict,bin_increment:int):
        super().__init__(img_dir, meta_csv, transforms_dict)
        self.bin_increment=bin_increment
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithMetaWithBinsBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithBinsBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

class PawpularityWithMetaWithConfidenceBinsDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str, transforms_dict: dict,bin_increment:int):
        super().__init__(img_dir, meta_csv, transforms_dict)
        self.bin_increment=bin_increment
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithMetaWithConfidenceBinsBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithConfidenceBinsBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

class PawpularityWithSpeciesWithConfBarDatasetSplitter(PawpularityDatasetSplitter):
    def __init__(self, img_dir, meta_csv: str,species_csv, transforms_dict: dict):
        super().__init__(img_dir, meta_csv, transforms_dict)
        species_df=pd.read_csv(species_csv)
        self.data_df.sort_values(by=['Id'],inplace=True)
        species_df.sort_values(by=['Id'],inplace=True)
        self.data_df['species']=species_df['class']
        # print(self.data_df)
    def generate_train_valid_dataset(self, train_split=0.8):
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield PawpularityWithSpeciesWithConfBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithSpeciesWithConfBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

class RandomGeneratorDataset(Dataset):
    def __init__(self,csv_file,mean=38,std=20,droplessthan=90) -> None:
        super().__init__()
        self.df=pd.read_csv(csv_file)
        self.df=self.df[['Id','Pawpularity']]
        self.droplessthan=droplessthan
        self.keys=[]
        for key,value in self.df['Pawpularity'].value_counts().items():
            if(value>self.droplessthan):
                self.keys.append(key)
        self.newdf=None
        for key in self.keys:
            if(self.newdf is None):
                self.newdf=self.df[self.df['Pawpularity']==key]
            else:
                self.newdf=pd.concat((self.newdf,self.df[self.df['Pawpularity']==key]))
        # print(self.newdf['Pawpularity'].value_counts())
        self.df=self.newdf
        self.paw_values=self.df['Pawpularity'].to_numpy()
        self.mean=mean
        self.std=std
        print(self.paw_values.shape)
    def __getitem__(self, index) :
        rand_value=np.random.normal(self.mean,self.std)
        # print(rand_value)
        # print()
        paw_value=self.paw_values[np.abs(self.paw_values-rand_value).argmin()]
        return rand_value/100,paw_value/100
    def __len__(self):
        return self.df.shape[0] 
    def plot_y_hist(self):
        sb.histplot(x=self.df['Pawpularity'])
        plt.show()
        return 

# if(self.with_meta):
#                 if(self.with_conf_bar):
#                     yield PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
#                 else:
#                     yield PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
#             else:
#                 if(self.with_conf_bar):
#                     yield PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
#                 else:
if (__name__ == '__main__'):
    tranfrom_pipeline= transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Tensor()
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) 

    transform_dict={'train':tranfrom_pipeline,'valid':tranfrom_pipeline}
    splitter=PawpularityWithConfbarWithInvConfBarDatasetSplitter('Dataset/train','Dataset/train.csv',transform_dict)
    dataset,_=next(iter(splitter.generate_train_valid_dataset()))
    img,confbar,inv_confbar=dataset[1]
    print(img.shape)
    print(confbar.sum())
    print(inv_confbar.sum())
    # dataset=RandomGeneratorDataset('Dataset/train.csv')
    # for i in range(10):
    #     print(dataset[i])
    # print(img.shape,meta.shape,value)
    # print(bin_bar)
    # # img,label=dataset[0]
    # # print(img.shape,label.shape)
    # print()
    # for i in range(20):
    #     img,value=dataset[i]
    #     print(img.shape,value.shape)
    #     cv2.imshow('',img)
    #     cv2.waitKey(0)
    
    
    
    
    # num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df)))))