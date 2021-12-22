import torch
from torch.functional import split
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,StratifiedGroupKFold
from augmentation import get_low_aug_transform_pipeline, get_transform_pipeline,get_torch_transform_pipeline
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import seaborn as sb
import matplotlib.pyplot as plt
from torchvision.transforms import Compose as t_compose
from albumentations import Compose as a_compose



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

class PawpularityClassificationDataset(Dataset):
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
        return img,torch.tensor(paw_value,dtype=torch.int64)
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return 
class PawpularityClassificationWithMetaDataset(Dataset):
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
        # img=img/255.0
        return img.float(),torch.tensor(meta).float(),(torch.tensor(paw_value,dtype=torch.int64)-1)
    
    def __len__(self):
        return self.dataframe.shape[0]
    def plot_y_hist(self):
        sb.histplot(x=self.dataframe['Pawpularity'])
        plt.show()
        return 

    
class PawpularityDatasetSplitter():
    def __init__(self,img_dir,meta_csv:str,transforms_dict:dict,with_meta=False,with_conf_bar=False):
        self.img_dir=img_dir
        self.data_df=pd.read_csv(meta_csv)
        self.transform_dict=transforms_dict
        
        self.with_meta=with_meta
        self.data_df.iloc[:,-1]=self.normalize(self.data_df.iloc[:,-1])
        # print(self.Y.describe())
        self.k_folder=KFold()
        self.with_conf_bar=with_conf_bar
        return 
    def generate_train_valid_dataset(self,train_split=0.8):
        # train_idx,valid_idx=train_test_split(np.arange(self.data_df.shape[0]),train_size=train_split,stratify=self.data_df.iloc[:,-1])
        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            if(self.with_meta):
                if(self.with_conf_bar):
                    yield PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
                else:
                    yield PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithMetaDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
            else:
                if(self.with_conf_bar):
                    yield PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityWithConfidenceBarDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
                else:
                    yield PawpularityDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

    def normalize(self,series:pd.Series) -> pd.Series:
        
        return (series-0)/(series.max()-0)
class PawpularityClassificationDatasetSplitter():
    def __init__(self,img_dir,meta_csv:str,transforms_dict:dict,with_meta=False):
        self.img_dir=img_dir
        self.data_df=pd.read_csv(meta_csv)
        self.transform_dict=transforms_dict
        
        self.with_meta=with_meta
        # self.data_df.iloc[:,-1]=self.normalize(self.data_df.iloc[:,-1])
        # print(self.Y.describe())
        self.k_folder=StratifiedKFold()
        return 
    def generate_train_valid_dataset(self,train_split=0.8):
        train_idx,valid_idx=train_test_split(np.arange(self.data_df.shape[0]),train_size=train_split,stratify=self.data_df.iloc[:,-1])
        # for train_idx,valid_idx in self.k_folder.split(self.data_df):
        if(self.with_meta):
            yield PawpularityClassificationWithMetaDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityClassificationWithMetaDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
        else:
            yield PawpularityClassificationDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),PawpularityClassificationDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))

    def normalize(self,series:pd.Series) -> pd.Series:
        
        return (series-0)/(series.max()-0)

class DogCatDataset(Dataset):
    def __init__(self,dataset_dir,transforms=None):
        super().__init__()
        self.data_dir=dataset_dir
        self.img_list=os.listdir(dataset_dir)
        self.transforms=transforms
    def __getitem__(self, index):
        img_name=self.img_list[index]
        img_path=os.path.join(self.data_dir,img_name)
        img=cv2.imread(img_path)
        if(self.transforms is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
        label=0 if 'dog' in str.lower(img_name) else 1
        
        return img,torch.tensor(label).float()
    def __len__(self):
        return len(self.img_list)

class OpenImageDogCatDataset(Dataset):
    def __init__(self,image_dir,label_dir,transform=None):
        super().__init__()
        self.image_dir=image_dir
        self.label_dir=label_dir
        self.img_list=[]
        self.label_list=[]
        self.dog_cat_labels=[8,7]
        self.transform=transform
        # print("HERE")
        samples_length=len(os.listdir(self.image_dir))
        for img_name in os.listdir(self.image_dir):
            img_path=os.path.join(self.image_dir,img_name)
            label_path=os.path.join(self.label_dir,img_name.replace('.jpg','.txt'))
            labels=self.read_label_file(label_path)
            if(len(labels)==0):
                continue
            for label in labels:
                self.img_list.append(img_path)
                self.label_list.append(label)
            # print(img_path,labels)
        return 
    
    def __getitem__(self, index):
        img_path=self.img_list[index]
        (c,center_x,center_y,w,h)=self.label_list[index]
        img=cv2.imread(img_path)
        height,width,_=img.shape
        xyxy=self.denormalize_coords(self.cxcywh_to_xyxy((center_x,center_y,w,h)),width,height)
        img=img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
            
        img=img/255.0

        return img,torch.tensor(c).float()
    def __len__(self):
        return len(self.img_list)
    def read_label_file(self,label_file_path):
        labels=[]
        with open(label_file_path,'r') as f:
            for line in f.readlines():
                c,center_x,center_y,w,h=line.replace("\n","").split()
                c=int(c)
                if(c in self.dog_cat_labels):
                    labels.append(((0 if c == 7 else 1),float(center_x),float(center_y),float(w),float(h)))
        return labels
    def denormalize_coords(self,xyxy,img_width,img_height):
        x1,y1,x2,y2=xyxy
        return int(x1*img_width),int(y1*img_height),int(x2*img_width),int(y2*img_height)
    def cxcywh_to_xyxy(self,cxcywh):
        cx,cy,w,h=cxcywh
        x1,y1,x2,y2=cx-(w/2),cy-(h/2),cx+(w/2),cy+(h/2)
        return x1,y1,x2,y2


class OpenImageDataset(Dataset):
    def __init__(self,image_dir,label_dir,img_list,label_list,transform=None):
        super().__init__()
        self.image_dir=image_dir
        self.label_dir=label_dir
        self.img_list=img_list
        self.label_list=label_list
        self.transform=transform
        return 
    
    def __getitem__(self, index):
        img_path=self.img_list[index]
        (c,center_x,center_y,w,h)=self.label_list[index]
        img=cv2.imread(img_path)
        height,width,_=img.shape
        xyxy=self.denormalize_coords(self.cxcywh_to_xyxy((center_x,center_y,w,h)),width,height)
        img=img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]
        if(self.transform is not None):
            if(isinstance(self.transform,t_compose)):
                img=Image.fromarray(img)
                img=self.transform(img)
            elif(isinstance(self.transform,a_compose)):
                img=self.transform(image=img)['image']
            # print(img.max(),img.min())
        # img=img/255.0
        return img.float(),torch.tensor(c).float()
    def __len__(self):
        return len(self.img_list)
    
    def denormalize_coords(self,xyxy,img_width,img_height):
        x1,y1,x2,y2=xyxy
        return int(x1*img_width),int(y1*img_height),int(x2*img_width),int(y2*img_height)
    def cxcywh_to_xyxy(self,cxcywh):
        cx,cy,w,h=cxcywh
        x1,y1,x2,y2=cx-(w/2),cy-(h/2),cx+(w/2),cy+(h/2)
        return x1,y1,x2,y2

class OpenImageDogCatDatasetSplitter():
    def __init__(self,image_dir,label_dir,transforms_dict:dict):
        self.image_dir=image_dir
        self.label_dir=label_dir
        self.transform_dict=transforms_dict
        
        self.img_list=[]
        self.label_list=[]
        self.dog_cat_labels=[8,7]
        self.transform_dict=transforms_dict
        # print("HERE")
        samples_length=len(os.listdir(self.image_dir))
        for img_name in os.listdir(self.image_dir):
            img_path=os.path.join(self.image_dir,img_name)
            label_path=os.path.join(self.label_dir,img_name.replace('.jpg','.txt'))
            labels=self.read_label_file(label_path)
            if(len(labels)==0):
                continue
            for label in labels:
                self.img_list.append(img_path)
                self.label_list.append(label)
        self.img_list=np.array(self.img_list)
        self.label_list=np.array(self.label_list)
        self.k_folder=StratifiedKFold()
        # print(self.label_list)
        return 
    def generate_train_valid_dataset(self,train_split=0.8):

        # for train_idx,valid_idx in self.k_folder.split(self.img_list):
        train_idx,valid_idx=train_test_split(np.arange(self.img_list.shape[0]),train_size=train_split,stratify=self.label_list[:,0])
        yield OpenImageDataset(self.image_dir,self.label_dir,self.img_list[train_idx],self.label_list[train_idx],self.transform_dict.get('train',None)),OpenImageDataset(self.image_dir,self.label_dir,self.img_list[valid_idx],self.label_list[valid_idx],self.transform_dict.get('valid',None))

    def read_label_file(self,label_file_path):
        labels=[]
        with open(label_file_path,'r') as f:
            for line in f.readlines():
                c,center_x,center_y,w,h=line.replace("\n","").split()
                c=int(c)
                if(c in self.dog_cat_labels):
                    labels.append(((0 if c == 7 else 1),float(center_x),float(center_y),float(w),float(h)))
        return labels


if (__name__ == '__main__'):
    # transform_dict={'train':get_torch_transform_pipeline(224,224,True),'valid':get_torch_transform_pipeline(224,224,False)}
    # splitter=PawpularityDatasetSplitter('Dataset/train','Dataset/train.csv',transform_dict,with_meta=True,with_conf_bar=True)
    # # splitter=OpenImageDogCatDatasetSplitter('openimage_dogcat/images/train','openimage_dogcat/labels/train',transform_dict)
    # dataset,_=next(iter(splitter.generate_train_valid_dataset()))
    # img,meta,value,conf_bar=dataset[0]
    # print(img.shape,meta.shape,value)
    # print(conf_bar)
    # # img,label=dataset[0]
    # # print(img.shape,label.shape)
    # print()
    # for i in range(20):
    #     img,value=dataset[i]
    #     print(img.shape,value.shape)
    #     cv2.imshow('',img)
    #     cv2.waitKey(0)
    num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df)))))