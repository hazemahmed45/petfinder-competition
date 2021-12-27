import cv2
import pandas as pd
import os 


df=pd.read_csv('train_dog_cat.csv')
img_dir='Dataset/train'

for ii,(id,(img_name,class_id)) in enumerate(df[df['class']==0].iterrows()):
    img_path=os.path.join(img_dir,img_name)
    print(str(ii)+'/'+str(df[df['class']==0].shape[0]),img_name,class_id)
    cv2.imshow('',cv2.imread(img_path))
    cv2.waitKey(0)
    # exit()

for ii,(id,(img_name,class_id)) in enumerate(df[df['class']==1].iterrows()):
    img_path=os.path.join(img_dir,img_name)
    print(str(ii)+'/'+str(df[df['class']==1].shape[0]),img_name,class_id)
    cv2.imshow('',cv2.imread(img_path))
    cv2.waitKey(0)
    # exit()