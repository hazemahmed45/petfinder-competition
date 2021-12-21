from cv2 import transform
import pandas as pd
import os
from model import PawpularityClassifier,InceptionV3WithMetaPawpularityClassifier
from augmentation import get_transform_pipeline
import torch
import cv2

def denormalize(x,max,min):
    return (x*(max-min))+min

model=InceptionV3WithMetaPawpularityClassifier()
model.load_state_dict(torch.load('checkpoints/pawpularity_inceptionv3_5.pt'))
model.eval()
test_img_dir='Dataset/test'
test_csv='Dataset/test.csv'
test_df=pd.read_csv(test_csv)


transform=get_transform_pipeline(224,224,False)
test_dict={"Id":[],"Pawpularity":[]}
for id,row in test_df.iterrows():
    # print(row)
    img_path=os.path.join(test_img_dir,row['Id']+'.jpg')
    img=cv2.imread(img_path)
    img=transform(image=img)['image']
    img=img.view((-1,3,224,224))
    with torch.no_grad():
        pred=model(img)
    pred=pred.detach().cpu().item()
    pred=denormalize(pred,100,0)
    print(pred)    
    test_dict['Id'].append(row['Id'])
    test_dict['Pawpularity'].append(pred)
    
pd.DataFrame(test_dict).to_csv('submission.csv',index=False)