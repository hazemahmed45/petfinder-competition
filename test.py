from pickle import FALSE
from model import InceptionV3WithMetaWithConfidenceBarPawpularityClassifier
import torch
from dataloader import PawpularityDatasetSplitter
from augmentation import get_torch_transform_pipeline
from sklearn.metrics import mean_squared_error
import math
import numpy as np


IMG_WIDTH=299
IMG_HEIGHT=299
BATCH_SIZE=1
TRAIN_SPLIT=0.7
AUGMENTATION=False
WITH_META=True
WITH_CONF_BAR=True

model = InceptionV3WithMetaWithConfidenceBarPawpularityClassifier().cuda()
model.load_state_dict(torch.load('checkpoints/pawpularity_inceptionv3_withmeta_withconfbar_4.pt'))
model.eval()
IMG_DIR='Dataset/train'
IMG_META_DIR='Dataset/train.csv'

transform_dict={
    'train':get_torch_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,AUGMENTATION),
    'valid':get_torch_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}
dataset_splitter=PawpularityDatasetSplitter(img_dir=IMG_DIR,meta_csv=IMG_META_DIR,transforms_dict=transform_dict,with_meta=WITH_META,with_conf_bar=WITH_CONF_BAR)
train_dataset,valid_dataset=next(iter(dataset_splitter.generate_train_valid_dataset(TRAIN_SPLIT)))

def mse(x,y):
    
    return (x-y)*(x-y)
# train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
# valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
y_pred_conf=[]
y_pred_reg=[]
y_true=[]
rmse_conf=[]
rmse_reg=[]
for i in range(10):
    img,meta,label,confbar=valid_dataset[i]
    # print(img.shape,meta.shape,label,confbar.shape)
    # print(img.max(),img.min())
    img=img.view((1,3,IMG_WIDTH,IMG_HEIGHT)).cuda()
    meta=meta.cuda()
    out_reg,out_conf=model(img,meta)
    out_conf=out_conf.detach().cpu()
    out_reg=int(out_reg.detach().cpu().item()*100)
    # print((out_conf>0.3).float())
    # if(out_reg != 100):
    #     if(out_conf[out_reg+1]==1):
    #         out_reg+=1
    y_pred_conf.append((out_conf>0.05).float().sum().item())
    y_pred_reg.append(out_reg)
    y_true.append(label.item()*100)
    # rmse_conf.append(math.sqrt(mse((out_conf>0.3).float().sum().item(),label.item()*100)+1e-9))
    # rmse_reg.append(math.sqrt(mse(out_reg,label.item()*100)+1e-9))
    # print("PRED: " ,out_reg," LABEL: ",label.item()," CONFBAR: ",(out_conf>0.3).float().sum())
    # exit()
print(math.sqrt(mean_squared_error(y_true,y_pred_conf)))
print(math.sqrt(mean_squared_error(y_true,y_pred_reg)))
print(np.sqrt(np.square(np.array(y_pred_conf)-np.array(y_true)).mean()))
print(np.sqrt(np.square(np.array(y_pred_reg)-np.array(y_true)).mean()))
print(np.sqrt(np.square(((np.array(y_pred_reg)+np.array(y_pred_conf))/2)-np.array(y_true)).mean()))
