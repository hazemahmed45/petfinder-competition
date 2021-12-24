from pickle import FALSE
from model import SmallSwinTransfromerWithBinsPawpularityClassifier
import torch
from dataloaders.classification_dataloaders import PawpularityClassificationWithBinsDatasetSplitter
from augmentation import get_torch_transform_pipeline,get_transform_pipeline
from sklearn.metrics import mean_squared_error
import math
import numpy as np


IMG_WIDTH=224
IMG_HEIGHT=224
BATCH_SIZE=1
TRAIN_SPLIT=0.7
AUGMENTATION=False
WITH_META=True
WITH_CONF_BAR=True

model = SmallSwinTransfromerWithBinsPawpularityClassifier(5).cuda()
model.load_state_dict(torch.load('exp/45/pawpularity_swin_withmeta_withconfbar_45.pt'))
model.eval()
IMG_DIR='Dataset/train'
IMG_META_DIR='Dataset/train.csv'

transform_dict={
    'train':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,AUGMENTATION),
    'valid':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}
dataset_splitter=PawpularityClassificationWithBinsDatasetSplitter(img_dir=IMG_DIR,meta_csv=IMG_META_DIR,transforms_dict=transform_dict,bin_increment=5)
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
    img,bin_confbar=valid_dataset[i]
    # print(img.shape,meta.shape,label,confbar.shape)
    # print(img.max(),img.min())
    img=img.view((1,3,IMG_WIDTH,IMG_HEIGHT)).cuda()
    out_bin=model(img)
    out_bin=out_bin.detach().cpu()
    print(bin_confbar)
    print(out_bin)
    # exit()
    # out_reg=int(out_reg.detach().cpu().item()*100)
    # print((out_conf>0.3).float())
    # if(out_reg != 100):
    #     if(out_conf[out_reg+1]==1):
    #         out_reg+=1
    # y_pred_conf.append((out_conf>0.05).float().sum().item())
    # y_pred_reg.append(out_reg)
    # y_true.append(label.item()*100)
    # rmse_conf.append(math.sqrt(mse((out_conf>0.3).float().sum().item(),label.item()*100)+1e-9))
    # rmse_reg.append(math.sqrt(mse(out_reg,label.item()*100)+1e-9))
    # print("PRED: " ,out_reg," LABEL: ",label.item()," CONFBAR: ",(out_conf>0.3).float().sum())
    # exit()
# print(math.sqrt(mean_squared_error(y_true,y_pred_conf)))
# print(math.sqrt(mean_squared_error(y_true,y_pred_reg)))
# print(np.sqrt(np.square(np.array(y_pred_conf)-np.array(y_true)).mean()))
# print(np.sqrt(np.square(np.array(y_pred_reg)-np.array(y_true)).mean()))
# print(np.sqrt(np.square(((np.array(y_pred_reg)+np.array(y_pred_conf))/2)-np.array(y_true)).mean()))
