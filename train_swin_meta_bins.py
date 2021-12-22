import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import SwinTransfromerWithMetaWithBinsPawpularityRegressor
from dataloaders.regression_dataloader import PawpularityWithMetaWithBinsDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss,CrossEntropyLoss
# from loss import RMSELoss
from torch.utils.data import DataLoader
from augmentation import get_low_aug_transform_pipeline
from metric import RunningLoss,ClassificationRMSE,BinAccuracy,RMSE
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np
from config import Configs
from utils import train_meta_bins_regression_loop

# from torchsummary import summary
config=Configs()
config.lr=1e-4
config.batch_size=16
np.random.seed(config.random_seed)
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

# EXP_NUM=4
# CKPT_DIR='checkpoints'

BACKBONE='swin'
MODEL_NAME='pawpularity_'+BACKBONE+'_withmeta_withconfbar_'+str(config.experiment_num)+'.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
config.backbone=BACKBONE
config.model_name=MODEL_NAME
config.device=device

wandb.init(name='pawpularity-regressor-confbar-'+config.backbone+'-'+("" if config.with_meta ==False else 'meta')+'-'+str(config.experiment_num),config=config.get_config_dict(),job_type='regression',project="pawpularity-regression", entity="hazem45")

transform_dict={
    'train':get_low_aug_transform_pipeline(config.img_width,config.img_height,config.augmentation,prob_range=[config.low_aug_bounds,config.high_aug_bounds]),
    'valid':get_low_aug_transform_pipeline(config.img_width,config.img_height,False)
}
dataset_splitter=PawpularityWithMetaWithBinsDatasetSplitter(img_dir=config.img_dir,meta_csv=config.img_meta_dir,transforms_dict=transform_dict,bin_increment=config.bin_increment)
train_dataset,valid_dataset=next(iter(dataset_splitter.generate_train_valid_dataset(config.train_split)))

train_loader=DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory)
# print(config.device)
model=SwinTransfromerWithMetaWithBinsPawpularityRegressor(config.bin_increment)
model.to(config.device)

print(summary(model,input_data=[(3,config.img_width,config.img_height),(1,12)]))

regression_criterion=MSELoss()
confidence_criterion=CrossEntropyLoss()
optimizer=Adam(params=model.parameters(),lr=config.lr)

running_loss=RunningLoss()
rmse_metric=RMSE()
classification_rmse_metric=ClassificationRMSE(config.bin_increment)
acc_metric=BinAccuracy()
ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,MODEL_NAME),'min',verbose=1)
schedular=StepLR(optimizer=optimizer,step_size=config.step_size,gamma=config.step_gamma)

wandb.watch(model,criterion=[regression_criterion,confidence_criterion],log_freq=1,log_graph=True)

for e in range(config.epochs):
    model.train()
    running_loss.reset()
    rmse_metric.reset()
    acc_metric.reset()
    classification_rmse_metric.reset()
    log_dict={}

    train_meta_bins_regression_loop(model,train_loader,optimizer,regression_criterion,confidence_criterion,running_loss,rmse_metric,classification_rmse_metric,acc_metric,e,device,True)
    
    log_dict['loss/train']=running_loss.get_value()
    log_dict['rmse/train']=rmse_metric.get_value()*100
    log_dict['classification_rmse/train']=classification_rmse_metric.get_value()*100
    log_dict['acc/train']=acc_metric.get_value()

    model.eval()
    running_loss.reset()
    rmse_metric.reset()
    acc_metric.reset()
    classification_rmse_metric.reset()
    
    with torch.no_grad():
        train_meta_bins_regression_loop(model,valid_loader,optimizer,regression_criterion,confidence_criterion,running_loss,rmse_metric,classification_rmse_metric,acc_metric,e,device,False)

        ckpt_callback.check_and_save(model,running_loss.get_value())   
    schedular.step()
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['rmse/valid']=rmse_metric.get_value()*100
    log_dict['classification_rmse/valid']=classification_rmse_metric.get_value()*100
    log_dict['acc/valid']=acc_metric.get_value()
    log_dict['epochs']=e
    log_dict['LR']=schedular.get_last_lr()
    wandb.log(log_dict)

wandb.finish()
config.save_config_dict()