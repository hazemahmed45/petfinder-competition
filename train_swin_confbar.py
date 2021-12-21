import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import SwinTransfromerWithConfidenceBarPawpularityRegressor
from dataloader import PawpularityDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss,BCEWithLogitsLoss
# from loss import RMSELoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from metric import RunningLoss,RMSE,BinaryAccuracy
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np
from config import Configs

# from torchsummary import summary
config=Configs()
config.with_meta=False
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
    'train':get_transform_pipeline(config.img_width,config.img_height,config.augmentation,prob_range=[config.low_aug_bounds,config.high_aug_bounds]),
    'valid':get_transform_pipeline(config.img_width,config.img_height,False)
}
dataset_splitter=PawpularityDatasetSplitter(img_dir=config.img_dir,meta_csv=config.img_meta_dir,transforms_dict=transform_dict,with_meta=config.with_meta,with_conf_bar=config.with_conf_bar)
train_dataset,valid_dataset=next(iter(dataset_splitter.generate_train_valid_dataset(config.train_split)))

train_loader=DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory)
# print(config.device)
model=SwinTransfromerWithConfidenceBarPawpularityRegressor()
model.to(config.device)

print(summary(model,input_data=(3,config.img_width,config.img_height)))

regression_criterion=MSELoss()
confidence_criterion=BCEWithLogitsLoss()
optimizer=Adam(params=model.parameters(),lr=config.lr)

running_loss=RunningLoss()
rmse_metric=RMSE()
binary_acc_metric=BinaryAccuracy()
ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,MODEL_NAME),'min',verbose=1)
schedular=StepLR(optimizer=optimizer,step_size=config.step_size,gamma=config.step_gamma)

wandb.watch(model,criterion=[regression_criterion,confidence_criterion],log_freq=1,log_graph=True)

for e in range(config.epochs):
    model.train()
    running_loss.reset()
    rmse_metric.reset()
    binary_acc_metric.reset()
    log_dict={}
    iter_loop=tqdm(enumerate(train_loader),total=len(train_loader))
    # running_loss=0
    for ii,(img_batch,label_batch,conf_label_batch) in iter_loop:
        optimizer.zero_grad(set_to_none=True)
        img_batch=img_batch.cuda()
        label_batch=label_batch.cuda()
        conf_label_batch=conf_label_batch.cuda()
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_reg,output_conf=model(img_batch)
        label_batch=label_batch.view(output_reg.shape)
        loss_reg=regression_criterion(output_reg,label_batch)
        loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss=loss_reg+loss_conf
        loss.backward()
        optimizer.step()
        running_loss.update(batch_loss=loss)
        rmse_metric.update(y_pred=output_reg,y_true=label_batch)
        binary_acc_metric.update(y_pred=output_conf,y_true=conf_label_batch)
        iter_loop.set_description('TRAIN LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            rmse_metric.name:rmse_metric.get_value()*100,
            binary_acc_metric.name:binary_acc_metric.get_value()
            })
    log_dict['loss/train']=running_loss.get_value()
    log_dict['rmse/train']=rmse_metric.get_value()*100
    log_dict['binary_acc/train']=binary_acc_metric.get_value()

    model.eval()
    running_loss.reset()
    rmse_metric.reset()
    binary_acc_metric.reset()
    with torch.no_grad():
        iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
        for ii,(img_batch,label_batch,conf_label_batch) in iter_loop:
            img_batch=img_batch.cuda()
            label_batch=label_batch.cuda()
            conf_label_batch=conf_label_batch.cuda()
            output_reg,output_conf=model(img_batch)
            label_batch=label_batch.view(output_reg.shape)
            loss_reg=regression_criterion(output_reg,label_batch)
            loss_conf=confidence_criterion(output_conf,conf_label_batch)
            loss=loss_reg+loss_conf
            running_loss.update(batch_loss=loss)
            rmse_metric.update(y_pred=output_reg,y_true=label_batch)
            binary_acc_metric.update(y_pred=output_conf,y_true=conf_label_batch)
            iter_loop.set_description('VALID LOOP E: '+str(e))
            iter_loop.set_postfix({
                running_loss.name:running_loss.get_value(),
                rmse_metric.name:rmse_metric.get_value()*100,
                binary_acc_metric.name:binary_acc_metric.get_value()
            })
        ckpt_callback.check_and_save(model,running_loss.get_value())   
    schedular.step()
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['rmse/valid']=rmse_metric.get_value()*100
    log_dict['binary_acc/valid']=binary_acc_metric.get_value()
    log_dict['epochs']=e
    log_dict['LR']=schedular.get_last_lr()
    wandb.log(log_dict)

wandb.finish()