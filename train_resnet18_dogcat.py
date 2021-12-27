import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import Resnet18Backbone
from dataloaders.classification_dataloaders import DogCatDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss,BCEWithLogitsLoss,BCELoss,CrossEntropyLoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline,get_low_aug_transform_pipeline
from metric import RunningLoss,BinaryAccuracy,Accuracy,Precision,Recall,F1Score
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np
from utils import train_bins_classification_loop, train_classification_loop
from config import Configs
import atexit


# from torchsummary import summary
# from thop import profile,clever_formats1

config=Configs()
def exit_interupt_handler():
    global config
    config.is_stoped=True
    config.save_config_dict()
    wandb.finish()
    return 
def exit_handler():
    global config
    config.is_stoped=False
    config.save_config_dict()
    wandb.finish()
    return 
atexit.register(exit_interupt_handler)

np.random.seed(config.random_seed)
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed(config.random_seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

config.script_name=__file__
config.dataset_name=DogCatDatasetSplitter.__name__
config.model_name=Resnet18Backbone.__name__
# EXP_NUM=4
# CKPT_DIR='checkpoints'
config.epochs=50
BACKBONE='resnet18'
config.backbone=BACKBONE
MODEL_NAME='pawpularity_'+config.backbone+'_classifier_'+str(config.experiment_num)+'.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
config.model_name=MODEL_NAME
config.device=device
config.job_type='classification'
config.img_dir='./dogs-vs-cats/train'

wandb.init(name='pawpularity-regressor-'+config.backbone+'-backbone-'+str(config.experiment_num),config=config,job_type=config.job_type,project="pawpularity-regression", entity="hazem45")
transform_dict={
    'train':get_transform_pipeline(config.img_width,config.img_height,config.augmentation),
    'valid':get_transform_pipeline(config.img_width,config.img_height,False)
}
splitter=DogCatDatasetSplitter(dataset_dir=config.img_dir,transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(splitter.generate_train_valid_dataset(config.train_split)))

train_loader=DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory,drop_last=True)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,pin_memory=config.pin_memory,drop_last=False)
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
model.to(device)

print(summary(model,input_data=(3,config.img_width,config.img_height)))

criterion=BCEWithLogitsLoss()
optimizer=Adam(params=model.parameters(),lr=config.lr)
# schedular=StepLR(optimizer=optimizer,step_size=5)
acc_metric=BinaryAccuracy()

running_loss=RunningLoss()
ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,MODEL_NAME),'min',verbose=1)
backbone_ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,"backbone_"+MODEL_NAME),'min',verbose=1)

wandb.watch(model,criterion=criterion,log_freq=1,log_graph=True)

for e in range(config.epochs):
    model.train()
    acc_metric.reset()
    running_loss.reset()

    log_dict={}
    train_classification_loop(model,train_loader,optimizer,criterion,running_loss,acc_metric,e,config.device,True)

    log_dict['loss/train']=running_loss.get_value()
    log_dict['accuracy/train']=acc_metric.get_value()
 
    model.eval()
    acc_metric.reset()
    running_loss.reset()


    with torch.no_grad():
        train_classification_loop(model,valid_loader,optimizer,criterion,running_loss,acc_metric,e,config.device,False)
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['accuracy/valid']=acc_metric.get_value()
    ckpt_callback.check_and_save(model,running_loss.get_value())   
    backbone_ckpt_callback.check_and_save(backone,running_loss.get_value())   
    # schedular.step()
                 
    # log_dict['loss/valid']=running_loss.get_value()
    log_dict['epochs']=e
    # log_dict['lr']=schedular.get_last_lr()[-1]
    wandb.log(log_dict)

wandb.finish()
atexit.unregister(exit_interupt_handler)
atexit.register(exit_handler)