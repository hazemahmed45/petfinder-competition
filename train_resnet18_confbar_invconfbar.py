import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import Resnet18WithConfBarWithInvConfBarPawpularityRegressor
from dataloaders.regression_dataloader import PawpularityWithConfbarWithInvConfBarDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam,SGD
from torch.nn import MSELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline,get_low_aug_transform_pipeline
from metric import RMSE, RunningLoss,BinaryAccuracy,Accuracy,Precision,Recall,F1Score
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np
from utils import train_confbar_invconfbar_regression_loop
from config import Configs
import atexit


# from torchsummary import summary
# from thop import profile,clever_formats1

config=Configs()
def exit_handler():
    global config
    config.is_stoped=False
    config.save_config_dict()
    wandb.finish()
    return 
def exit_interupt_handler():
    global config
    config.is_stoped=True
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
config.dataset_name=PawpularityWithConfbarWithInvConfBarDatasetSplitter.__name__
config.model_name=Resnet18WithConfBarWithInvConfBarPawpularityRegressor.__name__
# EXP_NUM=4
# CKPT_DIR='checkpoints'

BACKBONE='resnet18'
config.backbone=BACKBONE
MODEL_NAME='pawpularity_'+config.backbone+"_"+str(config.experiment_num)+'.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
config.model_name=MODEL_NAME
config.device=device
config.species_csv='Dataset/train_dog_cat.csv'
config.epochs=100
config.batch_size=256
config.finetune=True
config.finetune_weights='exp/111/pawpularity_resnet18_111.pt'
config.lr=1e-2
config.backbone_weights='exp/75/backbone_pawpularity_resnet18_classifier_75.pt'
# config.img_width=512
# config.img_height=512

wandb.init(name='pawpularity-regressor-'+config.backbone+'-backbone-'+str(config.experiment_num),config=config,job_type=config.job_type,project="pawpularity-regression", entity="hazem45")
transform_dict={
    'train':get_transform_pipeline(config.img_width,config.img_height,config.augmentation),
    'valid':get_transform_pipeline(config.img_width,config.img_height,False)
}
splitter=PawpularityWithConfbarWithInvConfBarDatasetSplitter(img_dir=config.img_dir,meta_csv=config.img_meta_dir,transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(splitter.generate_train_valid_dataset(config.train_split)))

train_loader=DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=config.shuffle,num_workers=config.num_workers,pin_memory=config.pin_memory,drop_last=True)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,pin_memory=config.pin_memory,drop_last=False)
model=Resnet18WithConfBarWithInvConfBarPawpularityRegressor()
model.to(device)
if(config.finetune):
    model.load_state_dict(torch.load(config.finetune_weights))

print(summary(model,input_data=(3,config.img_width,config.img_height)))

conf_criterion=BCEWithLogitsLoss()
inv_conf_criterion=BCEWithLogitsLoss()

optimizer=SGD(params=model.parameters(),lr=config.lr)
# schedular=StepLR(optimizer=optimizer,step_size=5)
conf_acc_metric=BinaryAccuracy('confidence_acc')
inv_conf_acc_metric=BinaryAccuracy('inverse_confidence_acc')
# rmse_metric=RMSE()
running_loss=RunningLoss()
ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,MODEL_NAME),'min',verbose=1)
backbone_ckpt_callback=CheckpointCallback(os.path.join(config.cur_exp_dir,"backbone_"+MODEL_NAME),'min',verbose=1)

wandb.watch(model,criterion=[conf_criterion,inv_conf_acc_metric],log_freq=1,log_graph=True)

for e in range(config.epochs):
    model.train()
    conf_acc_metric.reset()
    inv_conf_acc_metric.reset()
    running_loss.reset()

    log_dict={}
    train_confbar_invconfbar_regression_loop(model=model,
                                          dataloader=train_loader,
                                          optimizer=optimizer,
                                          confidence_criterion=conf_criterion,
                                          inv_confidence_criterion=inv_conf_criterion,
                                          running_loss=running_loss,
                                          conf_acc_metric=conf_acc_metric,
                                          inv_conf_acc_metric=inv_conf_acc_metric,
                                          e=e,
                                          device=config.device,
                                          is_train=True
                                          )

    log_dict['loss/train']=running_loss.get_value()
    log_dict['confidence_accuracy/train']=conf_acc_metric.get_value()
    log_dict['inverse_confidence_accuracy/train']=inv_conf_acc_metric.get_value()
 
    model.eval()
    conf_acc_metric.reset()
    inv_conf_acc_metric.reset()
    running_loss.reset()


    with torch.no_grad():
        train_confbar_invconfbar_regression_loop(model=model,
                                          dataloader=valid_loader,
                                          optimizer=optimizer,
                                          confidence_criterion=conf_criterion,
                                          inv_confidence_criterion=inv_conf_criterion,
                                          running_loss=running_loss,
                                          conf_acc_metric=conf_acc_metric,
                                          inv_conf_acc_metric=inv_conf_acc_metric,
                                          e=e,
                                          device=config.device,
                                          is_train=False
                                          )

    log_dict['loss/valid']=running_loss.get_value()
    log_dict['confidence_accuracy/valid']=conf_acc_metric.get_value()
    log_dict['inverse_confidence_accuracy/valid']=inv_conf_acc_metric.get_value()
    ckpt_callback.check_and_save(model,running_loss.get_value())   
    # schedular.step()
                 
    # log_dict['loss/valid']=running_loss.get_value()
    log_dict['epochs']=e
    # log_dict['lr']=schedular.get_last_lr()[-1]
    wandb.log(log_dict)

wandb.finish()
atexit.unregister(exit_handler)
atexit.register(exit_handler)