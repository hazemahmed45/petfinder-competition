import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import SwinTransfromerWithMetaPawpularityClassifier
from dataloaders.regression_dataloader import PawpularityClassificationDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
# from loss import RMSELoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline,get_low_aug_transform_pipeline,get_torch_transform_pipeline,get_swin_transform_pipeline
from metric import RunningLoss,Accuracy,RMSE,BinaryAccuracy
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np

# from torchsummary import summary

SEED=999
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

EXP_NUM=0
CKPT_DIR='checkpoints'
IMG_DIR='Dataset/train'
IMG_META_DIR='Dataset/train.csv'

IMG_WIDTH=224
IMG_HEIGHT=224
BATCH_SIZE=16
SHUFFLE=True
NUM_WORKERS=8
PIN_MEMORY=True
EPOCHS=30
LR=1e-3
TRAIN_SPLIT=0.7 
LOW_AUG_BOUND=0.5
HIGH_AUG_BOUND=0.6
LOSS_MAGNIFIER=3
AUGMENTATION=True
WITH_META=True
WITH_CONF_BAR=True
SCHEDULAR='step'
BACKBONE='swin'
STEP_SIZE=5
STEP_GAMMA=0.1
MODEL_NAME='pawpularity_'+BACKBONE+'_withmeta_classification_'+str(EXP_NUM)+'.pt'
# BACKBONE_WEIGHTS='checkpoints/pawpularity_inceptionv3_backbone_2.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
config={
    'experiment_number':EXP_NUM,
    'checkpoint_name':MODEL_NAME,
    'random_seed':SEED,
    'low_augmentation_bounds':LOW_AUG_BOUND,
    'high_augmentation_bounds':HIGH_AUG_BOUND,
    'image_dir':IMG_DIR,
    'img_meta_file':IMG_META_DIR,
    'image_width':IMG_WIDTH,
    'image_height':IMG_HEIGHT,
    'batch_size':BATCH_SIZE,
    'shuffle':SHUFFLE,
    'number_workers':NUM_WORKERS,
    'pin_memory':PIN_MEMORY,
    'device':device,
    'epochs':EPOCHS,
    'learning_rate':LR,
    'train_split':TRAIN_SPLIT,
    'augmentation':AUGMENTATION,
    'backbone':BACKBONE,
    # 'backbone_weights':BACKBONE_WEIGHTS,
    'loss_magnifier':LOSS_MAGNIFIER,
    'with_meta':WITH_META,
    'schedular':SCHEDULAR,
    'step_size':STEP_SIZE,
    'step_gamma':STEP_GAMMA
}
wandb.init(name='pawpularity-classification-'+BACKBONE+'-'+("" if WITH_META ==False else 'meta')+'-'+str(EXP_NUM),config=config,job_type='classification',project="pawpularity-regression", entity="hazem45")
# transform_dict={
#     'train':get_low_aug_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,AUGMENTATION),
#     'valid':get_low_aug_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
# }
transform_dict={
    'train':get_swin_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,AUGMENTATION),
    'valid':get_swin_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}
dataset_splitter=PawpularityClassificationDatasetSplitter(img_dir=IMG_DIR,meta_csv=IMG_META_DIR,transforms_dict=transform_dict,with_meta=WITH_META)
train_dataset,valid_dataset=next(iter(dataset_splitter.generate_train_valid_dataset(TRAIN_SPLIT)))

train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

model=SwinTransfromerWithMetaPawpularityClassifier()
   
# model.load_state_dict(torch.load('checkpoints/pawpularity_inceptionv3_withmeta_withconfbar_1.pt'))
model.to(device)

print(summary(model,input_data=[(3,IMG_WIDTH,IMG_HEIGHT),(1,12)]))

criterion=CrossEntropyLoss()
# confidence_criterion=BCEWithLogitsLoss()
optimizer=Adam(params=model.parameters(),lr=LR)

running_loss=RunningLoss()
# rmse_metric=RMSE()
binary_acc_metric=Accuracy()
ckpt_callback=CheckpointCallback(os.path.join(CKPT_DIR,MODEL_NAME),'min',verbose=1)
schedular=StepLR(optimizer=optimizer,step_size=STEP_SIZE,gamma=STEP_GAMMA)

wandb.watch(model,criterion=criterion,log_freq=1,log_graph=True)

for e in range(EPOCHS):
    model.train()
    running_loss.reset()
    # rmse_metric.reset()
    binary_acc_metric.reset()
    log_dict={}
    iter_loop=tqdm(enumerate(train_loader),total=len(train_loader))
    # running_loss=0
    for ii,(img_batch,meta_batch,label_batch) in iter_loop:
        optimizer.zero_grad(set_to_none=True)
        img_batch=img_batch.cuda()
        label_batch=label_batch.cuda()
        meta_batch=meta_batch.cuda()        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output=model(img_batch,meta_batch)
        # print(label_batch.shape,output.shape)
        # print(label_batch)
        
        # label_batch=label_batch.view(output.shape)
        loss=criterion(output,label_batch)
        # loss_reg=regression_criterion(output_reg,label_batch)
        # loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss.backward()
        optimizer.step()
        running_loss.update(batch_loss=loss)
        # rmse_metric.update(y_pred=output_reg,y_true=label_batch)
        binary_acc_metric.update(y_pred=output,y_true=label_batch)
        iter_loop.set_description('TRAIN LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            # rmse_metric.name:rmse_metric.get_value()*100,
            binary_acc_metric.name:binary_acc_metric.get_value()
            })
    log_dict['loss/train']=running_loss.get_value()
    # log_dict['rmse/train']=rmse_metric.get_value()*100
    log_dict['binary_acc/train']=binary_acc_metric.get_value()

    model.eval()
    running_loss.reset()
    # rmse_metric.reset()
    binary_acc_metric.reset()
    with torch.no_grad():
        iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
        for ii,(img_batch,meta_batch,label_batch) in iter_loop:
            img_batch=img_batch.cuda()
            label_batch=label_batch.cuda()
            # conf_label_batch=conf_label_batch.cuda()
            meta_batch=meta_batch.cuda()
            output=model(img_batch,meta_batch)
            # label_batch=label_batch.view(output.shape)
            loss=criterion(output,label_batch)
            running_loss.update(batch_loss=loss)
            # rmse_metric.update(y_pred=output_reg,y_true=label_batch)
            binary_acc_metric.update(y_pred=output,y_true=label_batch)
            iter_loop.set_description('VALID LOOP E: '+str(e))
            iter_loop.set_postfix({
                running_loss.name:running_loss.get_value(),
                # rmse_metric.name:rmse_metric.get_value()*100,
                binary_acc_metric.name:binary_acc_metric.get_value()
            })
        ckpt_callback.check_and_save(model,running_loss.get_value())   
    schedular.step()
    log_dict['loss/valid']=running_loss.get_value()
    # log_dict['rmse/valid']=rmse_metric.get_value()*100
    log_dict['binary_acc/valid']=binary_acc_metric.get_value()
    log_dict['epochs']=e
    log_dict['LR']=schedular.get_last_lr()
    wandb.log(log_dict)

wandb.finish()