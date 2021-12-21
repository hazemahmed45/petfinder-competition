import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import InceptionV3ClassifierBackbone
from dataloader import OpenImageDogCatDatasetSplitter
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline,get_low_aug_transform_pipeline,get_torch_transform_pipeline
from metric import RunningLoss,BinaryAccuracy#,Accuracy,Precision,Recall,F1Score
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
import random
import numpy as np
# from torchsummary import summary
# from thop import profile,clever_formats1

SEED=128
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

EXP_NUM=2
CKPT_DIR='checkpoints'
BACKBONE='inceptionv3'
MODEL_NAME='pawpularity_'+BACKBONE+'_backbone_'+str(EXP_NUM)+'.pt'
IMG_DIR='openimage_dogcat/images/train'
LABEL_DIR='openimage_dogcat/labels/train'
# IMG_META_DIR='Dataset/train.csv'

IMG_WIDTH=299
IMG_HEIGHT=299
BATCH_SIZE=64
SHUFFLE=True
NUM_WORKERS=8
PIN_MEMORY=True
EPOCHS=100
LR=1e-3
TRAIN_SPLIT=0.8
LOW_AUG_BOUND=0.7
HIGH_AUG_BOUND=0.8
AUGMENTATION=True
device='cuda' if torch.cuda.is_available() else 'cpu'
config={
    'experiment_number':EXP_NUM,
    'checkpoint_name':MODEL_NAME,
    'random_seed':SEED,
    'low_augmentation_bounds':LOW_AUG_BOUND,
    'high_augmentation_bounds':HIGH_AUG_BOUND,
    'image_dir':IMG_DIR,
    'label_dir':LABEL_DIR,
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
    'backbone':BACKBONE
}
wandb.init(name='pawpularity-regressor-'+BACKBONE+'-backbone-'+str(EXP_NUM),config=config,job_type='regression',project="pawpularity-regression", entity="hazem45")
transform_dict={
    'train':get_torch_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,AUGMENTATION),
    'valid':get_torch_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}
splitter=OpenImageDogCatDatasetSplitter(image_dir=IMG_DIR,label_dir=LABEL_DIR,transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(splitter.generate_train_valid_dataset(TRAIN_SPLIT)))

train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,drop_last=True)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,drop_last=False)
backone=InceptionV3ClassifierBackbone()
model=torch.nn.Sequential(
    backone,
    # torch.nn.Flatten(),
    torch.nn.Linear(backone.emb_dim,512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,1)
    )                     
# model=FashionMnistClassifier(train_dataset.n_classes)
model.to(device)

print(summary(model,input_data=(3,IMG_WIDTH,IMG_HEIGHT)))

criterion=BCEWithLogitsLoss()
optimizer=Adam(params=model.parameters(),lr=LR)
# schedular=StepLR(optimizer=optimizer,step_size=5)
acc_metric=BinaryAccuracy()
# prec_metric=Precision()
# recall_metric=Recall()
# f1_metric=F1Score()
running_loss=RunningLoss()
ckpt_callback=CheckpointCallback(os.path.join(CKPT_DIR,MODEL_NAME),'min',verbose=1)

wandb.watch(model,criterion=criterion,log_freq=1,log_graph=True)

for e in range(EPOCHS):
    model.train()
    acc_metric.reset()
    running_loss.reset()
    # prec_metric.reset()
    # recall_metric.reset()
    # f1_metric.reset()
    log_dict={}
    iter_loop=tqdm(enumerate(train_loader),total=len(train_loader))
    # running_loss=0
    for ii,(img_batch,label_batch) in iter_loop:
        optimizer.zero_grad(set_to_none=True)
        img_batch=img_batch.cuda()
        label_batch=label_batch.cuda()
        output=model(img_batch)
        label_batch=label_batch.view(output.shape)
        # print(output.shape,label_batch.shape)
        loss=criterion(output,label_batch)
        loss.backward()
        optimizer.step()
        
        running_loss.update(batch_loss=loss)
        acc_metric.update(y_true=label_batch,y_pred=output)
        # prec_metric.update(y_true=label_batch,y_pred=output)
        # recall_metric.update(y_true=label_batch,y_pred=output)
        # f1_metric.update(y_true=label_batch,y_pred=output)
        iter_loop.set_description('TRAIN LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            acc_metric.name:acc_metric.get_value(),
            # prec_metric.name:prec_metric.get_value(),
            # recall_metric.name:recall_metric.get_value(),
            # f1_metric.name:f1_metric.get_value()
            })
    log_dict['loss/train']=running_loss.get_value()
    log_dict['accuracy/train']=acc_metric.get_value()
    # log_dict['precision/train']=prec_metric.get_value()
    # log_dict['recall/train']=recall_metric.get_value()
    # log_dict['f1score/train']=f1_metric.get_value()
    model.eval()
    acc_metric.reset()
    running_loss.reset()
    # prec_metric.reset()
    # recall_metric.reset()
    # f1_metric.reset()

    iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
    # running_loss=0
    for ii,(img_batch,label_batch) in iter_loop:
        with torch.no_grad():
            img_batch=img_batch.cuda()
            label_batch=label_batch.cuda()
            output=model(img_batch)
            label_batch=label_batch.view(output.shape)
        # print(output.shape,label_batch.shape)
        loss=criterion(output,label_batch)
        
        running_loss.update(batch_loss=loss)
        acc_metric.update(y_true=label_batch,y_pred=output)
        # prec_metric.update(y_true=label_batch,y_pred=output)
        # recall_metric.update(y_true=label_batch,y_pred=output)
        # f1_metric.update(y_true=label_batch,y_pred=output)
        iter_loop.set_description('VALID LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            acc_metric.name:acc_metric.get_value(),
            # prec_metric.name:prec_metric.get_value(),
            # recall_metric.name:recall_metric.get_value(),
            # f1_metric.name:f1_metric.get_value()
            })
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['accuracy/valid']=acc_metric.get_value()
    ckpt_callback.check_and_save(backone,running_loss.get_value())   
    # schedular.step()
                 
    # log_dict['loss/valid']=running_loss.get_value()
    log_dict['epochs']=e
    # log_dict['lr']=schedular.get_last_lr()[-1]
    wandb.log(log_dict)

wandb.finish()