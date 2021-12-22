import torch
from tqdm import tqdm



def train_meta_confbar_regression_loop(model
                            ,dataloader
                            ,optimizer
                            ,regression_criterion
                            ,confidence_criterion,
                            running_loss,
                            rmse_metric,
                            binary_acc_metric,
                            e,
                            device='cuda',is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,meta_batch,label_batch,conf_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        label_batch=label_batch.to(device)
        meta_batch=meta_batch.to(device)
        conf_label_batch=conf_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_reg,output_conf=model(img_batch,meta_batch)
        label_batch=label_batch.view(output_reg.shape)
        loss_reg=regression_criterion(output_reg,label_batch)
        loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss=loss_reg+loss_conf
        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        rmse_metric.update(y_pred=output_reg,y_true=label_batch)
        binary_acc_metric.update(y_pred=output_conf,y_true=conf_label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            rmse_metric.name:rmse_metric.get_value()*100,
            binary_acc_metric.name:binary_acc_metric.get_value()
            })
        
def train_confbar_regression_loop(model
                            ,dataloader
                            ,optimizer
                            ,regression_criterion
                            ,confidence_criterion,
                            running_loss,
                            rmse_metric,
                            binary_acc_metric,
                            e,
                            device='cuda',is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,label_batch,conf_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        label_batch=label_batch.to(device)
        conf_label_batch=conf_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_reg,output_conf=model(img_batch)
        label_batch=label_batch.view(output_reg.shape)
        loss_reg=regression_criterion(output_reg,label_batch)
        loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss=loss_reg+loss_conf
        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
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