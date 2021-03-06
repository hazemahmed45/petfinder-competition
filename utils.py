from abc import abstractmethod
import torch
from tqdm import tqdm


def train_meta_bar_regression_loop(model
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
        
def train_meta_bins_regression_loop(model
                            ,dataloader
                            ,optimizer
                            ,regression_criterion
                            ,confidence_criterion
                            ,running_loss
                            ,rmse_metric
                            ,classification_rmse_metric
                            ,acc_metric
                            ,e
                            ,device='cuda'
                            ,is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,meta_batch,label_batch,bins_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        label_batch=label_batch.to(device)
        meta_batch=meta_batch.to(device)
        bins_label_batch=bins_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_reg,output_bins=model(img_batch,meta_batch)
        label_batch=label_batch.view(output_reg.shape)
        loss_reg=regression_criterion(output_reg,label_batch)
        loss_bins=confidence_criterion(output_bins,bins_label_batch)
        loss=loss_reg+loss_bins
        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        rmse_metric.update(y_pred=output_reg,y_true=label_batch)
        classification_rmse_metric.update(y_pred=output_bins,y_true=bins_label_batch)
        acc_metric.update(y_pred=output_bins,y_true=bins_label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            rmse_metric.name:rmse_metric.get_value()*100,
            acc_metric.name:acc_metric.get_value(),
            classification_rmse_metric.name:classification_rmse_metric.get_value()*100
            })
def train_bins_classification_loop(model
                            ,dataloader
                            ,optimizer
                            ,bins_criterion
                            ,running_loss
                            ,classification_rmse_metric
                            ,acc_metric
                            ,e
                            ,device='cuda'
                            ,is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,bins_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        bins_label_batch=bins_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_bins=model(img_batch)
        loss=bins_criterion(output_bins,bins_label_batch)
        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        classification_rmse_metric.update(y_pred=output_bins,y_true=bins_label_batch)
        acc_metric.update(y_pred=output_bins,y_true=bins_label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            acc_metric.name:acc_metric.get_value(),
            classification_rmse_metric.name:classification_rmse_metric.get_value()*100
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
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            rmse_metric.name:rmse_metric.get_value()*100,
            binary_acc_metric.name:binary_acc_metric.get_value()
            })
        
        
def train_classification_loop(model
                        ,dataloader
                        ,optimizer
                        ,criterion
                        ,running_loss
                        ,acc_metric
                        ,e
                        ,device='cuda'
                        ,is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        label_batch=label_batch.to(device)
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output=model(img_batch)
        label_batch=label_batch.view(output.shape)
        loss=criterion(output,label_batch)

        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        acc_metric.update(y_pred=output,y_true=label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            acc_metric.name:acc_metric.get_value()
            })
    return 

def train_confbar_species_regression_loop(model
                            ,dataloader
                            ,optimizer
                            ,regression_criterion
                            ,confidence_criterion
                            ,species_criterion,
                            running_loss,
                            rmse_metric,
                            conf_acc_metric
                            ,species_metric,
                            e,
                            device='cuda',is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,label_batch,conf_label_batch,species_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        label_batch=label_batch.to(device)
        conf_label_batch=conf_label_batch.to(device)
        species_label_batch=species_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_reg,output_conf,output_species=model(img_batch)
        species_label_batch=species_label_batch.view(output_species.shape)
        label_batch=label_batch.view(output_reg.shape)
        loss_reg=regression_criterion(output_reg,label_batch)
        loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss_species=species_criterion(output_species,species_label_batch)
        loss=0.8*loss_reg+0.1*loss_conf+0.1*loss_species
        if(is_train):
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        rmse_metric.update(y_pred=output_reg,y_true=label_batch)
        conf_acc_metric.update(y_pred=output_conf,y_true=conf_label_batch)
        species_metric.update(y_pred=output_species,y_true=species_label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            rmse_metric.name:rmse_metric.get_value()*100,
            conf_acc_metric.name:conf_acc_metric.get_value(),
            species_metric.name:species_metric.get_value()
            })

def train_confbar_invconfbar_regression_loop(model
                            ,dataloader
                            ,optimizer
                            ,confidence_criterion
                            ,inv_confidence_criterion
                            ,running_loss
                            ,conf_acc_metric
                            ,inv_conf_acc_metric
                            ,e
                            ,device='cuda',is_train=True):
    iter_loop=tqdm(enumerate(dataloader),total=len(dataloader))
    # running_loss=0
    for ii,(img_batch,conf_label_batch,inv_conf_label_batch) in iter_loop:
        img_batch=img_batch.to(device)
        conf_label_batch=conf_label_batch.to(device)
        inv_conf_label_batch=inv_conf_label_batch.to(device)
        
        # print(img_batch.shape,label_batch.shape,meta_batch.shape)
        output_conf,output_inv_conf=model(img_batch)
        loss_conf=confidence_criterion(output_conf,conf_label_batch)
        loss_inv_conf=inv_confidence_criterion(output_inv_conf,inv_conf_label_batch)
        loss=loss_conf+loss_inv_conf
        if(is_train):
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss.update(batch_loss=loss)
        conf_acc_metric.update(y_pred=output_conf,y_true=conf_label_batch)
        inv_conf_acc_metric.update(y_pred=output_inv_conf,y_true=inv_conf_label_batch)
        iter_loop.set_description('TRAIN' if is_train else 'VALID'+' LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            conf_acc_metric.name:conf_acc_metric.get_value(),
            inv_conf_acc_metric.name:inv_conf_acc_metric.get_value(),
            })
        