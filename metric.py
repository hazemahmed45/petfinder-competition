from abc import abstractmethod
import torch
import numpy as np
from sklearn.metrics import recall_score,precision_score,f1_score
import math
class Metric():
    def __init__(self) -> None:
        self.name='metric'
    @abstractmethod
    def update(self,**kwargs):
        return 
    @abstractmethod
    def get_value(self):
        return 
    @abstractmethod
    def reset(self):
        return 
class Accuracy(Metric):
    def __init__(self):
        self.name='Accuracy'
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        batch_size=targets.shape[0]
        self.value+=(prediciton.argmax(dim=1)==targets).float().mean().item()
        
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
        
class BinAccuracy(Metric):
    def __init__(self):
        self.name='BinAccuracy'
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        batch_size=targets.shape[0]
        self.value+=(prediciton.argmax(dim=1)==targets.argmax(dim=1)).float().mean().item()
        
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
class BinaryAccuracy(Metric):
    def __init__(self):
        self.name='BinaryAccuracy'
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        # batch_size=targets.shape[0]
        # print(prediciton)
        
        self.value+=((prediciton>0.5)==(targets>0.5)).float().mean().item()
        
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
class RMSE(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name='rmse'
        self.value=0
        self.num=0
        self.eps=1e-9
        
    def update(self, **kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        self.value+=math.sqrt(torch.nn.functional.mse_loss(prediciton,targets)+self.eps)
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
        return 
class ClassificationRMSE(Metric):
    def __init__(self,bin_increment) -> None:
        super().__init__()
        self.name='classification_rmse'
        self.value=0
        self.num=0
        self.eps=1e-9
        self.bin_increment=bin_increment
        
    def update(self, **kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        # bin_num=targets.shape[1]
        
        self.value+=math.sqrt(torch.nn.functional.mse_loss((prediciton.argmax(dim=1)*self.bin_increment)/100,(targets.argmax(dim=1)*self.bin_increment)/100)+self.eps)
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
        return 
class BinConfidenceBarRMSE(Metric):
    def __init__(self,bin_increment) -> None:
        super().__init__()
        self.name='classification_rmse'
        self.value=0
        self.num=0
        self.eps=1e-9
        self.bin_increment=bin_increment
        
    def update(self, **kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()

        
        self.value+=math.sqrt(torch.nn.functional.mse_loss((prediciton>0.5).sum()*self.bin_increment/100,((targets>0.5).sum()*self.bin_increment)/100)+self.eps)
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
        return 
class RunningLoss(Metric):
    def __init__(self,name='loss'):
        self.name=name
        self.value=0
        self.num=0
    def update(self,**kwargs):
        self.num+=1
        step_loss=kwargs['batch_loss']
        self.value+=step_loss.detach().cpu().item()
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
    def reset(self):
        self.value=0
        self.num=0
        return 
   
    
class Precision(Metric):
    def __init__(self):
        self.name='Precision'
        # self.tp=0
        # self.fp=0
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        y_true=y_true.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy().argmax(axis=1)
        
        # print(y_true)
        # print(y_pred)
        self.value+=precision_score(y_true,y_pred,average='macro',zero_division=0)
        # print(precision_score(y_true,y_pred,average='macro'))
        # exit()
        return 
    def reset(self):
        self.num=0
        self.value=0
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
class Recall(Metric):
    def __init__(self):
        self.name='Recall'
        self.value=0
        self.num=0
        return 
    def update(self, **kwargs):
        self.num+=1
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        y_true=y_true.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy().argmax(axis=1)
        self.value+=recall_score(y_true,y_pred,average='macro',zero_division=0)
        return 
    def reset(self):
        self.num=0
        self.value=0
        return 
    def get_value(self):
        return self.value/self.num if self.num !=0 else 0
class F1Score(Metric):
    def __init__(self):
        self.name='F1Score'
        self.recall=Recall()
        self.precision=Precision()
        return 
    def update(self, **kwargs):
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        self.recall.update(y_true=y_true,y_pred=y_pred)
        self.precision.update(y_true=y_true,y_pred=y_pred)
        return
    def reset(self):
        self.precision.reset()
        self.recall.reset()
        return
    def get_value(self):
        return 2*((self.precision.get_value()*self.recall.get_value())/(self.precision.get_value()+self.recall.get_value()))


if(__name__ == '__main__'):
    m=BinConfidenceBarRMSE(5)
    x=torch.cat((torch.ones((1,2)),torch.zeros((1,18))),dim=1)
    y=torch.cat((torch.ones((1,10)),torch.zeros((1,10))),dim=1)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    m.update(y_pred=x,y_true=y)
    print(m.get_value())
    pass