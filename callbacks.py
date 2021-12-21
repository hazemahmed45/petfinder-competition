import torch
import os
from metric import Metric

class CheckpointCallback():
    def __init__(self,model_filename,mode='max',verbose=0) :
        self.model_filename=model_filename
        self.mode=mode
        self.verbose=verbose
        if(mode=='max'):
            self.value=-1e9
        else:
            self.value=1e9
    def check_and_save(self,model:torch.nn.Module,value):
        save=False
        if(self.mode =='max'):
            if(value>self.value):
                if(self.verbose==1):
                    print("MODEL SAVED, NEW VALUE IS ",value," AND PREVIOUS VALUE IS ",self.value)
                self.value=value
                save=True
        if(self.mode == 'min'):
            if(value<self.value):
                if(self.verbose==1):
                    print("MODEL SAVED, NEW VALUE IS ",value," AND PREVIOUS VALUE IS ",self.value)
                self.value=value
                save=True
        if(save):
            torch.save(model.state_dict(),self.model_filename)
        return 
    

# class CheckpointByMetricCallback():
#     def __init__(self,model_filename,metric:Metric,verbose=0) :
#         self.model_filename=model_filename
#         # self.mode=mode
#         self.metric=metric
#         self.verbose=verbose
#         if(mode=='max'):
#             self.value=-1e9
#         else:
#             self.value=1e9
#     def check_and_save(self,model:torch.nn.Module,value):
#         save=False
#         if(self.mode =='max'):
#             if(value>self.value):
#                 if(self.verbose==1):
#                     print("MODEL SAVED, NEW VALUE IS ",value," AND PREVIOUS VALUE IS ",self.value)
#                 self.value=value
#                 save=True
#         if(self.mode == 'min'):
#             if(value<self.value):
#                 if(self.verbose==1):
#                     print("MODEL SAVED, NEW VALUE IS ",value," AND PREVIOUS VALUE IS ",self.value)
#                 self.value=value
#                 save=True
#         if(save):
#             torch.save(model.state_dict(),self.model_filename)
#         return 