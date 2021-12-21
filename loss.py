import torch
from torch import nn
from torch.nn import MSELoss

class RMSELoss(nn.Module):
    def __init__(self,lambda_value=1):
        super(RMSELoss,self).__init__()
        self.lambda_value=lambda_value
        self.mse=MSELoss()
        self.eps=1e-9
    def forward(self,output,target):
        return self.lambda_value*torch.sqrt(self.mse(output,target)+self.eps)
    