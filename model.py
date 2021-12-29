from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.linear import Linear
from torchvision.models import resnet18,inception_v3
from torchvision.models.mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
from torchsummary import summary
import torch
from torch import nn
from timm import create_model
# from timm.models import SwinTransformer



class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity,self).__init__()
    def forward(self,x):
        return x
    
class MetaBlock(nn.Module):
    def __init__(self) -> None:
        super(MetaBlock,self).__init__()
        self.meta_emb_dim=256
        self.meta_block=nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,self.meta_emb_dim),
            nn.ReLU(),

        )
    def forward(self,x):
        return self.meta_block(x).view((-1,self.meta_emb_dim))
    
class SmallMetaBlock(nn.Module):
    def __init__(self) -> None:
        super(SmallMetaBlock,self).__init__()
        self.meta_emb_dim=64
        self.meta_block=nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,self.meta_emb_dim),
            nn.ReLU(),

        )
    def forward(self,x):
        return self.meta_block(x).view((-1,self.meta_emb_dim))
    
    
class Resnet18Backbone(nn.Module):
    def __init__(self,pretrained=True) -> None:
        super(Resnet18Backbone,self).__init__()
        self.backbone=create_model('resnet18',pretrained=pretrained,num_classes=0)
        self.emb_dim=512
        # self.backbone.fc=Identity()
        return 
    def forward(self,x):
        x=self.backbone(x)
        # print(x.shape)
        return x
class InceptionV3RegressorBackbone(nn.Module):
    def __init__(self,pretrained=False) -> None:
        super(InceptionV3RegressorBackbone,self).__init__()
        self.backbone=inception_v3(pretrained=pretrained)
        self.backbone.aux_logits=False
        self.emb_dim=self.backbone.fc.in_features
    
        self.backbone.fc=Identity()
        return 
    def forward(self,x):
        x=self.backbone(x)

        x=x.view((-1,self.emb_dim))

        return x


class Resnet18PawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(Resnet18PawpularityRegressor,self).__init__()
        self.in_layer=nn.Conv2d(1,3,3,padding=1)
        self.backbone=Resnet18Backbone(False)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        
        self.fc=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
        )

    def forward(self,x):
        x = self.backbone(x)
        out=self.fc(x)
        return out


class Resnet18WithMetaPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(Resnet18WithMetaPawpularityRegressor,self).__init__()
        self.backbone=Resnet18Backbone(False)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.meta_head=MetaBlock()
        self.meta_head_dim=self.meta_head.meta_emb_dim       
        self.fc=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )

    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out=self.fc(x)
        return out
class Resnet18WithMetaWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(Resnet18WithMetaWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=Resnet18Backbone(True)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        for param in self.backbone.parameters():
            param.requires_grad = False
   
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.ReLU()
            )
        return 
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        return out_reg,out_conf
class Resnet18WithConfBarWithSpeciesPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(Resnet18WithConfBarWithSpeciesPawpularityRegressor,self).__init__()
        self.backbone=Resnet18Backbone(True)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))

   
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,1),
            nn.SiLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,100),
            )
        self.fc_species=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,1),
            )
        return 
    def forward(self,img_in):
        x = self.backbone(img_in)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        out_species=self.fc_species(x)
        return out_reg,out_conf,out_species
    
class Resnet18WithConfBarWithInvConfBarPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(Resnet18WithConfBarWithInvConfBarPawpularityRegressor,self).__init__()
        self.backbone=Resnet18Backbone(True)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
            
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,100),
            nn.ReLU()
            )
        self.fc_inv_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,100),
            nn.ReLU()
            )
        return 
    def forward(self,img_in):
        x = self.backbone(img_in)
        out_inv_conf_bar=self.fc_inv_conf_bar(x)
        out_conf=self.fc_conf_bar(x)
        return out_conf,out_inv_conf_bar
class InceptionV3WithMetaWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(InceptionV3WithMetaWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=InceptionV3RegressorBackbone(True)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.meta_head=MetaBlock()
        self.meta_head_dim=self.meta_head.meta_emb_dim       
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.Sigmoid()
            )
        return 
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        return out_reg,out_conf


class InceptionV3WithMetaPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(InceptionV3WithMetaPawpularityRegressor,self).__init__()
        self.backbone=InceptionV3RegressorBackbone(False)
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.meta_head=MetaBlock()
        self.meta_head_dim=self.meta_head.meta_emb_dim
        
        self.fc=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,128),
            nn.Dropout(0.4),
            nn.Linear(128,128),
            nn.Dropout(0.5),
            nn.Linear(128,1),
            nn.ReLU()
        )

    def forward(self,image_input,meta_input):
        batch_size=image_input.shape[0]
        image_out = self.backbone(image_input)
        meta_out=self.meta_head(meta_input)
        x=torch.cat((image_out,meta_out),dim=1)
        out=self.fc(x)
        return out

class MobileNetV3RegressorBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone=mobilenet_v3_large(True)
        self.emb_dim=960
        self.backbone.Regressor=Identity()
    def forward(self,x):
        x=self.backbone(x)
        return x
class MobileNetV3WithMetaPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(MobileNetV3WithMetaPawpularityRegressor,self).__init__()
        self.backbone=MobileNetV3RegressorBackbone()
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        self.meta_head=MetaBlock()
        self.fc=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.ReLU()
        )
        return 
    def forward(self,img_in,meta_in):
        img_out=self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        x=torch.cat([img_out,meta_out],dim=1)
        x=self.fc(x)
        return x

class MobileNetV3WithMetaWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(MobileNetV3WithMetaWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=MobileNetV3RegressorBackbone()
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.meta_head=MetaBlock()
        self.meta_head_dim=self.meta_head.meta_emb_dim       
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.ReLU()
            )
        return 
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        return out_reg,out_conf

class CustomRegressorBackBone(nn.Module):
    def __init__(self,input_size=224) -> None:
        super(CustomRegressorBackBone,self).__init__()
        self.emb_dim=512
        self.backbone=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(input_size//16,input_size//16),
        )
    def forward(self,x):
        x=self.backbone(x)
        x=x.view((x.shape[0],-1))
        return x
class PawpularityCustomRegressor(nn.Module):
    def __init__(self,backbone_weights=None) -> None:
        super(PawpularityCustomRegressor,self).__init__()
        self.backbone=CustomRegressorBackBone()
        if(backbone_weights is not None):
            self.backbone.load_state_dict(torch.load(backbone_weights))
        self.meta_head=MetaBlock()
        self.fc=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        return 
    def forward(self,img_in,meta_in):
        img_out=self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        x=torch.cat([img_out,meta_out],dim=1)
        x=self.fc(x)
        return x
    
class LargeSwinTransformerBackbone(nn.Module):
    def __init__(self) -> None:
        super(LargeSwinTransformerBackbone,self).__init__()
        self.backbone=create_model(model_name='swin_large_patch4_window7_224',pretrained=True,num_classes=0)
        # self.backbone.head=Identity()
        self.emb_dim=1536
        return 
    def forward(self,x):
        x=self.backbone(x)
        return x
class SmallSwinTransformerBackbone(nn.Module):
    def __init__(self) -> None:
        super(SmallSwinTransformerBackbone,self).__init__()
        self.backbone=create_model(model_name='swin_small_patch4_window7_224',pretrained=True,num_classes=0)
        # self.backbone.head=Identity()
        self.emb_dim=768
        return 
    def forward(self,x):
        x=self.backbone(x)
        return x
    
class SwinTransfromerWithMetaWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self) -> None:
        super(SwinTransfromerWithMetaWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=LargeSwinTransformerBackbone()
        l =[layer for layer in self.backbone.parameters()]
        for layer in l[:316]:
            layer.requires_grad=False
        # for layer in self.backbone.modules[:-5].parameters():
        #     layer.require_grad=False
        self.meta_head=MetaBlock()
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            )
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head.meta_emb_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        return out_reg,out_conf
    
class SwinTransfromerWithMetaWithBinsPawpularityRegressor(nn.Module):
    def __init__(self,bin_increment) -> None:
        super(SwinTransfromerWithMetaWithBinsPawpularityRegressor,self).__init__()
        self.backbone=LargeSwinTransformerBackbone()
        # l =[layer for layer in self.backbone.parameters()]
        # for layer in l[:316]:
        #     layer.requires_grad=False
        # for layer in self.backbone.modules[:-5].parameters():
        #     layer.require_grad=False
        self.meta_head=MetaBlock()
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_bin_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100//bin_increment),
            )
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head.meta_emb_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_bin=self.fc_bin_bar(x)
        return out_reg,out_bin
class SwinTransfromerWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self) -> None:
        super(SwinTransfromerWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=LargeSwinTransformerBackbone()
        l =[layer for layer in self.backbone.parameters()]
        for layer in l[:316]:
            layer.requires_grad=False
        # for layer in self.backbone.modules[:-5].parameters():
        #     layer.require_grad=False
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            )
    def forward(self,img_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        out_reg=self.fc_paw(img_out)
        out_conf=self.fc_conf_bar(img_out)
        return out_reg,out_conf
    
    
class SmallSwinTransfromerWithMetaWithConfidenceBarPawpularityRegressor(nn.Module):
    def __init__(self) -> None:
        super(SmallSwinTransfromerWithMetaWithConfidenceBarPawpularityRegressor,self).__init__()
        self.backbone=SmallSwinTransformerBackbone()
        l =[layer for layer in self.backbone.parameters()]
        for layer in l[:315]:
            layer.requires_grad=False
        # for layer in self.backbone.modules[:-5].parameters():
        #     layer.require_grad=False
        self.meta_head=SmallMetaBlock()
        self.fc_paw=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.ReLU()
            )
        self.fc_conf_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100),
            nn.Sigmoid()
            )
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head.meta_emb_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out_reg=self.fc_paw(x)
        out_conf=self.fc_conf_bar(x)
        return out_reg,out_conf
    

class SwinTransfromerWithMetaPawpularityClassifier(nn.Module):
    def __init__(self) -> None:
        super(SwinTransfromerWithMetaPawpularityClassifier,self).__init__()
        self.backbone=LargeSwinTransformerBackbone()
        l =[layer for layer in self.backbone.parameters()]
        for layer in l[:310]:
            layer.requires_grad=False
        # for layer in self.backbone.modules[:-5].parameters():
        #     layer.require_grad=False
        self.meta_head=MetaBlock()
        self.fc=nn.Sequential(
            torch.nn.Linear(self.backbone.emb_dim+self.meta_head.meta_emb_dim,1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,100)
            )   
        return 
    def forward(self,img_in,meta_in):
        batch_size=img_in.shape[0]
        img_out = self.backbone(img_in)
        meta_out=self.meta_head(meta_in)
        meta_out=meta_out.view((batch_size,self.meta_head.meta_emb_dim))
        x=torch.cat((img_out,meta_out),dim=1)
        out=self.fc(x)
        return out
class SmallSwinTransfromerWithBinsPawpularityClassifier(nn.Module):
    def __init__(self,bin_increment) -> None:
        super(SmallSwinTransfromerWithBinsPawpularityClassifier,self).__init__()
        self.backbone=SmallSwinTransformerBackbone()
        # l =[layer for layer in self.backbone.parameters()]
        # for layer in l[:316]:
        #     layer.requires_grad=False

        self.fc_bin_bar=nn.Sequential(
            nn.Linear(self.backbone.emb_dim,512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.Linear(512,100//bin_increment),
            )
    def forward(self,img_in):
        img_out = self.backbone(img_in)
        out_bin=self.fc_bin_bar(img_out)
        return out_bin
    
    
if(__name__ == '__main__'):
    # model=MobileNetV3RegressorBackbone().cuda()
    # print(224//32)
    img_in=torch.randn((32,3,224,224)).cuda()
    # meta_in=torch.randn((16,1,12)).cuda()
    # print(summary(model,[img_in,meta_in]))
    # print(model(img_in,torch.randn((16,12)).cuda()))
    # model(img_in)
    # model=SmallSwinTransformerBackbone().cuda()
    model=create_model('swin_large_patch4_window7_224',pretrained=True)
    l=[layer for layer in model.parameters()]
    print(summary(model,img_in))
    print(model(img_in).shape)
    print(len(l))