# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import torchvision
from torchvision.models import VGG16_Weights
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

#### TEXTURE LOSS  
'''
Loss Functions used in NTIRE 2023 Challenge 
1. VGG 
2. TextureLossVGG19  
'''

class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(0), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features


class TextureLossVGG19(torch.nn.Module):
    def __init__(self, loss_weight = 1.): 
        super(TextureLossVGG19, self).__init__()
        self.loss_weight = loss_weight 
        
        texture_layers = ['8','17','26','35']
        vgg_layers = [int(i) for i in texture_layers]
        self.vgg_texture = VGG(layers=vgg_layers, replace_pooling = False)

        if torch.cuda.is_available():
            self.vgg_texture = self.vgg_texture.cuda()
    
    
    def forward(self, inp, gt): 
        with torch.no_grad():
            sr_l, sr_r = torch.split(inp,(3,3),dim=1)  
            hr_l, hr_r = torch.split(gt,(3,3),dim=1) 
            
            # we process both the left view and right view one at a time
            # Processing the Left View: 
            vgg_sr = self.vgg_texture.forward(sr_l) 
            vgg_gt = self.vgg_texture.forward(hr_l) 
            text_loss_l = []
            gram_sr = [self.gram_matrix(y) for y in vgg_sr]
            gram_gt = [self.gram_matrix(y) for y in vgg_gt]
            for m in range(0,len(vgg_sr)):
                text_loss_l += [self.criterion(gram_sr[m],gram_gt[m])]
            text_loss_l = sum(text_loss_l)
        
            # Processing the right view: 
            vgg_sr = self.vgg_texture.forward(sr_r) 
            vgg_gt = self.vgg_texture.forward(hr_r) 
            text_loss_r = [] 
            gram_sr = [self.gram_matrix(y) for y in vgg_sr]
            gram_gt = [self.gram_matrix(y) for y in vgg_gt]
            for m in range(0,len(vgg_sr)):
                text_loss_r += [self.criterion(gram_sr[m],gram_gt[m])]
            text_loss_r = sum(text_loss_r)
            
            texture_loss = text_loss_l + text_loss_r 
            texture_loss = texture_loss * self.loss_weight 
        
        return texture_loss
    
        
    
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def criterion(self,a, b):
        return torch.mean(torch.abs((a-b)**2).view(-1))

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, loss_weight=1., resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.loss_weight = loss_weight
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

