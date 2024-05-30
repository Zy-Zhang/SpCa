from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from utils import resnet_block_dilation

import math
import numpy as np


## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def extract_features_from_e2e(model):
    state_dict_features = OrderedDict()
    for key, value in model['state_dict'].items():
        if key.startswith('features'):
            state_dict_features[key[9:]] = value
    return state_dict_features

def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T

class ResNet(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool, pretrained = None):
        super(ResNet, self).__init__()
        if pretrained is None or pretrained.startswith('filip'):
            net_in = getattr(torchvision.models, name)(pretrained = False)
        elif pretrained.startswith('v1'):
            net_in = getattr(torchvision.models, name)(pretrained = pretrained)
        elif pretrained.startswith('v2'):
            from torchvision.models import ResNet101_Weights, ResNet50_Weights
            if name.startswith('resnet101'):
                net_in = getattr(torchvision.models, name)(weights = ResNet101_Weights.DEFAULT)
            else:
                net_in = getattr(torchvision.models, name)(weights = ResNet50_Weights.DEFAULT)
        else:
            raise ValueError('Unsupported or unknown model initialization mode: {}!'.format(pretrained))
        # if v2 and pretrained:
        #     from torchvision.models import ResNet101_Weights
        #     net_in = getattr(torchvision.models, name)(weights = ResNet101_Weights.DEFAULT)
        # else:
        #     net_in = getattr(torchvision.models, name)(pretrained = pretrained) # was False by default
        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        
        if pretrained is not None and pretrained.startswith('filip'):
            state = torch.load('/g/data/rw5/zz1363/models/imagenet-caffe-resnet101-features-10a101d.pth')
            features.load_state_dict(state)
        
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class ResNet_STAGE45(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool, pretrained = None):
        super(ResNet_STAGE45, self).__init__()
        
        if pretrained is None or pretrained.startswith('filip'):
            net_in = getattr(torchvision.models, name)(pretrained = False)
        elif pretrained.startswith('v1'):
            net_in = getattr(torchvision.models, name)(pretrained = pretrained)
        elif pretrained.startswith('v2'):
            from torchvision.models import ResNet101_Weights
            net_in = getattr(torchvision.models, name)(weights = ResNet101_Weights.DEFAULT)
        else:
            raise ValueError('Unsupported or unknown model initialization mode: {}!'.format(pretrained))

        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x4)
        return x4, x5

class ResNet_STAGE4(nn.Module):
    def __init__(self, name: str, train_backbone: bool, pretrained = None):
        super(ResNet_STAGE4, self).__init__()
        
        if pretrained is None or pretrained.startswith('filip'):
            net_in = getattr(torchvision.models, name)(pretrained = False)
        elif pretrained.startswith('v1'):
            net_in = getattr(torchvision.models, name)(pretrained = pretrained)
        elif pretrained.startswith('v2'):
            from torchvision.models import ResNet101_Weights
            net_in = getattr(torchvision.models, name)(weights = ResNet101_Weights.DEFAULT)
        else:
            raise ValueError('Unsupported or unknown model initialization mode: {}!'.format(pretrained))

        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        # self.block5 = features[7]
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x4 = self.block4(x)
        return x4

class ResNet_S4(nn.Module):
    def __init__(self, name: str, train_backbone: bool, pretrained = None):
        super(ResNet_S4, self).__init__()
        
        if pretrained is None or pretrained.startswith('filip'):
            net_in = getattr(torchvision.models, name)(pretrained = False)
        elif pretrained.startswith('v1'):
            net_in = getattr(torchvision.models, name)(pretrained = pretrained)
        elif pretrained.startswith('v2'):
            from torchvision.models import ResNet101_Weights
            net_in = getattr(torchvision.models, name)(weights = ResNet101_Weights.DEFAULT)
        else:
            raise ValueError('Unsupported or unknown model initialization mode: {}!'.format(pretrained))

        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.block5 = features[7]
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x5 = self.block5(x)
        return x5

class ResNet_DOLG(nn.Module):
    """ResNet model."""

    def __init__(self):
        super(ResNet_DOLG, self).__init__()
        self._construct()
        self.apply(init_weights)

    def _construct(self):
        g, gw = 1, 64
        (d1, d2, d3, d4) = (3, 4, 23, 3)
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        return x3, x4

class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun('bottleneck_transform')
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]

class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1)
        self.a = nn.Conv2d(w_in, w_b, 1, stride=s1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = True
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()
