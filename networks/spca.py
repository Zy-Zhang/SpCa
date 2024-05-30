import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast

from .backbone import ResNet, ResNet_STAGE45, ResNet_STAGE4, weights_init, constant_init, pcawhitenlearn_shrinkage
from .RetrievalNet import *


# SpCa: --------

class SpCa(nn.Module):
    
    def __init__(self, outputdim, classifier_num, meta, s = 45, m = 0.2, backbone = 'resnet101'):
        super(SpCa, self).__init__()
        
        self.outputdim = outputdim
        self.mediumdim = 2048
        self.backbone = ResNet(name=backbone, train_backbone=True, dilation_block5=False, pretrained=meta['pretrained'])
        self.pos_branch = Spca_layer(inputdim = self.mediumdim, K = meta['K'], local_dim = meta['local_dim'], multi = meta['multi'], combine = meta['combine'])

        # self.local_embed = None
        if meta['combine'].startswith('cro'):
            self.fuser = Fuser(dim = self.mediumdim, inputdim = meta['K'] * meta['multi'])
            self.whiten = nn.Conv2d(self.mediumdim, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True) # if a batchnorm1d added?
        elif meta['combine'].startswith('fur'):
            self.fuser = Fuser_(dim = self.mediumdim, inputdim = meta['K'] * meta['multi'])
            self.whiten = nn.Conv2d(self.mediumdim, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        elif meta['combine'].startswith('cat'):
            self.fuser = ConCate(dim = self.mediumdim, inputdim = meta['K'] * meta['multi'])
            self.whiten = nn.Conv2d(2*self.mediumdim, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        elif meta['combine'].startswith('had'):
            self.fuser = HadaMard(dim = self.mediumdim, inputdim = meta['K'] * meta['multi'])
            self.whiten = nn.Conv2d(self.mediumdim * meta['K'] * meta['multi'] // 4, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        elif meta['combine'].startswith('orth'):
            self.fuser = Orthogonal(dim = self.mediumdim, inputdim = meta['K'] * meta['multi'])
            self.whiten = nn.Linear(self.mediumdim, outputdim, bias=True)
        elif meta['combine'].startswith('dec'):
            self.fuser = Decoder_c(self.mediumdim, meta['K'] * meta['multi'], 8, True, 0.0, 0.0, 0.0)
            self.whiten = nn.Conv2d(self.mediumdim, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        else:
            print('Unseen Selected Fusion Scheme!!!')
        
        self.pooling = gem(p=3.0)    
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=s, m=m)
        self.meta = meta

    @torch.no_grad()
    def forward_test(self, x):
        
        g_fea = self.backbone(x)
        p_fea, attn = self.pos_branch(g_fea)
        f_fea = self.fuser(g_fea, p_fea)
        if self.meta['combine'].startswith('orth'):
            f_fea = self.whiten(f_fea)
        else:
            f_fea = self.pooling(f_fea)
            f_fea = self.whiten(f_fea).squeeze(-1).squeeze(-1)
        f_fea = F.normalize(f_fea, dim=-1)        
        
        return f_fea

    @torch.no_grad()
    def feature_extract(self, x):

        g_fea = self.backbone(x)
        B,C,H,W = g_fea.size()
        feats = g_fea.permute(0,2,3,1).reshape(B*H*W,C)
        
        return feats.cpu()

    @autocast()
    def forward(self, x, label):
        
        g_fea = self.backbone(x)

        p_fea, attn = self.pos_branch(g_fea)

        f_fea = self.fuser(g_fea, p_fea)
        if self.meta['combine'].startswith('orth'):
            f_fea = self.whiten(f_fea)
        else:
            f_fea = self.pooling(f_fea)
            f_fea = self.whiten(f_fea).squeeze(-1).squeeze(-1)

        global_logits = self.classifier(f_fea, label)
        global_loss = F.cross_entropy(global_logits, label)
        
        return global_loss, global_logits

class Spca_layer(nn.Module):

    def __init__(self, inputdim, K, local_dim, multi = 1, combine = 'cro'):
        super(Spca_layer, self).__init__()
        
        self.clusters = nn.Parameter(torch.randn(1, K, local_dim))
        nn.init.xavier_uniform_(self.clusters)
        self.pi = nn.Parameter(torch.ones(1, K)/K)
        self.cov = nn.Parameter(torch.ones(1, K))
        self.proj_kv = nn.Conv2d(inputdim, local_dim, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.p_norm = nn.LayerNorm(K * multi)
        self.norm_templates = nn.LayerNorm(local_dim)
        self.softmax = nn.Softmax(dim=-1) #nn.Softmax(dim = 1)
        self.K = K
        self.multi = multi
        self.gamma = 1-1e-10 # .9999
        self.iter = 1

    def forward(self, x):
        
        gamma = self.gamma ** (self.iter**(0.6))
        
        x = self.proj_kv(x)
        B,C,H,W = x.size()
        x = x.reshape(B,C,H*W).permute(0,2,1)

        templates = torch.repeat_interleave(self.clusters, B, dim=0)
        pi = torch.repeat_interleave(self.pi, B, dim = 0)
        cov = torch.repeat_interleave(self.cov, B, dim = 0)
        attn = None

        templates_prev = templates
        pi_prev = pi
        cov_prev = cov
        templates = self.norm_templates(templates)
        templates = templates.permute(0,2,1)
            
        sub = x.permute(0,2,1).unsqueeze(3) - templates.unsqueeze(2) # B x D x HW x K
        square_sub = torch.square(sub).sum(dim = 1).squeeze(1) # B x HW x K
        attn_logits = torch.log(self.pi / torch.sqrt(self.cov)) - square_sub/self.cov/2# B x HW x K
        attn = self.softmax(attn_logits) # B x HW x K            

        # pi = pi_prev + gamma * (attn.sum(dim = 1).sum(dim=0, keepdim = True)/(B*H*W)- pi_prev)
        pi = pi_prev + gamma * (attn.sum(dim = 1)/(H*W)- pi_prev)
        # pi /= pi.sum(dim = 1, keepdim = True)
            
        attn_ = attn + 1e-8 # to avoid zero when with the L1 norm below
        attn_ = attn_ / attn_.sum(dim=-2, keepdim=True)  
        # attn_ = attn_ / attn_.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)            
        templates = templates_prev + gamma * (torch.einsum('bld,blk->bkd', x, attn_) - templates_prev)
            
        sub_ = x.permute(0,2,1).unsqueeze(3) - templates.permute(0,2,1).unsqueeze(2)
        square_sub_ = torch.square(sub_).sum(dim = 1).squeeze(1) # B x HW x K
        cov_ = square_sub_*attn_ # B x HW x K
        cov = cov_prev + gamma * (cov_.sum(1).squeeze(1) - cov_prev)

        # build position embedding:
        with torch.no_grad():
            coord = torch.stack((torch.meshgrid(torch.arange(H),torch.arange(W), indexing = 'ij')), dim = 0).reshape(2,-1).permute(1,0)
            x_inner = -2*torch.matmul(coord, coord.transpose(1,0))
            x_square = torch.sum(torch.mul(coord, coord), dim = -1, keepdim = True)
            mask = x_square + x_inner + x_square.transpose(1,0)
            # masks = []
            # for num in range(self.multi):
            #     mask_ = torch.exp(-torch.abs(mask.sqrt()))
            #     masks.append(distance_encoding_(torch.repeat_interleave(mask_.unsqueeze(0), B, dim = 0), beta = num + 1).cuda())
            mask = mask.cuda()
            mask_ = torch.exp(-torch.abs(mask.sqrt()))
            masks = distance_encoding_m(torch.repeat_interleave(mask_.unsqueeze(0), B, dim = 0), beta = self.multi)

        for num in range(self.multi): # we try attn_ instead of attn
            if num == 0:
                OutP = torch.einsum('bnk,bnm->bkm', attn_, masks[num])
            else:
                OutP = torch.cat((OutP, torch.einsum('bnk,bnm->bkm', attn_, masks[num])), dim = 1)

        OutP = self.p_norm(OutP.permute(0,2,1)).permute(0,2,1)
        OutP = OutP.reshape(B, -1, H, W).cuda()
        attn = attn.permute(0,2,1).view(B,self.K,H,W)

        if self.training:
            self.iter = self.iter + 1
        
        return OutP, attn

def distance_encoding_(mask, beta = 1):
    
    B,H,W = mask.size()
    D = torch.repeat_interleave(mask.sum(dim = 1, keepdim = True), H, dim = 1)
    A = mask / D
    mask = A
    for num in range(beta-1):
        mask = torch.matmul(mask,A)
    return mask

def distance_encoding_m(mask, beta = 1):

    B,H,W = mask.size()
    D = torch.repeat_interleave(mask.sum(dim = 1, keepdim = True), H, dim = 1)
    A = mask / D
    mask = A
    mask_ = []
    for num in range(beta):
        mask = torch.matmul(mask,A)
        mask_.append(mask)
    return mask_

# dim reduction --------------------------------------
class ConvDimReduction(nn.Conv2d):
    """Dimensionality reduction as a convolutional layer

    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, (1, 1), padding=0, bias=True)

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)
        self.weight.requires_grad = False

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        self.bias.requires_grad = False
        return m.T, P.T

# feature fusion approaches: -----------------------------------------------------------
class Decoder_c(nn.Module):
    def __init__(self, dim, l_dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.proj = nn.Sequential(nn.Linear(l_dim, dim, bias = True), nn.LayerNorm(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        
        B,C1,H,W = q.size()
        q = q.reshape(B, C1, H * W).permute(0, 2, 1)
        _,C2,H_,W_ = x.size()
        x = x.reshape(B, C2, H_ * W_).permute(0, 2, 1)
        x = self.proj(x)
        
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))

        q = q.permute(0, 2, 1).reshape(B, -1, H, W)
        
        return q

class Fuser(nn.Module):

    def __init__(self, dim, inputdim, num_heads = 8, qkv_bias = True, drop = 0., attn_drop = 0., drop_path = 0.):
        super(Fuser, self).__init__()
        self.cross_attn = Attention(dim = dim, num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.mlp = Mlp(in_features = dim, hidden_features = 2 * dim, out_features = dim, drop = drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.ln = nn.LayerNorm(dim) # ------ original
        self.proj = nn.Linear(inputdim, dim, bias = True)

    def forward(self, q, x):
        # reshape q and x
        B,C1,H,W = q.size()
        q = q.reshape(B, C1, H * W).permute(0, 2, 1)
        _,C2,H_,W_ = x.size()
        x = x.reshape(B, C2, H_ * W_).permute(0, 2, 1)
        
        # cross attention two channels
        q_ln = self.ln(q) # ------ original
        x = self.proj(x)
        q = q + self.drop_path(self.cross_attn(q_ln, x, x))
        
        # batchnorm before mlp
        q = q + self.drop_path(self.mlp(q))
        q = q.permute(0, 2, 1).reshape(B, -1, H, W)
        
        return q

class ConCate(nn.Module):

    def __init__(self, dim, inputdim, drop = 0., drop_path = 0.):
        super(ConCate, self).__init__()
        self.mlp = Mlp(in_features = 2*dim, hidden_features = 4*dim, out_features = 2*dim, drop = drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(inputdim, dim, bias = True)

    def forward(self, q, x):
        B, C1, H, W = q.size()
        q = q.reshape(B, C1, H * W).permute(0,2,1)
        _, C2, _, _ = x.size()
        x = x.reshape(B, C2, H * W).permute(0,2,1)
        x = self.proj(x)
        q = self.bn1(q)
        x = self.bn2(x)
        q = torch.cat((q, x), dim = 2)
        q = q + self.drop_path(self.mlp(q))
        q = q.permute(0,2,1).reshape(B, -1, H, W)
        return q

class HadaMard(nn.Module):

    def __init__(self, dim, inputdim):
        super(HadaMard, self).__init__()
        self.bn1 = nn.LayerNorm(dim // 4)
        self.bn2 = nn.LayerNorm(inputdim)
        self.proj = nn.Linear(dim, dim // 4, bias = True)

    def forward(self, q, x):
        B, C1, H, W = q.size()
        q = q.reshape(B, C1, H * W).permute(0,2,1)
        _, C2, _, _ = x.size()
        x = x.reshape(B, C2, H * W).permute(0,2,1)
        q = self.proj(q)
        q = self.bn1(q)
        x = self.bn2(x)
        q = torch.einsum('bnc,bnd->bncd',q,x)
        B2, N2, H2, W2 = q.size()
        q = q.reshape(B2, N2, H2 * W2)
        q = q.permute(0,2,1).reshape(B, -1, H, W)
        return q

class Orthogonal(nn.Module):

    def __init__(self, dim, inputdim):
        super(Orthogonal, self).__init__()
        self.fc_t = nn.Linear(dim, dim // 2, bias=True)
        self.pool_g = gem(p=3.0)
        self.fc = nn.Conv2d(inputdim, dim // 2, kernel_size=(1, 1), stride=1, padding=0, bias = True)
        self.pool_l = nn.AdaptiveAvgPool2d((1, 1)) 

    def forward(self, g, l):
        fg_o = self.pool_g(g)
        fg_o = fg_o.view(fg_o.size(0), 2048)
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        fl = self.fc(l)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), 1024)

        final_feat=torch.cat((fg, fo), 1)

        return final_feat
