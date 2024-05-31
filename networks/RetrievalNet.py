import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
from .backbone import ResNet, ResNet_STAGE45, ResNet_STAGE4, weights_init, constant_init
import numpy as np

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor):
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor):
        return drop_path(x, self.drop_prob, self.training)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        nn.init.constant_(self.proj.weight.data, 0.0)
        nn.init.constant_(self.proj.bias.data, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        B_q, N_q, _ = q.size()
        B_k, N_k, _ = k.size()
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        q = (attn @ v).transpose(1, 2).reshape(q.size(0), q.size(2), -1)
        q = self.proj_drop(self.proj(q))
        return q


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn = nn.BatchNorm1d(dim)
        self.mlp = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, d = x.size()
        x = x + self.drop_path(self.attn(x, x, x))
        x_bn = self.bn(x.reshape(b * n, d)).reshape(b, n, d)
        x = x + self.drop_path(self.mlp(x_bn))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
        return q


class Token_Refine(nn.Module):
    def __init__(self, num_heads, num_object, mid_dim=1024, encoder_layer=1, decoder_layer=2, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_object, mid_dim))
        self.token_norm = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.LayerNorm(mid_dim))
        self.encoder = nn.ModuleList([Encoder(mid_dim, num_heads, qkv_bias, drop, attn_drop, drop_path) for _ in range(encoder_layer)])
        self.decoder = nn.ModuleList([Decoder(mid_dim, num_heads, qkv_bias, drop, attn_drop, drop_path) for _ in range(decoder_layer)])
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=mid_dim, kernel_size=(1, 1), stride=1, padding=0), nn.BatchNorm2d(mid_dim))
        self.mid_dim = mid_dim
        self.proj = nn.Sequential(nn.Linear(in_features=mid_dim * num_object, out_features=1024), nn.BatchNorm1d(1024))

    def forward(self, x: Tensor):
        B, _, H, W = x.size()
        x = self.conv(x).reshape(B, self.mid_dim, H * W).permute(0, 2, 1)
        for encoder in self.encoder:
            x = encoder(x)
        q = self.query.repeat(B, 1, 1)  # B x num_object x mid_dim
        attns = F.softmax(torch.bmm(q, x.permute(0, 2, 1)), dim=1)  # b x num_object x (H x W)
        token = torch.bmm(attns, x)
        token = self.token_norm(token)
        for decoder in self.decoder:
            token = decoder(token, x)
        token = self.proj(token.reshape(B, -1))
        return token


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, eps=1e-6):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.threshold = math.pi - self.m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, dim=-1), F.normalize(self.weight, dim=-1))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros(cos_theta.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot).bool()

        output = torch.cos(torch.where(selected, theta + self.m, theta))
        output *= self.s
        return output

class ArcFace_Delg(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, eps=1e-6):
        super(ArcFace_Delg, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._s = s
        self._m = m
        self.eps = eps

        self.cos_m = torch.tensor(math.cos(self._m), dtype = torch.float16)
        self.sin_m = torch.tensor(math.sin(self._m), dtype = torch.float16)
        self.threshold = torch.tensor(math.cos(math.pi - self._m), dtype = torch.float16)
        self.mm = torch.tensor(math.sin(math.pi - self._m) * self._m, dtype = torch.float16)

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2)).type(torch.float16)
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            self.t = self.t.type(torch.float16)
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_features, self.out_features, self._s, self._m
        )


class RetrievalNet(nn.Module):
    def __init__(self, classifier_num):
        super().__init__()
        outputdim = 1024
        self.outputdim = 1024
        self.backbone = ResNet(name='resnet101', train_backbone=True, dilation_block5=False)
        self.tr = Token_Refine(num_heads=8, num_object=4, mid_dim=outputdim, encoder_layer=1, decoder_layer=2)
        # self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)
        self.classifier = ArcFace_Delg(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)

    # @autocast()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.tr(x)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    # @autocast()
    def forward(self, x: Tensor, label):
        x = self.backbone(x)
        global_feature = self.tr(x)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits

#----------------------------------------------------------------------------------------------------------------------------------
# Token
class Token(nn.Module):
    def __init__(self, outputdim=1024, classifier_num=81313, pretrained = 'filip', backbone = 'resnet101'):
        super().__init__()
        self.outputdim = 1024
        self.backbone = ResNet(name=backbone, train_backbone=True, dilation_block5=False, pretrained = pretrained)
        self.tr = Token_Refine(num_heads=8, num_object=4, mid_dim=outputdim, encoder_layer=1, decoder_layer=2)
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)
        self.meta = {}
        self.meta['outputdim'] = outputdim

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.tr(x)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    # @autocast()#torch.no_grad()
    def forward(self, x, label):
        x = self.backbone(x)
        x = self.tr(x)
        global_logits = self.classifier(x, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits, global_loss

#----------------------------------------------------------------------------------------------------------------------------------
# GeM

class gem(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(gem, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

class GeM(nn.Module):
    def __init__(self, outputdim, classifier_num, meta = None, s = 32, m = 0.15, backbone = 'resnet101'):
        super(GeM, self).__init__()
        self.backbone = ResNet(name=backbone, train_backbone=True, dilation_block5=False, pretrained=meta['pretrained'])
        self.pooling = gem()
        self.whiten = nn.Conv2d(outputdim, 2048, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = outputdim
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s = s, m = m)#s=math.sqrt(self.outputdim), m=0.2)
        self.meta = meta

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        # x = F.normalize(x, p=2.0, dim=1)
        x = self.whiten(x).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    @autocast()
    def forward(self, x, label):
        x = self.backbone(x)
        global_feature = self.pooling(x)
        # global_feature = F.normalize(x, p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits, global_loss

#--------------------------------------------------------------------------------------------------------------------------------------
# Pos-OC:

class spoc(nn.Module):
    def __init__(self, eps=1e-6):
        super(spoc, self).__init__()
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps), (x.size(-2), x.size(-1)))

class DOLG(nn.Module):
    """ DOLG model """
    def __init__(self, pretrained = None, with_aspp = False, backbone = 'resnet101'):
        super(DOLG, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2d((1, 1)) 
        self.pool_g = gem(p=3.0) 
        self.fc_t = nn.Linear(2048, 1024, bias=True)
        self.fc = nn.Linear(2048, 512, bias=True)
        self.globalmodel = ResNet_STAGE45(name=backbone, train_backbone=True, dilation_block5=False, pretrained = pretrained)
        self.localmodel = SpatialAttention2d(1024, with_aspp = with_aspp)
        self.desc_cls = ArcFace(in_features=512, out_features=81313, s=30, m=0.15)
        self.outputdim = 512
        self.meta = {}
        self.meta['outputdim'] = 512

    @autocast()
    def forward(self, x, targets):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), 2048)
        
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), 1024)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)

        global_logits = self.desc_cls(global_feature, targets)
        global_loss = F.cross_entropy(global_logits, targets)
        return global_loss, global_logits, global_loss

    def forward_test(self, x):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), 2048)
        
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), 1024)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)
        
        return F.normalize(global_feature, dim=-1)

class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, act_fn='relu', with_aspp=False):
        super(SpatialAttention2d, self).__init__()
        
        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(1024)
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module 
    '''
    def __init__(self, in_c, mid_c = 512, out_c = 1024):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, mid_c, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, mid_c, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, mid_c, 1, 1),
                                     nn.ReLU())
        conv_after_dim = mid_c * (len(self.aspp)+1)
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, out_c, 1, 1), nn.ReLU())
        
        for dilation_conv in self.aspp:
            dilation_conv.apply(init_weights)
        for model in self.im_pool:
            if isinstance(model, nn.Conv2d):
                model.apply(init_weights)
        for model in self.conv_after:
            if isinstance(model, nn.Conv2d):
                model.apply(init_weights)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x

def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

class SOABlock_GeM(nn.Module):
    def __init__(self, in_ch, k):
        super(SOABlock_GeM, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.g = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))
        self.pooling = gem(p=3.0)

        for conv in [self.f, self.g, self.h]:
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def forward(self, x: Tensor):
        B, C, H, W = x.size()

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W
        attn = F.softmax((self.mid_ch**-.50) * z, dim=-1)
        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W
        z = self.v(z)
        z = z + x
        z = self.pooling(z)
        return z


class SOLAR(nn.Module):
    def __init__(self, outputdim, classifier_num, s = 32, m = 0.15, backbone = 'resnet101'):
        super(SOLAR, self).__init__()
        self.backbone = ResNet(name=backbone, train_backbone=True, dilation_block5=False)
        self.pooling = SOABlock_GeM(in_ch=2048, k=2)
        self.whiten = nn.Conv2d(self.backbone.outputdim_block5, 2048, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = outputdim
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m = m)
        self.meta = {}
        self.meta['outputdim'] = outputdim

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = F.normalize(x, p=2.0, dim=1)
        x = self.whiten(x).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    @autocast()
    def forward(self, x, label):
        x = self.backbone(x)
        x = self.pooling(x)
        global_feature = F.normalize(x, p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits, global_loss

def freeze_all_but_bn(m):
    
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def freeze_all(m):

    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(False)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(False)
