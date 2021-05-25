import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn

loss_function = nn.BCEWithLogitsLoss()


# class WeightedSum2d(nn.Module):
#     def __init__(self):
#         super(WeightedSum2d, self).__init__()
#     def forward(self, x):
#         x, weights = x
#         assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
#                 'err: h, w of tensors x({}) and weights({}) must be the same.'\
#                 .format(x.size, weights.size)
#         y = x * weights                                       # element-wise multiplication
#         y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
#         return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
#     def __repr__(self):
#         return self.__class__.__name__


# class Attention(nn.Module):
#     '''
#     SpatialAttention2d
#     2-layer 1x1 conv network with softplus activation.
#     <!!!> attention score normalization will be added for experiment.
#     '''
#     def __init__(self, in_c, act_fn='relu'):
#         super(Attention, self).__init__()
#         self.conv1 = nn.Linear(in_c, 512)                 # 1x1 conv
#         if act_fn.lower() in ['relu']:
#             self.act1 = nn.ReLU()
#         elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
#             self.act1 = nn.LeakyReLU()
#         self.conv2 = nn.Linear(512, 1)                    # 1x1 conv
#         self.softplus = nn.Softplus(beta=1, threshold=20)       # use default setting.

#     def forward(self, x):
#         '''
#         x : spatial feature map. (b x c x w x h)
#         s : softplus attention score 
#         '''
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.softplus(x)
#         return x

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.n_classes = len(cfg.label_cols)
        in_chans = 4
        self.maxlen = cfg.maxlen
        
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool='',in_chans=in_chans)
        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']
#         self.pool = nn.AdaptiveMaxPool2d(1)
        self.head_in_units = backbone_out
        self.head = nn.Linear(self.head_in_units, self.n_classes)
        if cfg.pretrained_weights is not None:
            self.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from',cfg.pretrained_weights)
        self.bce = loss_function
        self.cfg = cfg

#         self.att = Attention(backbone_out)
        self.att = nn.Sequential(nn.Linear(backbone_out, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 1))

    def forward(self, batch):

        x = batch['input']
        m = batch['mask']
        x = self.backbone(x)
        
        m2 = F.interpolate(m[:,None,:,:].float(), size=(x.shape[2],x.shape[3]), mode='nearest')
        m2 = m2.long()
        
        x_new = []
        for i in range(x.shape[0]):
            m1 = m2[i]
            x1 = x[i]
            ml = m1.unique()[1:].long()
            idx = torch.randperm(ml.shape[0])
            ml = ml[idx[:self.maxlen]]
            cell_means = []
            for j in ml:
                mask = (m1 ==j).float()
                x2 = x1 * mask
                x3 = x2.sum((1,2)) / mask.sum((1,2))
                cell_means += [x3]
            if len(cell_means) < self.maxlen:
                cell_means += [torch.zeros(self.head_in_units, device=x.device)] * (self.maxlen - len(cell_means))
            cell_means = torch.stack(cell_means)
            cell_att = self.att(cell_means)
            cell_att = torch.softmax(cell_att,dim=0)
            cell_means = self.head(cell_means)
            cell_means = cell_means * cell_att    
            x_new += [cell_means.sum(0)]
        logits = torch.stack(x_new)
        
    
        loss = self.bce(logits,batch['target'])
#         print(loss)
        return {'logits': logits,
                'loss':loss}

#         return x, m2

