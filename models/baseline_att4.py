import timm
import torch
from torch.nn import functional as F
from torch import nn
from torch.distributions import Beta
import numpy as np

import math
import timm





class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.n_classes = len(cfg.label_cols)
        in_chans = 4
        
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool='avg',in_chans=in_chans)
        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']

        

#         self.pool = nn.AdaptiveMaxPool2d(1)
        self.head_in_units = backbone_out
        self.head = nn.Linear(self.head_in_units, self.n_classes)
        self.att = nn.Sequential(nn.Linear(self.head_in_units, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 1))
        
        if cfg.pretrained_weights is not None:
            self.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from',cfg.pretrained_weights)

        self.bce = nn.BCEWithLogitsLoss()
#         self.mixup = Mixup(cfg.mixup)

    def forward(self, batch):

        x = batch['input']
        y = batch['target']
        bs, n, c, w, h = x.shape
        x = x.reshape(bs*n,c,w,h)
        x = self.backbone(x)
        cell_logits = x.reshape(bs,n,-1)
        x_att = torch.softmax(self.att(cell_logits),dim=1)
        x = (cell_logits * x_att).sum(1)
        img_logits = self.head(x)
        loss = self.bce(img_logits,y)

        return {'logits': img_logits,
                'cell_logits': cell_logits,
                'loss':loss}

