import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn

loss_function = nn.BCEWithLogitsLoss()

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.n_classes = len(cfg.label_cols)
        in_chans = len(cfg.channels)
        
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, global_pool=cfg.pool,in_chans=in_chans,
                                        drop_rate=cfg.drop_rate,
                                          drop_path_rate=cfg.drop_path_rate
                                         )
        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']
#         self.pool = nn.AdaptiveMaxPool2d(1)
        self.head_in_units = backbone_out
        self.head = nn.Linear(self.head_in_units, self.n_classes)
        
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.pretrained_weights is not None:
            self.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from',cfg.pretrained_weights)
        self.bce = loss_function

    def forward(self, batch):

        x = batch['input']
        x = self.backbone(x)
        x = self.dropout(x)
        logits = self.head(x)
    
        loss = self.bce(logits,batch['target'])
    
        return {'logits': logits,
                'loss':loss}



