from types import SimpleNamespace
from copy import deepcopy

cfg = SimpleNamespace(**{})

#dataset
cfg.dataset = 'default_dataset'
cfg.val_df = None
cfg.batch_size = 32
cfg.normalization = 'image'
cfg.train_aug = None
cfg.val_aug = None
cfg.test_augs = None
cfg.cache_n_img = 0
cfg.suffix = 'png'
cfg.label_cols = [f't{i}' for i in range(19)]
cfg.eval_label_cols = [f't{i}' for i in range(18)]
cfg.sample_weighting = None
cfg.channels = ['red','green','blue','yellow']
#model
cfg.backbone = 'resnet18'
cfg.dropout = 0
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pool = 'avg'
cfg.train = True

cfg.alpha = 1
cfg.cls_loss_pos_weight = None
cfg.train_val = False
cfg.eval_epochs = 1
cfg.drop_rate=0.
cfg.drop_path_rate=0.

cfg.warmup = 0

#training
cfg.fold = 0
cfg.lr = 1e-4
cfg.schedule = 'cosine'
cfg.weight_decay = 0
cfg.optimizer = 'Adam' # "Adam", "fused_Adam", "SGD", "fused_SGD"
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.simple_eval = False
cfg.do_test = True
cfg.eval_ddp = True
cfg.class_weights =None
cfg.mixup = 1
cfg.clip_grad = 0

#eval
cfg.img_eval = False

#ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 4
cfg.drop_last = True            
#logging,
cfg.tags = None


basic_cfg = cfg











