from default_config import basic_cfg
import albumentations as A
import os

cfg = basic_cfg
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = '/raid/hpa-single-cell-image-classification/'
cfg.data_folder = cfg.data_dir + 'train/'
cfg.train_df = 'train_4folded.csv'
if os.uname()[1] == 'cloe':
    cfg.output_dir = f"/home/christof/kaggle/hpass/models/{os.path.basename(__file__).split('.')[0]}"
else:
    cfg.output_dir = f"/mount/hpass/models/{os.path.basename(__file__).split('.')[0]}"

cfg.lr = 0.001
cfg.epochs = 10
cfg.batch_size = 16
cfg.model = 'baseline_v3'
cfg.backbone = 'tf_efficientnet_b7_ns'
cfg.channels = ['red','green','blue','yellow']
#dataset
cfg.dataset = 'base_ds_img_v3'
cfg.suffix = 'png'

cfg.gpu = 0
cfg.num_workers = 8
cfg.tags = 'img'
cfg.fold = 0
cfg.img_size = (512,512)
# cfg.weight_decay = 0.0001
cfg.normalization = 'channel'
cfg.drop_last=True
cfg.img_eval = True
cfg.label_cols = [f't{i}' for i in range(18)]


cfg.train_aug = A.Compose([A.RandomRotate90(),
                           A.Flip(),
                           A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
#                            A.RandomResizedCrop(cfg.img_size[0], cfg.img_size[1], scale=(0.8, 1.0)),
                           A.Resize(int(cfg.img_size[0]*1.2),int(cfg.img_size[0]*1.2)),
                           A.RandomCrop(cfg.img_size[0], cfg.img_size[0]),

])

cfg.val_aug = A.Compose([
                               A.Resize(int(cfg.img_size[0]*1.2),int(cfg.img_size[0]*1.2)),
                           A.CenterCrop(cfg.img_size[0], cfg.img_size[0]),
                        ])
