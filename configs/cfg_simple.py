from default_config import basic_cfg
import albumentations as A
import os

cfg = basic_cfg
cfg.name = os.path.basename(__file__).split(".")[0]

#ADJUST PATHS
cfg.data_dir = './input/'
cfg.mask_folder = './input/cell_masks_v5/train/'
cfg.output_dir = f"./output/models/{os.path.basename(__file__).split('.')[0]}"


cfg.data_folder = cfg.data_dir + 'train/'
cfg.train_df = 'train_4folded_cells.csv'
cfg.lr = 0.0001
cfg.epochs = 10
cfg.batch_size = 16
cfg.model = 'baseline_att4'
cfg.backbone = 'seresnext26t_32x4d'
cfg.maxlen = 16
#dataset
cfg.dataset = 'cell_ds7'
# cfg.suffix = 'npy'

cfg.suffix = 'png'
cfg.gpu = 0
cfg.num_workers = 16
cfg.tags = 'debug'
cfg.fold = 0
cfg.sample_weighting = None
cfg.img_size = (256,256)
# cfg.weight_decay = 0.0001
cfg.normalization = 'channel'
cfg.drop_last=True
cfg.img_eval = True
cfg.label_cols = [f't{i}' for i in range(19)]


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
