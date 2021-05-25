from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
from tqdm import tqdm

def batch_to_device(batch,device):
    batch_dict = {key:batch[key].to(device) for key in batch}
    return batch_dict

def collate_fn(batch):
    
    out_dict = {'input':torch.stack([b['input'] for b in batch]),
               'target':torch.stack([b['target'] for b in batch]),
                'mask':torch.stack([b['mask'] for b in batch]),
               }
    return out_dict

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

        self.cfg = cfg
        self.df = df.copy()
        self.image_ids = self.df['image_id'].values
        
        self.label_cols = cfg.label_cols
        if not self.label_cols[0] in self.df.columns:
            self.df[self.label_cols] = np.zeros((self.df.shape[0],len(self.label_cols)))
        
        self.labels = self.df[self.label_cols].values
        self.mode = mode
        self.aug = aug
        self.normalization = cfg.normalization
        self.data_folder = cfg.data_folder
        self.cache_n_img = self.cfg.cache_n_img
        self.cached_img = 0
        

    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        label = self.labels[idx]


        img = self.load_one(image_id)   
        mask = self.load_mask(image_id)
        if self.aug:
            img, mask = self.augment(img, mask)

        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)


        img = self.to_torch_tensor(img)
        feature_dict = {'input':img,
                       'target':torch.tensor(label).float(),
                        'mask':torch.tensor(mask.astype(np.int16))
                       }
        return feature_dict

    
    def __len__(self):
        return len(self.image_ids)


    def load_one(self, id_):

        if self.cfg.suffix == 'png':
            imgs = []
            for color in ['red','green','blue','yellow']:
                fp = f'{self.data_folder}{id_}_{color}.png'
                imgs += [cv2.imread(fp, cv2.IMREAD_UNCHANGED)]
            img = np.stack(imgs, axis = -1)
        elif self.cfg.suffix == 'npy':
            img = np.load(f'{self.data_folder}{id_}.npy')
#             print(img.shape)
        else:
            pass
#         except:
#             print("FAIL READING img", id_)
#             img = np.zeros((512,512,4), dtype=np.float32) 
        if img.max() > 256:
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        return img
    
    def load_mask(self, id_):
        mask = cv2.imread(self.cfg.mask_folder + id_ + '.png',cv2.IMREAD_UNCHANGED)
        return mask
    
    def load_cache(self):
        for fn in tqdm(self.fns):
            self.data_cache[fn] = self.load_one(fn)

    def augment(self,img, mask):
        a = self.aug(image=img, mask=mask)
        img_aug = a['image']
        mask_aug = a['mask']

        return img_aug, mask_aug

    def normalize_img(self,img):
        
        if self.normalization == 'channel':
            #print(img.shape)
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + 1e-4
            img = (img - pixel_mean[None,None,:]) / pixel_std[None,None,:]
            img = img.clip(-20,20)
           
        elif self.normalization == 'image':
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20,20)
            
        elif self.normalization == 'simple':
            img = img/255
            
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'min_max':
            img = img - np.min(img)
            img = img / np.max(img)
            return img
        
        else:
            pass
        
        return img
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))
