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


tr_collate_fn = None
val_collate_fn = None

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

        self.cfg = cfg
        self.df = df.copy()
        self.fns = (self.df['ID'] + '_' + self.df['cell_id'].astype(str)).values
        self.label_cols = cfg.label_cols
        if not self.label_cols[0] in self.df.columns:
            self.labels = np.zeros((self.df.shape[0],len(self.label_cols)))
        else:
            self.labels = self.df[self.label_cols].values
        self.mode = mode
        self.aug = aug
        self.normalization = cfg.normalization
        self.data_folder = cfg.data_folder
        self.cache_n_img = self.cfg.cache_n_img
        self.cached_img = 0
        

    def __getitem__(self, idx):
        
        fn = self.fns[idx]
        label = self.labels[idx]
        
        
        if (self.cache_n_img > 0) & (self.cached_img < self.cache_n_img):
            try:
                img = self.data_cache[fn] 
            except:
                img = self.load_one(fn)
                if self.cached_img < self.cache_n_img:
                    self.data_cache[fn] = img
                    self.cached_img += 1
        else:
            img = self.load_one(fn)
            
        
        if self.aug:
            img = self.augment(img)
        
        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)

    
        img_tensor = self.to_torch_tensor(img)

        
        feature_dict = {'input':img_tensor,
                       'target':torch.tensor(label).float()
                       }
        return feature_dict

    
    def __len__(self):
        return len(self.fns)


    def load_one(self, id_):

        if self.cfg.suffix == 'png':
            imgs = []
            for color in self.cfg.channels:
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
            if len(img.shape) == 2:
                img = img[:,:,None]
        return img
    
    def load_cache(self):
        for fn in tqdm(self.fns):
            self.data_cache[fn] = self.load_one(fn)

    def augment(self,img):
        a = self.aug(image=img)
        img_aug = a['image']

        return img_aug 

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
