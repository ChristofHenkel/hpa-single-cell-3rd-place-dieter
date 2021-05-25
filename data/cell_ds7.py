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
    batch_dict = {'target':batch['target'].to(device),
                  'input':batch['input'].to(device)
                  }
    return batch_dict

def collate_fn(batch):
    
    out_dict = {'input':torch.stack([b['input'] for b in batch]),
               'target':torch.stack([b['target'] for b in batch])
               }
    return out_dict

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

        self.cfg = cfg
        self.maxlen = cfg.maxlen
        self.df = df.copy()
#         self.df['image_id'] = self.df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
#         self.df['cell_id'] = self.df['ID'].apply(lambda x: int(x.split('_')[-1]))
#         self.fns = self.df['ID'].values
        
        self.label_cols = cfg.label_cols
        if not self.label_cols[0] in self.df.columns:
            self.df[self.label_cols] = np.zeros((self.df.shape[0],len(self.label_cols)))
        
            
#         self.df_gr = self.df.groupby('image_id')
        self.image_ids = self.df['image_id'].values
        self.cells = self.df['cells'].values
        self.labels = self.df[self.label_cols].values
        
        self.mode = mode
        self.aug = aug
        self.normalization = cfg.normalization
        self.data_folder = cfg.data_folder
        self.cache_n_img = self.cfg.cache_n_img
        self.cached_img = 0
        self.shuffle = True
        
        
#     def get_sample_weights(self):
        
#         if self.cfg.sample_weighting == 'simple':
#             print('weighting simple')
#             w = 1/self.labels.sum(0)
#             sample_weights = (self.labels * w[None,:]).sum(1)
#         elif self.cfg.sample_weighting == 'log':
#             print('weighting log')
#             w = 1/np.log1p(self.labels.sum(0))
#             sample_weights = (self.labels * w[None,:]).sum(1)
#         elif self.cfg.sample_weighting == 'precomputed':
#             print('weighting precomputed')
#             sample_weights = self.df['weight'].values
#         else:
#             pass
        
#         sample_weights = torch.from_numpy(sample_weights).double()
#         return sample_weights

    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        
#         sub_df = self.df_gr.get_group(image_id)
#         fns = sub_df['ID'].values
        label = self.labels[idx]
        
        cell_ids = self.cells[idx]
        cell_ids = np.array([int(c) for c in cell_ids.split(' ')])
        
        
        if self.shuffle:
            rand_idx = np.random.permutation(np.arange(len(cell_ids)))
            cell_ids = cell_ids[rand_idx]
        
        cell_ids = cell_ids[:self.maxlen]
        
        raw_img = self.load_one(image_id)
        cell_mask = self.load_mask(image_id)
        
        imgs = []
        for cell_id in cell_ids:
            img = self.crop_cell(cell_id,cell_mask,raw_img)
            if self.aug:
                img = self.augment(img)

            img = img.astype(np.float32)
            if self.normalization:
                img = self.normalize_img(img)
            imgs += [self.to_torch_tensor(img)]
            
        # pad
        if len(imgs) < self.cfg.maxlen:
            imgs += [torch.zeros(4,self.cfg.img_size[0],self.cfg.img_size[1])] * (self.cfg.maxlen - len(imgs))
            
        imgs = torch.stack(imgs)
        if self.shuffle:
            rand_idx = np.random.permutation(np.arange(self.cfg.maxlen))
            imgs = imgs[rand_idx]
        
        feature_dict = {'input':imgs,
                       'target':torch.tensor(label).float()
                       }
        return feature_dict

    def crop_cell(self,cell_id,cell_mask,img):
        binary_mask = np.uint8(cell_mask == cell_id)
        x, y, w, h = cv2.boundingRect(binary_mask)
        cropped_img = img[y:y+h,x:x+w,:]
        cropped_mask = binary_mask[y:y+h,x:x+w]
        new_img = np.zeros((h,w,4), dtype=np.uint8)
        new_img[cropped_mask>0] = cropped_img[cropped_mask>0]
        PAD = (np.array(new_img.shape[:2]) * 0.1).astype(np.int32)
        new_img = np.pad(new_img,((PAD[0], PAD[0]), (PAD[1], PAD[1]), (0,0)))
        return new_img
    
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
