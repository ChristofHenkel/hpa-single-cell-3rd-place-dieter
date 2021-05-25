import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import math

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler


from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import pandas as pd
import cv2

cv2.setNumThreads(0)


sys.path.append('configs')
sys.path.append('models')
sys.path.append('data')
sys.path.append('losses')
sys.path.append('utils')


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-f", "--fold",type=int , default=-1, help="fold")
parser_args, _ = parser.parse_known_args(sys.argv)


cfg = importlib.import_module(parser_args.config).cfg


if parser_args.fold > -1:
    cfg.fold = parser_args.fold

os.makedirs(str(cfg.output_dir + f'/fold{cfg.fold}/'), exist_ok=True)

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device



def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_train_dataset(train_df, cfg):
    print("Loading train dataset")
    
    train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    return train_dataset

def get_train_dataloader(train_ds, cfg):

    sampler = None
    if cfg.sample_weighting is not None:
        sampler = WeightedRandomSampler(train_dataset.sample_weights, len(train_dataset.sample_weights))
        print('weighting2')
    train_dataloader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  shuffle=(sampler is None),
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  collate_fn= tr_collate_fn,
                                  drop_last = cfg.drop_last
                                 )
    print(f"train: dataset {len(train_dataset)}, dataloader {len(train_dataloader)}")
    return train_dataloader

def get_val_dataset(val_df, cfg):
    print("Loading val dataset")
    val_dataset = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode='val')
    return val_dataset

def get_val_dataloader(val_dataset, cfg):

    sampler = SequentialSampler(val_dataset)

    val_dataloader = DataLoader(val_dataset,
#                                   shuffle=False,
                                  sampler=sampler,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                collate_fn= val_collate_fn
                                
                                 )
    print(f"valid: dataset {len(val_dataset)}, dataloader {len(val_dataloader)}")
    return val_dataloader



def get_model(cfg):
    Net = importlib.import_module(cfg.model).Net
    return Net(cfg)

def get_optimizer(model, cfg):

    params = [{"params": [param for name, param in model.named_parameters()], "lr": cfg.lr,"weight_decay":cfg.weight_decay}]
    
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=params[0]["lr"])
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(params,lr=params[0]["lr"],momentum=0.9,nesterov=True)
    elif cfg.optimizer == "fused_SGD":
        import apex

        optimizer = apex.optimizers.FusedSGD(params, lr=params[0]["lr"], momentum=0.9, nesterov=True)
    elif cfg.optimizer == "fused_Adam":
        import apex

        optimizer = apex.optimizers.FusedAdam(params, lr=params[0]["lr"], weight_decay=params[0]["weight_decay"],)
        print("wd ",optimizer.param_groups[0]["weight_decay"])

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):
        

    if cfg.schedule == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs_step * (total_steps // cfg.batch_size) , gamma=0.5)
    elif cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) ,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) ,
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) ,
        )
        
        print("num_steps", (total_steps // cfg.batch_size) )

        
        
    else:
        scheduler = None

    return scheduler


import numpy as np
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc



def run_simple_eval(model, val_dataloader, cfg, pre='val'):
    
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []

    for data in tqdm(val_dataloader):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output['loss']]

    val_losses = torch.stack(val_losses)

    val_losses = val_losses.cpu().numpy()

    val_loss = np.mean(val_losses)


    print("Mean val_loss", np.mean(val_losses))



    return val_loss


def run_img_eval(model, val_dataloader, cfg, pre='val'):
    
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []
    val_preds = []
    val_targets = []
    for data in tqdm(val_dataloader):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output['loss']]
        val_preds += [output['logits']]
        val_targets += [batch['target']]
        
    val_losses = torch.stack(val_losses)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    
      

    val_losses = val_losses.cpu().numpy()



    val_loss = np.mean(val_losses)


    print("Mean val_loss", np.mean(val_losses))

    val_preds = val_preds.cpu().numpy().astype(np.float32)
    val_targets = val_targets.cpu().numpy().astype(np.float32)

    rocs = [fast_auc(val_targets[:,i], val_preds[:,i]) for i in range(len(cfg.eval_label_cols))]

    avg_roc = np.mean(rocs)



    return val_loss

def run_eval(model, val_dataloader, cfg, pre="val"):
    
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []
    val_preds = []
    val_targets = []
    val_seg_losses = []
    val_cls_losses = []
    for data in tqdm(val_dataloader):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output['loss']]
        val_preds += [output['logits'].sigmoid()]
        val_targets += [batch['target']]
        
        if 'seg_loss' in output.keys():
            val_cls_losses += [output['cls_loss']]
            val_seg_losses += [output['seg_loss']]
            
        
        
    val_losses = torch.stack(val_losses)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    


    val_losses = val_losses.cpu().numpy()
    val_loss = np.mean(val_losses)

    val_preds = val_preds.cpu().numpy().astype(np.float32)
    val_targets = val_targets.cpu().numpy().astype(np.float32)
    rocs = [fast_auc(val_targets[:,i], val_preds[:,i]) for i in range(len(val_dataloader.dataset.label_cols))]

    avg_roc = np.mean(rocs)

    print(f"{pre}_loss", val_loss)
    print(f"{pre}_avg_roc", avg_roc)



    return val_loss





def create_checkpoint(model, optimizer, epoch, scheduler =None, scaler=None):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,          
        }

        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        return checkpoint



if __name__ == "__main__":
    
    #set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)  

    
    cfg.local_rank = 0
    cfg.world_size = 1
    rank = 0  # global rank

    device = "cuda"


    #setup dataset
    train = pd.read_csv(cfg.data_dir + cfg.train_df)

    val_df = train[train['fold'] == cfg.fold]
    train_df = train[train['fold'] != cfg.fold]

    train_dataset = get_train_dataset(train_df,cfg)
    val_dataset = get_val_dataset(val_df,cfg)
    

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    model = get_model(cfg)
    model.to(device)

    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()


    if cfg.simple_eval:
        run_eval = run_simple_eval
    elif cfg.img_eval:
        run_eval = run_img_eval
    step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                step += cfg.batch_size


                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue

                model.train()
                torch.set_grad_enabled(True)

                

                # Forward pass

                batch = batch_to_device(data,device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict['loss']


                losses.append(loss.item())

                # Backward pass
                
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if step % cfg.batch_size == 0:


                    progress_bar.set_description(
                        f"loss: {np.mean(losses[-10:]):.2f}"
                    )


        if (epoch+1) % cfg.eval_epochs == 0 or (epoch+1) == cfg.epochs:

            val_loss = run_eval(model, val_dataloader, cfg)

        else:
            val_score = 0


        if val_loss < best_val_loss:
            print(f'SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5}')

            checkpoint = create_checkpoint(model, 
                                        optimizer, 
                                        epoch, 
                                        scheduler=scheduler, 
                                        scaler=scaler)

            torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best.pth")
            best_val_loss = val_loss



    #END of training
    
    
    print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
    checkpoint = create_checkpoint(model, 
                                   optimizer, 
                                   epoch, 
                                   scheduler=scheduler, 
                                   scaler=scaler)

    torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last.pth")


