# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm

from models import utils
from  torch.cuda.amp import GradScaler, autocast


scaler = GradScaler()


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)
    i = 1
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks, ids in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1], ids, False)
                loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            scaler.scale(loss).backward()

    
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            
            scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
           
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'epoch_loss': epoch_loss/i})
            pbar.update(1)
            i += 1

    return epoch_loss / total

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks, ids in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1], ids)
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)
        
    return validation_loss / total