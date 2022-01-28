# sudo apt-get install libcudart10.1
import torch
from torch.utils.data import DataLoader
from build_vocab import Vocabulary
import numpy as np
import time
import sys
import os
from torchvision import transforms
from models import utils, caption

from configuration import Config
from engine import train_one_epoch, evaluate
from data_loader import get_train_loader
import pickle

transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(244),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

with open("./vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)


def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, criterion = caption.build_model(config)
    model.to(device)

    

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

 
    data_loader_train = get_train_loader("./dataset/Flickr8k_Dataset", "./dataset/Flickr8k.token.txt", "./dataset/Flickr_8k.trainImages.txt", vocab,
                             transform, config.batch_size,
                             shuffle=True, num_workers=2, cpi=5)

    data_loader_val = get_train_loader("./dataset/Flickr8k_Dataset", "./dataset/Flickr8k.token.txt", "./dataset/Flickr_8k.devImages.txt", vocab,
                             transform, config.batch_size,
                             shuffle=True, num_workers=2, cpi=5)


    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1
    
    if config.load_pretrained_weights:
        # load pretrained state dict
        print("Loading pre-trained weights...")
        checkpoint = torch.load("./checkpoint_ft.pth", map_location='cpu')
        pre_trained_model_weights = checkpoint['model']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # config.start_epoch = checkpoint['epoch'] + 1

        model = caption.load_transformer_weights(model, pre_trained_model_weights, config)
    


    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, "./checkpoint_" + str(epoch) + ".pth")

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        print()


if __name__ == "__main__":
    config = Config()
    main(config)
