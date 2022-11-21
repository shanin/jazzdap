import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config
from dataset import WeimarDB, WeimarSFWrapper
from model import CRNN
from trainer import CRNNtrainer

import mlflow


def prepare_dataloader(config, partition):
    
    batch_size = config['crnn_trainer']['batch_size']
    
    data = WeimarSFWrapper(
        WeimarDB(
            config,
            partition = partition,
            autoload_sfnmf=True, 
            autoload_audio=False
        ), config
    )
    
    dataloader = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return dataloader

def setup_dataloader_dict(config, partitions):
    dataloader_dict = {}
    for part in partitions:
        dataloader_dict[part] = prepare_dataloader(config, part)
    return dataloader_dict

def setup_device(config):
    index = config['crnn_trainer']['device']
    return torch.device(f'cuda:{index}' if torch.cuda.is_available() else 'cpu')

def setup_optimizer(config, parameters):
    lr = config['crnn_trainer']['lr']
    weight_decay = config['crnn_trainer']['weight_decay']
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def setup_scheduler(config, optimizer):
    lr = config['crnn_trainer']['lr']
    lr_final = config['crnn_trainer']['lr_final']
    epochs_num = config['crnn_trainer']['epochs_num']
    gamma = np.power(lr_final / lr, 1 / epochs_num)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

def setup_criterion(config):
    label_smoothing = config['crnn_trainer'].get('label_smoothing', 0)
    return torch.nn.CrossEntropyLoss(label_smoothing)

def setup_trainer(config):
    model = CRNN()
    parameters = model.parameters()
    dataloader_dict = setup_dataloader_dict(config, ['train', 'val'])
    device = setup_device(config)
    criterion = setup_criterion(config)
    optimizer = setup_optimizer(config, parameters)
    scheduler = setup_scheduler(config, optimizer)

    trainer = CRNNtrainer(
        device = device,
        model=model,
        optimitzer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataloader_dict=dataloader_dict,
        scheduler=scheduler,
        config=config
    )

    return trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    mlflow.start_run()
    
    trainer = setup_trainer(config)
    trainer.train()
