import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config
from dataset import WeimarDB, WeimarCollated, WeimarSeparate
from model import CRNN, CNN_base
from trainer import CRNNtrainer

import mlflow


def prepare_dataset(config, partition, wrapper_type):

    if wrapper_type == 'separated':
        wrapper = WeimarSeparate
    else:
        wrapper = WeimarCollated

    dataset = wrapper(
        WeimarDB(
            config,
            partition = partition,
            autoload_sfnmf=True, 
            autoload_audio=False
        ), config,
        tag = f'{partition}'
    )
    
    return dataset

def setup_dataset_dict(config, parts, types):
    dataset_dict = {}
    for part, type in zip(parts, types):
        print(f'loading partition: {part}-{type}')
        dataset_dict[f'{part}-{type}'] = prepare_dataset(config, part, type)
    return dataset_dict

def setup_device(config):
    index = config['crnn_trainer']['device']
    return torch.device(f'cuda:{index}' if torch.cuda.is_available() else 'cpu')

def setup_optimizer(config, parameters):
    lr = config['crnn_trainer']['lr']
    return torch.optim.Adam(parameters, lr=lr)

def setup_scheduler(config, optimizer):
    lr = config['crnn_trainer']['lr']
    lr_final = config['crnn_trainer']['lr_final']
    epochs_num = config['crnn_trainer']['epochs_num']
    gamma = np.power(lr_final / lr, 1 / epochs_num)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

def setup_criterion(config):
    label_smoothing = config['crnn_trainer'].get('label_smoothing', 0)
    return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def setup_trainer(config, model):
    parameters = model.parameters()
    dataset_dict = setup_dataset_dict(
        config, 
        ['train', 'test', 'val', 'val'],
        ['collated', 'separated', 'collated', 'separated']
    )
    device = setup_device(config)
    criterion = setup_criterion(config)
    optimizer = setup_optimizer(config, parameters)
    scheduler = setup_scheduler(config, optimizer)

    trainer = CRNNtrainer(
        device = device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataset_dict=dataset_dict,
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
    
    trainer = setup_trainer(config, CNN_base())
    trainer.train()

    mlflow.end_run()