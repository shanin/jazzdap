import argparse

import numpy as np
import torch

from utils import load_config
from dataset import WeimarDB, WeimarCollated, WeimarSeparate
from model import CRNN
from trainer import CRNNtrainer

import mlflow


def prepare_dataset(config, partition, wrapper_type, feature_type, data_tag):

    if wrapper_type == 'separated':
        wrapper = WeimarSeparate
    else:
        wrapper = WeimarCollated

    if feature_type == 'sfnmf':
        load_sfnmf, load_crepe = True, False
    elif feature_type == 'crepe':
        load_sfnmf, load_crepe = False, True

    dataset = wrapper(
        WeimarDB(
            config,
            partition = partition,
            load_sfnmf=load_sfnmf,
            load_crepe=load_crepe, 
            load_audio=False
        ), config,
        tag = f'{partition}-{feature_type}-{data_tag}'
    )
    
    return dataset

def setup_dataset_dict(config, parts, types, feature_type):
    data_tag = config['crnn_trainer'].get('data_tag', 'default')
    dataset_dict = {}
    for part, type in zip(parts, types):
        print(f'loading partition: {part}-{feature_type}-{type}')
        dataset_dict[f'{part}-{type}'] = prepare_dataset(config, part, type, feature_type, data_tag)
    return dataset_dict

def setup_device(config):
    index = config['crnn_trainer']['device']
    return torch.device(f'cuda:{index}' if torch.cuda.is_available() else 'cpu')

def setup_optimizer(config, parameters):
    lr = config['crnn_trainer']['lr']
    weight_decay = config['crnn_trainer'].get('weight_decay', 0)
    mlflow.log_param('weight_decay', weight_decay)
    return torch.optim.Adam(parameters, lr=lr, weight_decay = weight_decay)

def setup_scheduler(config, optimizer):
    lr = config['crnn_trainer']['lr']
    lr_final = config['crnn_trainer']['lr_final']
    epochs_num = config['crnn_trainer']['epochs_num']
    gamma = np.power(lr_final / lr, 1 / epochs_num)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

def setup_criterion(config):
    label_smoothing = config['crnn_trainer'].get('label_smoothing', 0)
    mlflow.log_param('label_smoothing', label_smoothing)
    return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def setup_trainer(config, 
                  model, 
                  partitions = ['train', 'test', 'val', 'val'],
                  modes = ['collated', 'separated', 'collated', 'separated'],
                  feature_type = 'sfnmf',
                  test_time_dataset = None):
    parameters = model.parameters()
    if test_time_dataset:
        dataset_dict = test_time_dataset
    else:
        dataset_dict = setup_dataset_dict(config, partitions, modes, feature_type)
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
    
    trainer = setup_trainer(config, CRNN(config))
    trainer.train()

    mlflow.end_run()
