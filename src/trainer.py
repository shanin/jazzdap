import torch
import numpy as np
import mlflow
from tqdm import tqdm


class CRNNtrainer:
    def __init__(
        self,
        model = None,
        optimizer = None,
        scheduler = None,
        dataloader_dict = None,
        criterion = None,
        device = None,
        config = None
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader_dict
        self.scheduler = scheduler
        self._parse_config(config)
        self.step_counter = 0
    
    def _parse_config(self, config):
        self.epochs_num = config['crnn_trainer']['epochs_num']
        self.validation_period = config['crnn_trainer']['validation_period']

    def train_epoch(self):
        self.model.train()
        for x, y in self.dataloader['train']:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            mlflow.log_metric('train_loss', loss.item(), self.step_counter)
            self.step_counter += 1    
    
    def calculate_validation_loss(self):
        self.model.eval()
        loss_batches = []
        with torch.no_grad():
            for x, y in self.dataloader['val']:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss_batches.append(loss.item())
        mlflow.log_metric('val_loss', np.mean(loss_batches))
        self.model.train()
            
    def train(self):
        self.step_counter = 0
        for epoch_num in tqdm(range(self.epochs_num)):
            self.train_epoch()
            self.calculate_validation_loss()
